"""
src/runners/completion_runner.py

Reads JSONL datasets, calls a ModelProvider for each sample, and writes
a structured completions record to results/completions.jsonl.

Each output record includes:
  - prompt_id, category, model, provider
  - timestamp (ISO 8601 UTC)
  - prompt text and system_prompt (if any)
  - response_text
  - token counts
  - latency_seconds
  - error / error_type (None on success)
  - retry attempts
  - expected_behavior and severity (from dataset)
  - run_id (groups records from a single invocation)

Output file: results/completions.jsonl  (one JSON object per line)
All runs append to the same file by default — filter by run_id to isolate runs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.providers.base import ModelProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"

CATEGORY_DATASETS = {
    "self_harm":         DATASETS_DIR / "self_harm.jsonl",
    "illicit_behavior":  DATASETS_DIR / "illicit_behavior.jsonl",
    "jailbreak_attempts": DATASETS_DIR / "jailbreak_attempts.jsonl",
    "prompt_injection":  DATASETS_DIR / "prompt_injection.jsonl",
    "benign_control":    DATASETS_DIR / "benign_control.jsonl",
}


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------

@dataclass
class CompletionRecord:
    """One record written to completions.jsonl per sample."""
    run_id: str
    prompt_id: str
    category: str
    model: str
    provider: str
    timestamp: str                       # ISO 8601 UTC

    prompt: str
    system_prompt: str

    response_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float | None

    error: str | None
    error_type: str | None
    attempts: int

    # Carried from dataset
    expected_behavior: str
    severity: str
    technique: str
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Controls what gets run and where output goes."""
    run_id: str
    categories: list[str] = field(default_factory=lambda: list(CATEGORY_DATASETS.keys()))
    output_path: Path = RESULTS_DIR / "completions.jsonl"
    system_prompt_override: str = ""     # If set, overrides per-sample system_prompt
    max_samples_per_category: int | None = None   # None = all samples
    dry_run: bool = False                # If True, skip API calls, write mock records


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL at %s line %d: %s", path, i, e)
    return records


def _iter_samples(config: RunConfig) -> Iterator[tuple[str, dict]]:
    """Yields (category, sample_dict) for all selected categories."""
    for category in config.categories:
        dataset_path = CATEGORY_DATASETS.get(category)
        if not dataset_path:
            logger.warning("No dataset registered for category '%s' — skipping.", category)
            continue

        try:
            samples = _load_jsonl(dataset_path)
        except FileNotFoundError as e:
            logger.warning("%s — skipping category.", e)
            continue

        if config.max_samples_per_category is not None:
            samples = samples[: config.max_samples_per_category]

        logger.info("Category '%s': %d samples", category, len(samples))
        for sample in samples:
            yield category, sample


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CompletionRunner:
    """
    Orchestrates: dataset → provider → completions.jsonl

    Usage:
        provider = get_provider("openai", model="gpt-4o")
        runner = CompletionRunner(provider)
        records = runner.run(RunConfig(run_id="my_run"))
    """

    def __init__(self, provider: ModelProvider):
        self._provider = provider

    def run(self, config: RunConfig) -> list[CompletionRecord]:
        """
        Run completions for all categories in config.
        Returns list of CompletionRecord — also writes to config.output_path.
        """
        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[CompletionRecord] = []
        total = 0
        successes = 0
        errors = 0

        logger.info(
            "Starting run '%s' | provider=%s model=%s | categories=%s",
            config.run_id,
            self._provider.provider_name,
            self._provider.model_name,
            config.categories,
        )

        with open(config.output_path, "a", encoding="utf-8") as out_file:
            for category, sample in _iter_samples(config):
                total += 1
                record = self._process_sample(config, category, sample)
                records.append(record)

                if record.error:
                    errors += 1
                    logger.warning(
                        "  [%s] ✗ %s: %s", sample.get("id"), record.error_type, record.error[:80]
                    )
                else:
                    successes += 1
                    preview = (record.response_text or "")[:60].replace("\n", " ")
                    logger.info(
                        "  [%s] ✓ %.2fs | '%s...'",
                        sample.get("id"), record.latency_seconds or 0, preview
                    )

                out_file.write(record.to_jsonl() + "\n")
                out_file.flush()   # flush after each record — safe against crashes

        logger.info(
            "Run '%s' complete: %d total | %d success | %d errors → %s",
            config.run_id, total, successes, errors, config.output_path,
        )
        return records

    def _process_sample(
        self, config: RunConfig, category: str, sample: dict
    ) -> CompletionRecord:
        """Process one dataset sample into a CompletionRecord."""
        prompt_id = sample.get("id", f"{category}_{int(time.time())}")
        prompt_text = sample.get("prompt", "")
        system_prompt = (
            config.system_prompt_override
            or sample.get("system_prompt", "")
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        if config.dry_run:
            return CompletionRecord(
                run_id=config.run_id,
                prompt_id=prompt_id,
                category=category,
                model=self._provider.model_name,
                provider=self._provider.provider_name,
                timestamp=timestamp,
                prompt=prompt_text,
                system_prompt=system_prompt,
                response_text="[DRY RUN]",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=0.0,
                error=None,
                error_type=None,
                attempts=0,
                expected_behavior=sample.get("expected_behavior", ""),
                severity=sample.get("severity", ""),
                technique=sample.get("technique", ""),
                notes=sample.get("notes", ""),
            )

        request = CompletionRequest(
            prompt=prompt_text,
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=0.0,
        )

        response: CompletionResponse = self._provider.complete(request)

        return CompletionRecord(
            run_id=config.run_id,
            prompt_id=prompt_id,
            category=category,
            model=response.model,
            provider=response.provider,
            timestamp=timestamp,
            prompt=prompt_text,
            system_prompt=system_prompt,
            response_text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
            error=response.error,
            error_type=response.error_type,
            attempts=response.attempts,
            expected_behavior=sample.get("expected_behavior", ""),
            severity=sample.get("severity", ""),
            technique=sample.get("technique", ""),
            notes=sample.get("notes", ""),
        )
