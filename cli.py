#!/usr/bin/env python3
"""
cli.py — LLM Safety Evaluation and Release Gating System

Commands:
  completions     Run model completions against JSONL datasets
  evaluate        Score completions using LLM judge or heuristic fallback
  gate            Aggregate scores and produce a release recommendation
  report          Generate markdown + JSON reports from a gating result
  run-all         Pipeline: completions → evaluate → gate → report
  validate-config Validate safety_policy.yaml and release_thresholds.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")

PROJECT_ROOT = Path(__file__).parent
POLICIES_DIR = PROJECT_ROOT / "policies"
RESULTS_DIR = PROJECT_ROOT / "results"
POLICY_PATH = POLICIES_DIR / "safety_policy.yaml"
THRESHOLDS_PATH = POLICIES_DIR / "release_thresholds.yaml"


def _run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# completions
# ---------------------------------------------------------------------------

def cmd_completions(args):
    from src.providers import get_provider
    from src.runners.completion_runner import CompletionRunner, RunConfig

    provider = get_provider(args.provider, model=args.model or None)
    logger.info("Provider: %s / %s", provider.provider_name, provider.model_name)

    config = RunConfig(
        run_id=args.run_id,
        categories=[args.category] if args.category else None,
        output_path=RESULTS_DIR / args.run_id / "completions.jsonl",
        max_samples_per_category=args.max_samples,
        dry_run=args.dry_run,
    )
    runner = CompletionRunner(provider)
    records = runner.run(config)
    logger.info("Done: %d records → %s", len(records), config.output_path)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    import json as _json
    from src.runners.completion_runner import CompletionRecord
    from src.evaluators.llm_judge_evaluator import LLMJudgeEvaluator
    from src.evaluators.heuristic_evaluator import HeuristicEvaluator
    from src.evaluators.llm_judge_evaluator import EvaluationRecord
    from datetime import timezone

    run_dir = RESULTS_DIR / args.run_id
    completions_path = run_dir / "completions.jsonl"
    if not completions_path.exists():
        logger.error("No completions file at %s — run 'completions' first.", completions_path)
        sys.exit(1)

    with open(completions_path) as f:
        raw_records = [_json.loads(l) for l in f if l.strip()]

    completion_records = [CompletionRecord(**r) for r in raw_records]
    output_path = run_dir / "evaluations.jsonl"

    if args.mode == "heuristic":
        logger.info("Running heuristic evaluator (no API calls)")
        evaluator = HeuristicEvaluator()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as out:
            for i, rec in enumerate(completion_records, 1):
                scores = evaluator.score(rec)
                eval_rec = EvaluationRecord(
                    run_id=rec.run_id,
                    prompt_id=rec.prompt_id,
                    category=rec.category,
                    model=rec.model,
                    provider=rec.provider,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    prompt=rec.prompt,
                    response_text=rec.response_text,
                    expected_behavior=rec.expected_behavior,
                    severity=rec.severity,
                    technique=rec.technique,
                    policy_compliance=scores["policy_compliance"],
                    refusal_quality=scores["refusal_quality"],
                    harmfulness=scores["harmfulness"],
                    explanation=scores["explanation"],
                    evaluator="heuristic",
                    judge_model=None,
                    parse_error=False,
                )
                out.write(eval_rec.to_jsonl() + "\n")
                if i % 10 == 0:
                    logger.info("  Evaluated %d / %d", i, len(completion_records))
    else:
        from src.providers import get_provider
        judge_provider = get_provider(args.judge_provider, model=args.judge_model or None)
        logger.info("LLM judge: %s / %s", judge_provider.provider_name, judge_provider.model_name)
        evaluator = LLMJudgeEvaluator(judge_provider)
        evaluator.evaluate_batch(completion_records, output_path=output_path)

    logger.info("Evaluations saved → %s", output_path)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def cmd_gate(args):
    from src.evaluators.llm_judge_evaluator import EvaluationRecord
    from src.gating.aggregator import make_recommendation, save_recommendation_json

    run_dir = RESULTS_DIR / args.run_id
    eval_path = run_dir / "evaluations.jsonl"
    if not eval_path.exists():
        logger.error("No evaluations at %s — run 'evaluate' first.", eval_path)
        sys.exit(1)

    with open(eval_path) as f:
        records = [EvaluationRecord(**json.loads(l)) for l in f if l.strip()]

    rec = make_recommendation(records, args.model_id, args.run_id)
    logger.info("Decision: %s | Risk: %.4f (%s)", rec.decision, rec.composite_risk_score, rec.risk_band)

    json_path = run_dir / "release_decision.json"
    save_recommendation_json(rec, json_path)
    logger.info("JSON report → %s", json_path)

    # Write recommendation to a temp file so cmd_report can pick it up
    rec_path = run_dir / "_recommendation.json"
    save_recommendation_json(rec, rec_path)

    return rec


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args):
    from src.evaluators.llm_judge_evaluator import EvaluationRecord
    from src.gating.aggregator import make_recommendation, save_recommendation_json
    from src.reporting.markdown_report import generate_markdown_report

    run_dir = RESULTS_DIR / args.run_id
    eval_path = run_dir / "evaluations.jsonl"
    if not eval_path.exists():
        logger.error("No evaluations at %s", eval_path)
        sys.exit(1)

    with open(eval_path) as f:
        records = [EvaluationRecord(**json.loads(l)) for l in f if l.strip()]

    rec = make_recommendation(records, args.model_id, args.run_id)

    md_path = run_dir / "release_report.md"
    report = generate_markdown_report(rec, output_path=md_path)
    print(report)

    json_path = run_dir / "release_decision.json"
    save_recommendation_json(rec, json_path)

    logger.info("Markdown report → %s", md_path)
    logger.info("JSON report     → %s", json_path)

    sys.exit({"SHIP": 0, "BLOCK": 1, "CONDITIONAL_SHIP": 2}[rec.decision])


# ---------------------------------------------------------------------------
# run-all
# ---------------------------------------------------------------------------

def cmd_run_all(args):
    logger.info("=== Step 1/3: Completions ===")
    cmd_completions(args)

    logger.info("=== Step 2/3: Evaluate ===")
    cmd_evaluate(args)

    logger.info("=== Step 3/3: Report ===")
    cmd_report(args)


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------

def cmd_validate_config(args):
    from src.gating.policy_loader import load_and_validate_configs
    try:
        cfg = load_and_validate_configs(POLICY_PATH, THRESHOLDS_PATH)
        logger.info("✅ Config valid. Policy v%s | Thresholds v%s",
                    cfg.policy.version, cfg.thresholds.version)
        print("Categories:", list(cfg.policy.categories.keys()))
    except Exception as e:
        logger.error("❌ Config invalid: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-gate",
        description="LLM Safety Evaluation and Release Gating System",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # completions
    p = sub.add_parser("completions", help="Run model completions")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=None, help="Model string (uses provider default if omitted)")
    p.add_argument("--category", default=None, help="Single category (omit for all)")
    p.add_argument("--run-id", default=_run_id())
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_completions)

    # evaluate
    p = sub.add_parser("evaluate", help="Score completions (LLM judge or heuristic)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", choices=["llm_judge", "heuristic"], default="heuristic",
                   help="Evaluator mode (heuristic = no API calls)")
    p.add_argument("--judge-provider", default="openai", help="Provider for LLM judge")
    p.add_argument("--judge-model", default=None, help="Judge model (uses provider default if omitted)")
    p.set_defaults(func=cmd_evaluate)

    # gate
    p = sub.add_parser("gate", help="Aggregate evaluations → release decision")
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-id", required=True)
    p.set_defaults(func=cmd_gate)

    # report
    p = sub.add_parser("report", help="Generate markdown + JSON release report")
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-id", required=True)
    p.set_defaults(func=cmd_report)

    # run-all
    p = sub.add_parser("run-all", help="Full pipeline: completions → evaluate → report")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--model-id", required=True)
    p.add_argument("--run-id", default=_run_id())
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--mode", choices=["llm_judge", "heuristic"], default="heuristic")
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--category", default=None)
    p.set_defaults(func=cmd_run_all)

    # validate-config
    p = sub.add_parser("validate-config", help="Validate policy + threshold configs")
    p.set_defaults(func=cmd_validate_config)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
