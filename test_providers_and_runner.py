"""
tests/test_providers_and_runner.py

Tests for the provider abstraction layer, retry logic, error handling,
and CompletionRunner. Uses mock/echo providers — no API keys required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RetryConfig,
)
from src.providers import get_provider, list_providers
from src.runners.completion_runner import CompletionRecord, CompletionRunner, RunConfig


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class EchoProvider(ModelProvider):
    """Returns the prompt text as the response. Supports injected failures."""

    def __init__(
        self,
        fail_times: int = 0,
        fail_with: type[Exception] | None = None,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(
            retry_config=retry_config
            or RetryConfig(max_attempts=3, initial_backoff_seconds=0.001)
        )
        self._fail_times = fail_times
        self._fail_with = fail_with or ProviderAPIError
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "echo"

    @property
    def model_name(self) -> str:
        return "echo-v1"

    def _call_api(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1
        if self._call_count <= self._fail_times:
            raise self._fail_with(f"Injected failure #{self._call_count}")
        return CompletionResponse(
            text=f"Echo: {request.prompt[:40]}",
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=5,
            total_tokens=len(request.prompt.split()) + 5,
        )


class AlwaysFailProvider(ModelProvider):
    """Always raises — tests retry exhaustion."""

    def __init__(self):
        super().__init__(
            retry_config=RetryConfig(max_attempts=2, initial_backoff_seconds=0.001)
        )
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "failing"

    @property
    def model_name(self) -> str:
        return "failing-v1"

    def _call_api(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1
        raise ProviderAPIError("Permanent failure")


# ---------------------------------------------------------------------------
# CompletionRequest / CompletionResponse dataclasses
# ---------------------------------------------------------------------------


class TestCompletionDataclasses:
    def test_request_defaults(self):
        req = CompletionRequest(prompt="hello")
        assert req.system_prompt == ""
        assert req.temperature == 0.0
        assert req.max_tokens == 512
        assert req.metadata == {}
        assert req.conversation_history == []

    def test_response_success_when_no_error(self):
        resp = CompletionResponse(text="hi", model="m", provider="p")
        assert resp.success is True

    def test_response_failure_when_error_set(self):
        resp = CompletionResponse(
            text="", model="m", provider="p", error="oops", error_type="api_error"
        )
        assert resp.success is False

    def test_response_repr_shows_ok(self):
        resp = CompletionResponse(
            text="Hello there", model="gpt-4o", provider="openai", latency_seconds=0.42
        )
        assert "OK" in repr(resp)
        assert "openai" in repr(resp)

    def test_response_repr_shows_error_type(self):
        resp = CompletionResponse(
            text="", model="m", provider="p",
            error="bad key", error_type="auth", latency_seconds=0.1
        )
        assert "ERROR" in repr(resp)


# ---------------------------------------------------------------------------
# EchoProvider — happy path
# ---------------------------------------------------------------------------


class TestEchoProviderHappyPath:
    def test_basic_completion_succeeds(self):
        provider = EchoProvider()
        resp = provider.complete(CompletionRequest(prompt="What is 2+2?"))
        assert resp.success
        assert "What is 2+2?" in resp.text

    def test_provider_and_model_name_set(self):
        provider = EchoProvider()
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.provider == "echo"
        assert resp.model == "echo-v1"

    def test_latency_is_non_negative(self):
        provider = EchoProvider()
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.latency_seconds is not None
        assert resp.latency_seconds >= 0.0

    def test_single_attempt_on_success(self):
        provider = EchoProvider()
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.attempts == 1
        assert provider._call_count == 1

    def test_tokens_populated(self):
        provider = EchoProvider()
        resp = provider.complete(CompletionRequest(prompt="hello world"))
        assert resp.prompt_tokens > 0
        assert resp.completion_tokens > 0

    def test_system_prompt_not_required(self):
        provider = EchoProvider()
        resp = provider.complete(CompletionRequest(prompt="test"))
        assert resp.success


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retries_on_rate_limit_and_succeeds(self):
        provider = EchoProvider(fail_times=1, fail_with=ProviderRateLimitError)
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.success
        assert provider._call_count == 2
        assert resp.attempts == 2

    def test_retries_on_api_error_and_succeeds(self):
        provider = EchoProvider(fail_times=1, fail_with=ProviderAPIError)
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.success
        assert provider._call_count == 2

    def test_retries_on_timeout_and_succeeds(self):
        provider = EchoProvider(fail_times=1, fail_with=ProviderTimeoutError)
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.success
        assert provider._call_count == 2

    def test_auth_error_not_retried(self):
        """Auth errors are fatal — no retry."""
        provider = EchoProvider(fail_times=5, fail_with=ProviderAuthError)
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert not resp.success
        assert resp.error_type == "auth"
        assert provider._call_count == 1

    def test_exhausted_retries_returns_error_response(self):
        provider = AlwaysFailProvider()
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert not resp.success
        assert resp.error_type in {"api_error", "exhausted"}
        assert provider._call_count == 2

    def test_success_after_two_failures(self):
        provider = EchoProvider(fail_times=2, fail_with=ProviderAPIError)
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert resp.success
        assert provider._call_count == 3
        assert resp.attempts == 3

    def test_max_attempts_one_means_no_retry(self):
        cfg = RetryConfig(max_attempts=1, initial_backoff_seconds=0.001)
        provider = EchoProvider(fail_times=1, fail_with=ProviderAPIError, retry_config=cfg)
        resp = provider.complete(CompletionRequest(prompt="hi"))
        assert not resp.success
        assert provider._call_count == 1

    def test_backoff_calculation_exponential(self):
        cfg = RetryConfig(
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=10.0,
        )
        assert cfg.backoff_for(1) == 1.0
        assert cfg.backoff_for(2) == 2.0
        assert cfg.backoff_for(3) == 4.0

    def test_backoff_capped_at_max(self):
        cfg = RetryConfig(
            initial_backoff_seconds=1.0,
            backoff_multiplier=10.0,
            max_backoff_seconds=5.0,
        )
        assert cfg.backoff_for(5) == 5.0


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


class TestProviderFactory:
    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("definitely_not_real")

    def test_list_providers_includes_openai(self):
        assert "openai" in list_providers()

    def test_openai_provider_raises_without_key(self):
        """Missing API key → ProviderAuthError at construction."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(Exception):  # ProviderAuthError or ImportError
                get_provider("openai", model="gpt-4o")


# ---------------------------------------------------------------------------
# CompletionRunner — dry run
# ---------------------------------------------------------------------------


class TestCompletionRunnerDryRun:
    def _config(self, tmp_path: Path, **kwargs) -> RunConfig:
        return RunConfig(
            run_id="test_run",
            categories=["self_harm", "benign_control"],
            output_path=tmp_path / "completions.jsonl",
            dry_run=True,
            **kwargs,
        )

    def test_dry_run_returns_correct_count(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        records = runner.run(self._config(tmp_path))
        assert len(records) == 28  # self_harm=14, benign_control=14

    def test_dry_run_creates_output_file(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = self._config(tmp_path)
        runner.run(config)
        assert config.output_path.exists()

    def test_output_is_valid_jsonl(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = self._config(tmp_path)
        runner.run(config)
        lines = [
            json.loads(l)
            for l in config.output_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(lines) == 28

    def test_all_required_fields_present(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = self._config(tmp_path)
        records = runner.run(config)
        required = {
            "run_id", "prompt_id", "category", "model", "provider",
            "timestamp", "prompt", "system_prompt", "response_text",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "latency_seconds", "error", "error_type", "attempts",
            "expected_behavior", "severity", "technique", "notes",
        }
        for r in records:
            assert required.issubset(set(r.to_dict().keys()))

    def test_categories_match_requested(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        records = runner.run(self._config(tmp_path))
        assert {r.category for r in records} == {"self_harm", "benign_control"}

    def test_max_samples_per_category(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = self._config(tmp_path, max_samples_per_category=3)
        records = runner.run(config)
        assert len(records) == 6  # 3 × 2 categories

    def test_run_id_propagated(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = self._config(tmp_path)
        records = runner.run(config)
        assert all(r.run_id == "test_run" for r in records)

    def test_timestamps_are_iso_format(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        records = runner.run(self._config(tmp_path))
        from datetime import datetime
        for r in records:
            # Should parse without error
            datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))

    def test_dataset_fields_carried_through(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        records = runner.run(self._config(tmp_path))
        for r in records:
            assert r.expected_behavior != ""
            assert r.severity != ""
            assert r.technique != ""

    def test_multiple_runs_append_to_file(self, tmp_path):
        """Two runs to the same output file → file has 2× the records."""
        runner = CompletionRunner(EchoProvider())
        config = self._config(tmp_path, max_samples_per_category=2)
        runner.run(config)
        runner.run(config)
        lines = [
            l for l in config.output_path.read_text().splitlines() if l.strip()
        ]
        assert len(lines) == 8  # 2 runs × 2 categories × 2 samples


# ---------------------------------------------------------------------------
# CompletionRunner — live (using EchoProvider, no real API)
# ---------------------------------------------------------------------------


class TestCompletionRunnerLive:
    def test_live_run_calls_provider(self, tmp_path):
        provider = EchoProvider()
        runner = CompletionRunner(provider)
        config = RunConfig(
            run_id="live_test",
            categories=["benign_control"],
            output_path=tmp_path / "live.jsonl",
            max_samples_per_category=3,
            dry_run=False,
        )
        records = runner.run(config)
        assert len(records) == 3
        assert provider._call_count == 3

    def test_live_run_response_text_populated(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = RunConfig(
            run_id="live_test",
            categories=["benign_control"],
            output_path=tmp_path / "live.jsonl",
            max_samples_per_category=2,
            dry_run=False,
        )
        records = runner.run(config)
        for r in records:
            assert r.response_text.startswith("Echo:")

    def test_live_run_latency_recorded(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = RunConfig(
            run_id="live_test",
            categories=["benign_control"],
            output_path=tmp_path / "live.jsonl",
            max_samples_per_category=2,
            dry_run=False,
        )
        records = runner.run(config)
        for r in records:
            assert r.latency_seconds is not None
            assert r.latency_seconds >= 0.0

    def test_provider_error_captured_not_raised(self, tmp_path):
        """Provider errors go into the record — they don't crash the run."""
        runner = CompletionRunner(AlwaysFailProvider())
        config = RunConfig(
            run_id="error_test",
            categories=["benign_control"],
            output_path=tmp_path / "errors.jsonl",
            max_samples_per_category=2,
            dry_run=False,
        )
        records = runner.run(config)   # should NOT raise
        assert len(records) == 2
        for r in records:
            assert r.error is not None
            assert r.response_text == ""

    def test_all_categories_full_run(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = RunConfig(
            run_id="full_run",
            output_path=tmp_path / "full.jsonl",
            max_samples_per_category=2,
            dry_run=False,
        )
        records = runner.run(config)
        cats = {r.category for r in records}
        assert cats == {
            "self_harm", "illicit_behavior", "jailbreak_attempts",
            "prompt_injection", "benign_control",
        }
        assert len(records) == 10  # 5 categories × 2 samples

    def test_jsonl_output_parseable_after_live_run(self, tmp_path):
        runner = CompletionRunner(EchoProvider())
        config = RunConfig(
            run_id="parse_test",
            categories=["benign_control"],
            output_path=tmp_path / "parse.jsonl",
            max_samples_per_category=3,
            dry_run=False,
        )
        runner.run(config)
        lines = [
            json.loads(l)
            for l in config.output_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(lines) == 3
        for line in lines:
            assert "response_text" in line
            assert "latency_seconds" in line
            assert "timestamp" in line
