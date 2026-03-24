"""
src/providers/base.py

Abstract base class for all model providers.
Every provider must implement complete(), and may optionally implement
health_check() and stream().

Design goals:
  - One interface for all providers — swap OpenAI for Anthropic with one flag
  - CompletionRequest carries everything a provider needs (prompt, system, params)
  - CompletionResponse carries everything downstream needs (text, tokens, latency, errors)
  - Retry + rate-limit logic lives here so each provider doesn't re-implement it
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

@dataclass
class CompletionRequest:
    """Everything a provider needs to generate a completion."""
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional: for multi-turn / conversation support
    conversation_history: list[dict] = field(default_factory=list)


@dataclass
class CompletionResponse:
    """
    Standardized response from any provider.
    All fields are populated regardless of provider — None if unavailable.
    """
    text: str
    model: str
    provider: str

    # Token accounting
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Latency (seconds, wall clock from request to first token or full response)
    latency_seconds: float | None = None

    # Error handling — None means success
    error: str | None = None
    error_type: str | None = None   # e.g. "rate_limit", "auth", "timeout", "api_error"

    # Retry accounting
    attempts: int = 1

    # Pass-through for raw API response (useful for debugging)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        preview = (self.text or "")[:60].replace("\n", " ")
        status = "OK" if self.success else f"ERROR:{self.error_type}"
        return (
            f"CompletionResponse({status} | {self.provider}/{self.model} | "
            f"{self.latency_seconds:.2f}s | '{preview}...')"
        )


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """Controls retry + backoff behaviour for API calls."""
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0

    # Which error types trigger a retry
    retryable_error_types: tuple[str, ...] = (
        "rate_limit",
        "timeout",
        "server_error",
        "connection_error",
    )

    def backoff_for(self, attempt: int) -> float:
        """Exponential backoff with cap."""
        delay = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_backoff_seconds)


DEFAULT_RETRY_CONFIG = RetryConfig()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses implement _call_api() — the raw single-attempt API call.
    Retry logic, error normalization, and latency tracking are handled here.
    """

    def __init__(self, retry_config: RetryConfig = DEFAULT_RETRY_CONFIG):
        self._retry_config = retry_config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier e.g. 'openai', 'anthropic'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model string as sent to the API e.g. 'gpt-4o'."""
        ...

    @abstractmethod
    def _call_api(self, request: CompletionRequest) -> CompletionResponse:
        """
        Single API call attempt. Must raise ProviderError on failure so
        the retry wrapper can handle it correctly.
        """
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Public interface: calls _call_api() with retry/backoff logic.
        Always returns a CompletionResponse — errors are surfaced in
        response.error rather than raised, so the caller can decide
        how to handle them.
        """
        cfg = self._retry_config
        last_response: CompletionResponse | None = None

        for attempt in range(1, cfg.max_attempts + 1):
            t0 = time.monotonic()
            try:
                response = self._call_api(request)
                response.latency_seconds = time.monotonic() - t0
                response.attempts = attempt
                return response

            except ProviderRateLimitError as e:
                backoff = cfg.backoff_for(attempt)
                logger.warning(
                    "[%s] Rate limit on attempt %d/%d — sleeping %.1fs: %s",
                    self.provider_name, attempt, cfg.max_attempts, backoff, e
                )
                last_response = CompletionResponse(
                    text="", model=self.model_name, provider=self.provider_name,
                    error=str(e), error_type="rate_limit", attempts=attempt,
                    latency_seconds=time.monotonic() - t0,
                )
                if attempt < cfg.max_attempts:
                    time.sleep(backoff)

            except ProviderTimeoutError as e:
                backoff = cfg.backoff_for(attempt)
                logger.warning(
                    "[%s] Timeout on attempt %d/%d — sleeping %.1fs",
                    self.provider_name, attempt, cfg.max_attempts, backoff
                )
                last_response = CompletionResponse(
                    text="", model=self.model_name, provider=self.provider_name,
                    error=str(e), error_type="timeout", attempts=attempt,
                    latency_seconds=time.monotonic() - t0,
                )
                if attempt < cfg.max_attempts:
                    time.sleep(backoff)

            except ProviderAuthError as e:
                # Auth errors are not retryable
                logger.error("[%s] Auth error — not retrying: %s", self.provider_name, e)
                return CompletionResponse(
                    text="", model=self.model_name, provider=self.provider_name,
                    error=str(e), error_type="auth", attempts=attempt,
                    latency_seconds=time.monotonic() - t0,
                )

            except ProviderAPIError as e:
                backoff = cfg.backoff_for(attempt)
                logger.warning(
                    "[%s] API error on attempt %d/%d — sleeping %.1fs: %s",
                    self.provider_name, attempt, cfg.max_attempts, backoff, e
                )
                last_response = CompletionResponse(
                    text="", model=self.model_name, provider=self.provider_name,
                    error=str(e), error_type="api_error", attempts=attempt,
                    latency_seconds=time.monotonic() - t0,
                )
                if attempt < cfg.max_attempts:
                    time.sleep(backoff)

            except Exception as e:
                logger.error(
                    "[%s] Unexpected error on attempt %d/%d: %s",
                    self.provider_name, attempt, cfg.max_attempts, e
                )
                return CompletionResponse(
                    text="", model=self.model_name, provider=self.provider_name,
                    error=str(e), error_type="unexpected", attempts=attempt,
                    latency_seconds=time.monotonic() - t0,
                )

        # Exhausted retries
        logger.error(
            "[%s] Exhausted %d attempts.", self.provider_name, cfg.max_attempts
        )
        return last_response or CompletionResponse(
            text="", model=self.model_name, provider=self.provider_name,
            error="Exhausted retries", error_type="exhausted",
            attempts=cfg.max_attempts,
        )

    def health_check(self) -> bool:
        """Override to verify connectivity. Returns True if OK."""
        return True


# ---------------------------------------------------------------------------
# Provider error hierarchy
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base class for provider errors."""

class ProviderRateLimitError(ProviderError):
    """HTTP 429 or equivalent."""

class ProviderAuthError(ProviderError):
    """Invalid or missing API key."""

class ProviderTimeoutError(ProviderError):
    """Request timed out."""

class ProviderAPIError(ProviderError):
    """Other API error (5xx, malformed response, etc.)."""
