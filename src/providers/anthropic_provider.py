"""
src/providers/anthropic_provider.py

Anthropic Claude provider — drop-in replacement for OpenAIProvider.

Install:   pip install anthropic>=0.25.0
Env var:   ANTHROPIC_API_KEY

Usage:
    provider = AnthropicProvider(model="claude-opus-4-5")
    response = provider.complete(CompletionRequest(prompt="Hello"))

All output is shaped into CompletionResponse, identical to OpenAIProvider output —
downstream evaluation code sees no difference between providers.
"""

from __future__ import annotations

import logging
import os

from .base import (
    ModelProvider,
    CompletionRequest,
    CompletionResponse,
    RetryConfig,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderAPIError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-5"


class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude provider (Messages API).

    Environment:
        ANTHROPIC_API_KEY  — required
        ANTHROPIC_BASE_URL — optional, for proxy endpoints
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(retry_config=retry_config or RetryConfig())

        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic>=0.25.0"
            )

        self._model = model
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ProviderAuthError(
                "ANTHROPIC_API_KEY not set. "
                "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        client_kwargs = {"api_key": resolved_key}
        if base_url or os.environ.get("ANTHROPIC_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.environ["ANTHROPIC_BASE_URL"]

        self._client = anthropic.Anthropic(**client_kwargs)
        logger.info("AnthropicProvider initialized: model=%s", self._model)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, request: CompletionRequest) -> CompletionResponse:
        anthropic = self._anthropic

        # Build messages — Anthropic doesn't use a "system" role in messages[]
        messages = []
        for turn in request.conversation_history:
            messages.append(turn)
        messages.append({"role": "user", "content": request.prompt})

        kwargs = dict(
            model=self._model,
            max_tokens=request.max_tokens,
            messages=messages,
        )
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        try:
            api_response = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise ProviderAuthError(str(e)) from e
        except anthropic.RateLimitError as e:
            raise ProviderRateLimitError(str(e)) from e
        except anthropic.APITimeoutError as e:
            raise ProviderTimeoutError(str(e)) from e
        except anthropic.APIError as e:
            raise ProviderAPIError(str(e)) from e

        text = api_response.content[0].text if api_response.content else ""
        usage = api_response.usage

        return CompletionResponse(
            text=text,
            model=api_response.model,
            provider=self.provider_name,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
        )

    def health_check(self) -> bool:
        try:
            # Lightweight check — send a minimal message
            self._client.messages.create(
                model=self._model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            logger.info("Anthropic health check OK")
            return True
        except Exception as e:
            logger.error("Anthropic health check FAILED: %s", e)
            return False
