"""
src/providers/openai_provider.py

OpenAI provider implementation.
Supports all chat-completion models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, o1, etc.

Environment variable:  OPENAI_API_KEY
Optional:              OPENAI_ORG_ID, OPENAI_BASE_URL (for Azure or proxies)
"""

from __future__ import annotations

import logging
import os
from typing import Any

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

# Default model — can be overridden at construction
DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(ModelProvider):
    """
    OpenAI Chat Completions provider.

    Usage:
        provider = OpenAIProvider(model="gpt-4o")
        response = provider.complete(CompletionRequest(prompt="Hello"))

    Environment:
        OPENAI_API_KEY   — required
        OPENAI_ORG_ID    — optional, for org-scoped billing
        OPENAI_BASE_URL  — optional, for Azure OpenAI or proxy endpoints
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        org_id: str | None = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(retry_config=retry_config or RetryConfig())

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai>=1.30.0"
            )

        self._model = model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ProviderAuthError(
                "OPENAI_API_KEY not set. "
                "Export it: export OPENAI_API_KEY=sk-..."
            )

        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if org_id or os.environ.get("OPENAI_ORG_ID"):
            client_kwargs["organization"] = org_id or os.environ["OPENAI_ORG_ID"]
        if base_url or os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.environ["OPENAI_BASE_URL"]

        self._client = openai.OpenAI(**client_kwargs)
        self._openai = openai

        logger.info(
            "OpenAIProvider initialized: model=%s org=%s base_url=%s",
            self._model,
            client_kwargs.get("organization", "default"),
            client_kwargs.get("base_url", "https://api.openai.com"),
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        # Inject conversation history if provided
        for turn in request.conversation_history:
            messages.append(turn)
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _call_api(self, request: CompletionRequest) -> CompletionResponse:
        messages = self._build_messages(request)

        try:
            api_response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except self._openai.AuthenticationError as e:
            raise ProviderAuthError(str(e)) from e
        except self._openai.RateLimitError as e:
            raise ProviderRateLimitError(str(e)) from e
        except self._openai.APITimeoutError as e:
            raise ProviderTimeoutError(str(e)) from e
        except self._openai.APIError as e:
            raise ProviderAPIError(str(e)) from e

        choice = api_response.choices[0]
        text = choice.message.content or ""
        usage = api_response.usage

        return CompletionResponse(
            text=text,
            model=api_response.model,
            provider=self.provider_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            raw=api_response.model_dump(),
        )

    def health_check(self) -> bool:
        try:
            models = self._client.models.list()
            logger.info(
                "OpenAI health check OK — %d models available", len(list(models))
            )
            return True
        except Exception as e:
            logger.error("OpenAI health check FAILED: %s", e)
            return False
