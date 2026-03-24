"""
src/providers/__init__.py

Provider registry and factory.

To add a new provider:
  1. Create src/providers/your_provider.py implementing ModelProvider
  2. Add it to _REGISTRY below
  3. Use it: get_provider("your_provider", model="model-name")
"""

from .base import (
    ModelProvider,
    CompletionRequest,
    CompletionResponse,
    RetryConfig,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderAPIError,
    DEFAULT_RETRY_CONFIG,
)
from .openai_provider import OpenAIProvider

# Lazy-load optional providers so missing dependencies don't break imports
_REGISTRY: dict[str, type[ModelProvider]] = {
    "openai": OpenAIProvider,
}

try:
    from .anthropic_provider import AnthropicProvider
    _REGISTRY["anthropic"] = AnthropicProvider
except ImportError:
    pass  # anthropic package not installed


def get_provider(
    name: str,
    model: str | None = None,
    api_key: str | None = None,
    retry_config: RetryConfig | None = None,
    **kwargs,
) -> ModelProvider:
    """
    Factory function. Returns a configured ModelProvider instance.

    Args:
        name:         Provider name — "openai" or "anthropic"
        model:        Model string (provider-specific). Uses provider default if None.
        api_key:      API key. Falls back to environment variable if None.
        retry_config: Custom retry settings. Uses DEFAULT_RETRY_CONFIG if None.
        **kwargs:     Extra kwargs passed to the provider constructor.

    Raises:
        ValueError:   Unknown provider name.
        ProviderAuthError: Missing or invalid API key.
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )

    provider_cls = _REGISTRY[name]

    # Build kwargs for the provider constructor
    ctor_kwargs = {}
    if model:
        ctor_kwargs["model"] = model
    if api_key:
        ctor_kwargs["api_key"] = api_key
    if retry_config:
        ctor_kwargs["retry_config"] = retry_config
    ctor_kwargs.update(kwargs)

    return provider_cls(**ctor_kwargs)


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    return sorted(_REGISTRY.keys())


__all__ = [
    # Core abstractions
    "ModelProvider",
    "CompletionRequest",
    "CompletionResponse",
    "RetryConfig",
    "DEFAULT_RETRY_CONFIG",
    # Errors
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderAPIError",
    # Concrete providers
    "OpenAIProvider",
    # Factory
    "get_provider",
    "list_providers",
]
