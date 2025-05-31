# test_agent/llm/__init__.py

from typing import Dict, Type

from .base import LLMProvider
from .claude import ClaudeProvider
from .openai import OpenAIProvider
from .deepseek import DeepSeekProvider
from .gemini import GeminiProvider

# Dictionary mapping provider names to provider classes
PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "gemini": GeminiProvider,
}


def get_provider(provider_name: str) -> LLMProvider:
    """
    Get an LLM provider instance for the specified provider.

    Args:
        provider_name: The name of the provider

    Returns:
        An instance of the appropriate LLM provider

    Raises:
        ValueError: If the provider is not supported
    """
    provider_name = provider_name.lower()
    if provider_name not in PROVIDERS:
        supported_providers = list(PROVIDERS.keys())
        raise ValueError(
            f"Unsupported LLM provider: {provider_name}. Supported providers: {supported_providers}"
        )

    return PROVIDERS[provider_name]()


def list_providers() -> Dict[str, Dict[str, str]]:
    """
    Get information about all available providers.

    Returns:
        Dictionary with provider names as keys and their information
    """
    result = {}
    for name, provider_class in PROVIDERS.items():
        provider = provider_class()
        result[name] = {
            "name": name,
            "default_model": provider.default_model,
            "available_models": provider.available_models,
        }
    return result
