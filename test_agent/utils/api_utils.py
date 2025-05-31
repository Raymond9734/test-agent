# test_agent/utils/api_utils.py

import os
import json
from typing import Optional
from pathlib import Path

# Config file location
CONFIG_DIR = Path.home() / ".test_agent"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for a provider from environment variables or config file.

    Args:
        provider: Provider name

    Returns:
        API key if found, None otherwise
    """
    # Try environment variables first with correct naming
    if provider.lower() == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
    elif provider.lower() == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
    elif provider.lower() == "deepseek":
        # Check both uppercase and lowercase variants
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            return key

        # Also check alternative naming
        key = os.environ.get("deepseek_api_key")
        if key:
            return key
    elif provider.lower() == "gemini":
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            return key

    # Try config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)

            # Get API key
            api_keys = config.get("api_keys", {})
            return api_keys.get(provider.lower())

        except (json.JSONDecodeError, IOError):
            return None

    return None


def save_api_key(provider: str, api_key: str) -> None:
    """
    Save an API key to environment and config file.

    Args:
        provider: Provider name
        api_key: API key to save
    """
    # Set environment variable with correct naming
    if provider.lower() == "claude":
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider.lower() == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider.lower() == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = api_key
    elif provider.lower() == "gemini":
        os.environ["GOOGLE_API_KEY"] = api_key

    # Save to config file
    CONFIG_DIR.mkdir(exist_ok=True)

    # Load existing config
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}

    # Update API keys
    if "api_keys" not in config:
        config["api_keys"] = {}

    config["api_keys"][provider.lower()] = api_key

    # Save config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_last_provider() -> Optional[str]:
    """
    Get the last used LLM provider from config file.

    Returns:
        Last used provider name or None if not found
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("last_provider")
        except (json.JSONDecodeError, IOError):
            return None

    return None


def save_last_provider(provider: str) -> None:
    """
    Save the current provider as the last used one.

    Args:
        provider: Provider name
    """
    CONFIG_DIR.mkdir(exist_ok=True)

    # Load existing config
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}

    # Save last provider
    config["last_provider"] = provider

    # Write config file
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
