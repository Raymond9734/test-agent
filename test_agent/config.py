# test_agent/config.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the test agent.

    Handles loading from files or environment variables and provides
    access to configuration values with defaults.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the config manager.

        Args:
            config_file: Optional path to config file. If None, uses default locations.
        """
        self.config: Dict[str, Any] = {}

        # Load config from file
        if config_file:
            self.config_file = config_file
        else:
            # Default to ~/.test_agent/config.json
            self.config_file = str(Path.home() / ".test_agent" / "config.json")

        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file."""
        # Check if config file exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                logger.debug(f"Loaded config from {self.config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading config from {self.config_file}: {e}")
                self.config = {}
        else:
            logger.debug(f"Config file {self.config_file} not found. Using defaults.")
            self.config = {}

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.debug(f"Saved config to {self.config_file}")
        except IOError as e:
            logger.error(f"Error saving config to {self.config_file}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Try to get from environment variables
        env_key = f"TEST_AGENT_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]

        # Try to get from config
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

        # Save changes
        self.save_config()

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get the API key for a provider from config or environment.

        Args:
            provider: Provider name

        Returns:
            API key if found, None otherwise
        """
        # Try environment variables first with provider-specific names
        if provider.lower() == "claude":
            if "ANTHROPIC_API_KEY" in os.environ:
                return os.environ["ANTHROPIC_API_KEY"]
        elif provider.lower() == "openai":
            if "OPENAI_API_KEY" in os.environ:
                return os.environ["OPENAI_API_KEY"]
        elif provider.lower() == "deepseek":
            if "DEEPSEEK_API_KEY" in os.environ:
                return os.environ["DEEPSEEK_API_KEY"]
        elif provider.lower() == "gemini":
            if "GOOGLE_API_KEY" in os.environ:
                return os.environ["GOOGLE_API_KEY"]

        # Try from config
        api_keys = self.config.get("api_keys", {})
        return api_keys.get(provider.lower())

    def set_api_key(self, provider: str, api_key: str) -> None:
        """
        Set an API key for a provider.

        Args:
            provider: Provider name
            api_key: API key to set
        """
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}

        self.config["api_keys"][provider.lower()] = api_key
        self.save_config()


# Global config instance
config = ConfigManager()
