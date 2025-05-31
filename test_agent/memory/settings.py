# test_agent/memory/settings.py

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Manages persistent settings for the test agent.

    This class handles saving and loading settings to/from a file,
    provides defaults, and ensures settings are validated.
    """

    # Default settings
    DEFAULT_SETTINGS = {
        "llm": {
            "provider": "claude",
            "model": "claude-3-5-sonnet-20240620",
            "temperature": 0.2,
            "max_tokens": 4000,
            "streaming": True,
        },
        "testing": {
            "default_framework": {
                "python": "pytest",
                "go": "go_test",
            },
            "max_test_size": 5000,  # Maximum size of generated tests in tokens
            "max_iterations": 3,  # Maximum test fixing iterations
            "run_tests": True,  # Whether to run tests after generation
            "fix_failures": True,  # Whether to attempt to fix failing tests
        },
        "project": {
            "default_patterns": {
                "python": {
                    "location_pattern": "tests_subdirectory",  # Changed from "tests_directory"
                    "naming_convention": "test_prefix",
                },
                "go": {
                    "location_pattern": "tests_subdirectory",  # Changed from "same_directory"
                    "package_pattern": "same_package",
                },
            },
            "default_exclude_dirs": [
                ".git",
                ".github",
                ".vscode",
                ".idea",
                "node_modules",
                "venv",
                "env",
                ".venv",
                ".env",
                "build",
                "dist",
                "target",
                "__pycache__",
            ],
            "default_exclude_files": [
                "*.pyc",
                "*.pyo",
                "*.pyd",
                "*.so",
                "*.dll",
                "*.exe",
                "*.min.js",
                "*.min.css",
                ".DS_Store",
                "Thumbs.db",
            ],
        },
        "cache": {
            "enabled": True,
            "ttl": 7 * 24 * 60 * 60,  # 7 days in seconds
            "max_size_mb": 100,
        },
        "ui": {
            "verbose": False,
            "show_progress": True,
            "color_output": True,
        },
    }

    def __init__(self, project_dir: str, settings_dir: Optional[str] = None):
        """
        Initialize the settings manager.

        Args:
            project_dir: Project directory (used for project-specific settings)
            settings_dir: Optional directory to store settings.
                          If None, uses ~/.test_agent/settings
        """
        self.project_dir = os.path.abspath(project_dir)
        self.project_hash = self._hash_project_path(project_dir)

        # Set up settings directory
        if settings_dir is None:
            settings_dir = os.path.join(str(Path.home()), ".test_agent", "settings")

        self.settings_dir = settings_dir
        os.makedirs(settings_dir, exist_ok=True)

        # File paths
        self.global_settings_path = os.path.join(settings_dir, "global_settings.json")
        self.project_settings_path = os.path.join(
            settings_dir, f"project_{self.project_hash}.json"
        )

        # Initialize settings
        self.global_settings = self._load_settings(self.global_settings_path)
        self.project_settings = self._load_settings(self.project_settings_path)

        # Merge with defaults
        self._merge_with_defaults()

    def _hash_project_path(self, project_path: str) -> str:
        """
        Create a hash of the project path to use as a unique identifier.

        Args:
            project_path: Path to the project directory

        Returns:
            MD5 hash of the normalized project path
        """
        import hashlib

        # Normalize path to handle different OS path separators
        normalized_path = os.path.normpath(os.path.abspath(project_path))
        return hashlib.md5(normalized_path.encode()).hexdigest()[:8]

    def _load_settings(self, file_path: str) -> Dict[str, Any]:
        """
        Load settings from a file.

        Args:
            file_path: Path to the settings file

        Returns:
            Dictionary with settings (empty if file not found or invalid)
        """
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, "r") as f:
                settings = json.load(f)

            logger.debug(f"Loaded settings from {file_path}")
            return settings
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading settings from {file_path}: {e}")
            return {}

    def _save_settings(self, file_path: str, settings: Dict[str, Any]) -> bool:
        """
        Save settings to a file.

        Args:
            file_path: Path to the settings file
            settings: Settings dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(settings, f, indent=2)

            logger.debug(f"Saved settings to {file_path}")
            return True
        except IOError as e:
            logger.error(f"Error saving settings to {file_path}: {e}")
            return False

    def _merge_with_defaults(self):
        """Merge loaded settings with default values for missing keys."""

        def deep_merge(source, destination):
            """Deep merge two dictionaries."""
            for key, value in source.items():
                if key in destination:
                    if isinstance(value, dict) and isinstance(destination[key], dict):
                        deep_merge(value, destination[key])
                else:
                    destination[key] = value

        # Start with defaults for global settings
        merged_globals = self.DEFAULT_SETTINGS.copy()
        deep_merge(self.global_settings, merged_globals)
        self.global_settings = merged_globals

        # For project settings, start with merged globals
        merged_project = self.global_settings.copy()
        deep_merge(self.project_settings, merged_project)
        self.project_settings = merged_project

    def get_global(self, key: str, default: Any = None) -> Any:
        """
        Get a global setting value.

        Args:
            key: Setting key (can use dot notation for nested keys)
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        return self._get_nested(self.global_settings, key, default)

    def get_project(self, key: str, default: Any = None) -> Any:
        """
        Get a project-specific setting value.

        Args:
            key: Setting key (can use dot notation for nested keys)
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        return self._get_nested(self.project_settings, key, default)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value (project takes precedence over global).

        Args:
            key: Setting key (can use dot notation for nested keys)
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        # Try project settings first, then global
        project_value = self._get_nested(self.project_settings, key, None)
        if project_value is not None:
            return project_value

        # Fall back to global settings
        return self._get_nested(self.global_settings, key, default)

    def _get_nested(
        self, settings: Dict[str, Any], key: str, default: Any = None
    ) -> Any:
        """
        Get a value from nested dictionaries using dot notation.

        Args:
            settings: Settings dictionary
            key: Setting key (can use dot notation for nested keys)
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        if "." not in key:
            return settings.get(key, default)

        parts = key.split(".")
        current = settings

        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return default
            current = current[part]

        return current.get(parts[-1], default)

    def set_global(self, key: str, value: Any) -> bool:
        """
        Set a global setting value.

        Args:
            key: Setting key (can use dot notation for nested keys)
            value: Setting value

        Returns:
            True if successful, False otherwise
        """
        success = self._set_nested(self.global_settings, key, value)
        if success:
            return self._save_settings(self.global_settings_path, self.global_settings)
        return False

    def set_project(self, key: str, value: Any) -> bool:
        """
        Set a project-specific setting value.

        Args:
            key: Setting key (can use dot notation for nested keys)
            value: Setting value

        Returns:
            True if successful, False otherwise
        """
        success = self._set_nested(self.project_settings, key, value)
        if success:
            return self._save_settings(
                self.project_settings_path, self.project_settings
            )
        return False

    def set(self, key: str, value: Any, scope: str = "project") -> bool:
        """
        Set a setting value.

        Args:
            key: Setting key (can use dot notation for nested keys)
            value: Setting value
            scope: Scope to set ("project" or "global")

        Returns:
            True if successful, False otherwise
        """
        if scope.lower() == "project":
            return self.set_project(key, value)
        elif scope.lower() == "global":
            return self.set_global(key, value)
        else:
            logger.error(f"Invalid scope: {scope}")
            return False

    def _set_nested(self, settings: Dict[str, Any], key: str, value: Any) -> bool:
        """
        Set a value in nested dictionaries using dot notation.

        Args:
            settings: Settings dictionary
            key: Setting key (can use dot notation for nested keys)
            value: Setting value

        Returns:
            True if successful, False otherwise
        """
        if "." not in key:
            settings[key] = value
            return True

        parts = key.split(".")
        current = settings

        # Create nested dictionaries as needed
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value
        return True

    def reset_to_defaults(self, scope: str = "project") -> bool:
        """
        Reset settings to defaults.

        Args:
            scope: Scope to reset ("project", "global", or "all")

        Returns:
            True if successful, False otherwise
        """
        if scope.lower() == "project":
            self.project_settings = {}
            self._merge_with_defaults()
            return self._save_settings(
                self.project_settings_path, self.project_settings
            )

        elif scope.lower() == "global":
            self.global_settings = self.DEFAULT_SETTINGS.copy()
            return self._save_settings(self.global_settings_path, self.global_settings)

        elif scope.lower() == "all":
            self.global_settings = self.DEFAULT_SETTINGS.copy()
            self.project_settings = self.DEFAULT_SETTINGS.copy()
            global_success = self._save_settings(
                self.global_settings_path, self.global_settings
            )
            project_success = self._save_settings(
                self.project_settings_path, self.project_settings
            )
            return global_success and project_success

        else:
            logger.error(f"Invalid scope: {scope}")
            return False

    def remove_setting(self, key: str, scope: str = "project") -> bool:
        """
        Remove a setting.

        Args:
            key: Setting key (can use dot notation for nested keys)
            scope: Scope to remove from ("project" or "global")

        Returns:
            True if successful, False otherwise
        """
        settings = (
            self.project_settings
            if scope.lower() == "project"
            else self.global_settings
        )
        file_path = (
            self.project_settings_path
            if scope.lower() == "project"
            else self.global_settings_path
        )

        if "." not in key:
            if key in settings:
                del settings[key]
                return self._save_settings(file_path, settings)
            return True  # Key doesn't exist, nothing to remove

        parts = key.split(".")
        current = settings

        # Navigate to the parent of the key to remove
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return True  # Key doesn't exist, nothing to remove
            current = current[part]

        # Remove the key
        if parts[-1] in current:
            del current[parts[-1]]
            return self._save_settings(file_path, settings)

        return True  # Key doesn't exist, nothing to remove

    def get_all_settings(self, scope: str = "effective") -> Dict[str, Any]:
        """
        Get all settings.

        Args:
            scope: Scope to get ("project", "global", or "effective")

        Returns:
            Dictionary with all settings
        """
        if scope.lower() == "project":
            return self.project_settings.copy()
        elif scope.lower() == "global":
            return self.global_settings.copy()
        elif scope.lower() == "effective":
            # Return project settings (which include merged globals)
            return self.project_settings.copy()
        else:
            logger.error(f"Invalid scope: {scope}")
            return {}

    def get_overridden_settings(self) -> Dict[str, Any]:
        """
        Get project settings that override global settings.

        Returns:
            Dictionary with overridden settings
        """

        def find_overrides(project_dict, global_dict, path=""):
            """Recursively find overridden settings."""
            overrides = {}

            for key, value in project_dict.items():
                current_path = f"{path}.{key}" if path else key

                if key not in global_dict:
                    # Key only exists in project settings
                    overrides[current_path] = value
                elif isinstance(value, dict) and isinstance(global_dict[key], dict):
                    # Recursively check nested dictionaries
                    nested_overrides = find_overrides(
                        value, global_dict[key], current_path
                    )
                    overrides.update(nested_overrides)
                elif value != global_dict[key]:
                    # Value is different
                    overrides[current_path] = value

            return overrides

        # Find overrides
        return find_overrides(self.project_settings, self.global_settings)


# Create a singleton instance for general use
def get_settings_manager(project_dir: str) -> SettingsManager:
    """
    Get a settings manager for a project.

    Args:
        project_dir: Project directory

    Returns:
        SettingsManager instance
    """
    return SettingsManager(project_dir)
