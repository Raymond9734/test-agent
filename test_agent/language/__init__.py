# test_agent/language/__init__.py

from typing import Dict, Type, Optional

from .base import LanguageAdapter, registry
from .detector import LanguageDetector

# Import language adapters to register them
from .python.adapter import PythonAdapter  # noqa: F401
from .go.adapter import GoAdapter  # noqa: F401

__ALL__ = [
    "GoAdapter",
    "PythonAdapter",
]


# Create simple API for getting language adapters
def get_adapter(language: str) -> Optional[LanguageAdapter]:
    """
    Get a language adapter instance for the specified language.

    Args:
        language: The language name or file extension

    Returns:
        An instance of the appropriate language adapter
    """
    return LanguageDetector.get_language_adapter(language)


def get_supported_languages() -> Dict[str, str]:
    """
    Get a dictionary of supported languages and their descriptions.

    Returns:
        Dictionary mapping language names to descriptions
    """
    adapters = {}
    for language in registry.get_all_languages():
        adapter = registry.get_by_language(language)
        if adapter:
            adapters[language] = f"{language.capitalize()} language adapter"
    return adapters


def detect_language(project_dir: str) -> Optional[str]:
    """
    Detect the primary programming language used in a project directory.

    Args:
        project_dir: Path to the project directory

    Returns:
        The detected language name or None if detection fails
    """
    return LanguageDetector.detect_language(project_dir)
