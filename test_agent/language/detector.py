# test_agent/language/detector.py

import os
from typing import Dict, Optional

from .base import registry, LanguageAdapter


class LanguageDetector:
    """Detects the primary programming language used in a project directory"""

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "python": {
            "extensions": [".py"],
            "config_files": [
                "requirements.txt",
                "setup.py",
                "pyproject.toml",
                "Pipfile",
            ],
            "weight": 1.0,
        },
        "go": {
            "extensions": [".go"],
            "config_files": ["go.mod", "go.sum"],
            "weight": 1.0,
        },
        # Additional languages can be added here
    }

    @classmethod
    def detect_language(cls, project_dir: str) -> Optional[str]:
        """
        Detects the primary programming language used in a project directory.

        Args:
            project_dir: Path to the project directory

        Returns:
            The detected language name or None if detection fails
        """
        if not os.path.exists(project_dir) or not os.path.isdir(project_dir):
            raise ValueError(f"Invalid project directory: {project_dir}")

        # Count language indicators
        language_scores = cls._count_language_indicators(project_dir)

        # If no language indicators found
        if not language_scores:
            return None

        # Return the language with the highest score
        return max(language_scores.items(), key=lambda x: x[1])[0]

    @classmethod
    def _count_language_indicators(cls, project_dir: str) -> Dict[str, float]:
        """
        Counts language indicators in a project directory.

        Args:
            project_dir: Path to the project directory

        Returns:
            Dictionary mapping language names to their scores
        """
        language_scores = {lang: 0.0 for lang in cls.LANGUAGE_PATTERNS}

        # Walk through the project directory
        for root, dirs, files in os.walk(project_dir):
            # Skip hidden directories and common non-source directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "venv", ".git", "__pycache__"]
            ]

            # Check language-specific config files
            for lang, pattern in cls.LANGUAGE_PATTERNS.items():
                for config_file in pattern["config_files"]:
                    if config_file in files:
                        # Config files are strong indicators, give them extra weight
                        language_scores[lang] += 2.0 * pattern["weight"]

            # Count file extensions
            for file in files:
                for lang, pattern in cls.LANGUAGE_PATTERNS.items():
                    if any(file.endswith(ext) for ext in pattern["extensions"]):
                        language_scores[lang] += pattern["weight"]

        # Remove languages with no indicators
        return {lang: score for lang, score in language_scores.items() if score > 0}

    @classmethod
    def get_language_adapter(cls, language: str) -> Optional[LanguageAdapter]:
        """
        Get a language adapter instance for the specified language or extension.

        Args:
            language: Language name or file extension

        Returns:
            Language adapter instance or None if not found
        """
        # Check if it's a file extension
        if language.startswith("."):
            return registry.get_by_extension(language)

        # Otherwise treat it as a language name
        return registry.get_by_language(language)
