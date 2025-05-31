# test_agent/language/base.py

import os

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LanguageAdapter(ABC):
    """
    Interface for language adapters that provide language-specific functionalities
    while adapting to a common interface.
    """

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the name of the language"""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Return the file extensions for this language"""
        pass

    @property
    @abstractmethod
    def test_file_prefix(self) -> str:
        """Return the prefix or suffix for test files"""
        pass

    @property
    @abstractmethod
    def test_command(self) -> List[str]:
        """Return the command to run tests"""
        pass

    @abstractmethod
    def analyze_source_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a source file to extract structure information"""
        pass

    @abstractmethod
    def analyze_test_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a test file to extract test cases and framework information"""
        pass

    @abstractmethod
    def detect_project_structure(self, project_dir: str) -> Dict[str, Any]:
        """Detect project structure, test patterns, and framework information"""
        pass

    @abstractmethod
    def generate_test_path(
        self,
        source_file: str,
        test_directory: str,
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the proper path for a test file given a source file and pattern"""
        pass

    @abstractmethod
    def generate_test_template(
        self,
        source_file: str,
        analysis: Dict[str, Any],
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a test template based on source analysis and project patterns"""
        pass

    @abstractmethod
    def check_is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file"""
        pass

    @abstractmethod
    def find_corresponding_source(
        self, test_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the source file that a test file is testing"""
        pass

    @abstractmethod
    def find_corresponding_test(
        self, source_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the test file for a given source file"""
        pass

    @abstractmethod
    def get_environment_location(self) -> str:
        """Get the recommended location for any build environment"""
        pass

    def should_skip_file(self, file_path: str) -> bool:
        """
        Check if a file should be skipped when generating tests

        Args:
            file_path: Path to the file

        Returns:
            bool: True if the file should be skipped, False otherwise
        """
        # Skip files in __pycache__ directories
        if "__pycache__" in file_path:
            return True

        # Skip test files
        if self.check_is_test_file(file_path):
            return True

        # Skip non-Python files
        if not file_path.endswith(".py"):
            return True

        # Skip __init__.py files if they're empty
        if os.path.basename(file_path) == "__init__.py":
            try:
                # Check if file is empty or only contains comments/whitespace
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # Skip if the file is empty or only contains comments
                    if not content or all(
                        line.strip().startswith("#")
                        for line in content.splitlines()
                        if line.strip()
                    ):

                        return True
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        project_dir = self._find_project_root(file_path)
        existing_test = self.find_corresponding_test(file_path, project_dir)
        if existing_test and os.path.exists(existing_test):

            return True

        return False

    def _find_project_root(self, file_path: str) -> str:
        """
        Find the project root directory

        Args:
            file_path: Path to a file or directory within the project

        Returns:
            Absolute path to the identified project root
        """
        import os

        # Get absolute path
        file_path = os.path.abspath(file_path)
        if os.path.isfile(file_path):
            file_path = os.path.dirname(file_path)

        # Start from the given directory and traverse upward
        current_dir = file_path
        while current_dir and current_dir != os.path.dirname(current_dir):
            # Check for common project root indicators
            if os.path.exists(os.path.join(current_dir, ".git")):
                return current_dir

            # Language-specific indicators
            if self.language_name == "python":
                if any(
                    os.path.exists(os.path.join(current_dir, f))
                    for f in ["setup.py", "pyproject.toml", "requirements.txt"]
                ):
                    return current_dir
            elif self.language_name == "go":
                if os.path.exists(os.path.join(current_dir, "go.mod")):
                    return current_dir

            # Move up one directory
            current_dir = os.path.dirname(current_dir)

        # If no definite root found, return the original directory
        return file_path


class LanguageAdapterRegistry:
    """Registry for language adapters"""

    def __init__(self):
        """Initialize the registry"""
        self._adapters = {}
        self._language_adapters = {}

    def register(self, extensions: List[str], adapter_class: type) -> None:
        """
        Register a language adapter for specific file extensions.

        Args:
            extensions: List of file extensions that this adapter handles
            adapter_class: The adapter class to register
        """
        # Register by extension
        for ext in extensions:
            self._adapters[ext] = adapter_class

        # Create an instance to get the language name
        instance = adapter_class()
        self._language_adapters[instance.language_name.lower()] = adapter_class

    def get_by_extension(self, extension: str) -> Optional[LanguageAdapter]:
        """
        Get a language adapter instance for a specific file extension.

        Args:
            extension: File extension (e.g. '.py')

        Returns:
            LanguageAdapter instance or None if not registered
        """
        adapter_class = self._adapters.get(extension)
        if adapter_class:
            return adapter_class()
        return None

    def get_by_language(self, language: str) -> Optional[LanguageAdapter]:
        """
        Get a language adapter instance for a language by name.

        Args:
            language: Name of the language (e.g. 'python')

        Returns:
            LanguageAdapter instance or None if not registered
        """
        adapter_class = self._language_adapters.get(language.lower())
        if adapter_class:
            return adapter_class()
        return None

    def get_all_languages(self) -> List[str]:
        """
        Get list of all registered languages.

        Returns:
            List of language names
        """
        return list(self._language_adapters.keys())


# Global registry instance
registry = LanguageAdapterRegistry()
