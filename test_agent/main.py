# test_agent/main.py

import os
import sys
import time
import asyncio
import argparse
import logging
from typing import Dict, Any, Optional, List

from .workflow import WorkflowState, TestGenerationGraph, TestStatus
from .workflow.state import ProjectInfo, LLMInfo, MemoryInfo, CacheInfo
from .language import get_supported_languages, detect_language, get_adapter
from .llm import list_providers, get_provider
from .memory import MemoryManager, CacheManager
from .utils.api_utils import (
    get_api_key,
    save_api_key,
    get_last_provider,
    save_last_provider,
)
from .utils.logging import setup_logging

# Configure root logger
logger = logging.getLogger("test_agent")


class TestAgent:
    """
    Main entry point for the test generation agent.
    """

    def __init__(
        self,
        project_directory: str,
        language: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        test_directory: Optional[str] = None,
        api_key: Optional[str] = None,
        excluded_dirs: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
        verbose: bool = False,
        cache_enabled: bool = True,
    ):
        """
        Initialize the test agent.

        Args:
            project_directory: Path to the project directory
            language: Programming language to use. If None, will be auto-detected.
            llm_provider: LLM provider to use (claude, openai, deepseek, gemini)
            llm_model: Specific model to use for the provider
            test_directory: Directory to store generated tests
            api_key: API key for the LLM provider
            excluded_dirs: Directories to exclude from analysis
            excluded_files: Files to exclude from analysis
            verbose: Whether to enable verbose logging
            cache_enabled: Whether to enable caching
        """
        self.project_directory = os.path.abspath(project_directory)
        self.language = language
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.test_directory = test_directory
        self.api_key = api_key
        self.excluded_dirs = excluded_dirs or []
        self.excluded_files = excluded_files or []
        self.verbose = verbose
        self.cache_enabled = cache_enabled

        # Set up logging level
        if verbose:
            logger.setLevel(logging.DEBUG)

        # Initialize memory and cache
        self.memory_manager = MemoryManager(self.project_directory)
        self.cache_manager = CacheManager(self.project_directory)

        # Initialize workflow graph
        self.workflow = TestGenerationGraph()

        logger.info(f"Test Agent initialized for project: {self.project_directory}")

    def _initialize_state(self) -> WorkflowState:
        """
        Initialize the workflow state.

        Returns:
            Initial workflow state
        """
        # Create project info
        project_info = ProjectInfo(
            root_directory=self.project_directory,
            language=self.language,
            test_directory=self.test_directory,
            excluded_directories=self.excluded_dirs,
            excluded_files=self.excluded_files,
        )

        # Create LLM info if provider specified
        llm_info = None
        if self.llm_provider:
            llm_info = LLMInfo(
                provider=self.llm_provider,
                model=self.llm_model,
                api_key=self.api_key,
                streaming=True,
                temperature=0.2,
            )

        # Create a MemoryInfo object
        memory_info = MemoryInfo(
            conversation_history=[],
            decisions={},
            cache_info=CacheInfo(cache_enabled=self.cache_enabled),
        )

        # Create state with the memory_info object
        state = WorkflowState(project=project_info, llm=llm_info, memory=memory_info)

        return state

    def _prompt_provider_selection(self) -> str:
        """
        Prompt the user to select an LLM provider.

        Returns:
            Selected provider name
        """
        providers = list_providers()

        print("\nAvailable LLM providers:")
        for i, (name, info) in enumerate(providers.items(), 1):
            print(f"{i}. {name.capitalize()}")

        while True:
            try:
                choice = int(input("\nSelect a provider (enter number): ").strip())
                if 1 <= choice <= len(providers):
                    return list(providers.keys())[choice - 1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

    def _prompt_api_key(self, provider: str) -> str:
        """
        Prompt the user for an API key.

        Args:
            provider: Provider name

        Returns:
            API key entered by the user
        """
        api_key = input(f"Enter API key for {provider.capitalize()}: ")
        return api_key.strip()

    async def run(self) -> Dict[str, Any]:
        """
        Run the test generation workflow.

        Returns:
            Dictionary with results and summary
        """
        # Check if there's a previous provider in config and if LLM provider is specified
        if not self.llm_provider:
            last_provider = get_last_provider()

            if last_provider:
                # Ask user if they want to use the last provider
                use_last = input(
                    f"Use previous LLM provider ({last_provider.capitalize()})? (y/n): "
                )
                if use_last.strip().lower() == "y":
                    self.llm_provider = last_provider
                else:
                    self.llm_provider = self._prompt_provider_selection()
            else:
                self.llm_provider = self._prompt_provider_selection()

        # Save current provider as the last used
        save_last_provider(self.llm_provider)

        # Check if API key is specified
        if not self.api_key:
            # Try to load from saved settings or environment
            self.api_key = get_api_key(self.llm_provider)

            if not self.api_key:
                self.api_key = self._prompt_api_key(self.llm_provider)

                # Ask if user wants to save the key
                save_key = input(
                    "Do you want to save this API key for future use? (y/n): "
                )
                if save_key.strip().lower() == "y":
                    save_api_key(self.llm_provider, self.api_key)

        # Set API key in environment variable for specific providers
        if self.llm_provider.lower() == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = self.api_key
        elif self.llm_provider.lower() == "claude":
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        elif self.llm_provider.lower() == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key
        elif self.llm_provider.lower() == "gemini":
            os.environ["GOOGLE_API_KEY"] = self.api_key

        # Initialize state with the provider info
        state = self._initialize_state()

        # Run workflow
        try:
            logger.info("Starting test generation workflow")

            result = await self.workflow.run(state)

            # Build results dictionary
            # Handle the case where result is a dictionary-like object instead of WorkflowState
            if hasattr(result, "get"):
                # It's a dictionary-like object
                errors = result.get("errors", [])
                project_data = result.get("project", {})
                tests_data = result.get("tests", {})
                start_time = result.get("start_time", 0)
                end_time = result.get("end_time", 0)
            else:
                # It's the original WorkflowState object
                errors = result.errors
                project_data = result.project
                tests_data = result.tests
                start_time = result.start_time
                end_time = result.end_time

            # Get counts
            total_files = 0
            if hasattr(project_data, "source_files"):
                total_files = len(project_data.source_files)
            elif isinstance(project_data, dict) and "source_files" in project_data:
                total_files = len(project_data.get("source_files", []))

            # Handle tests - check type and access attributes safely
            tests_generated = 0
            passed_tests = 0
            failed_tests = 0
            skipped_tests = 0
            test_files = []

            # Process each test item
            if tests_data:
                for test_key, test_info in tests_data.items():
                    # Variables to store test info
                    has_content = False
                    test_status = None
                    test_path = None
                    source_file_path = None

                    # Check object type and extract info accordingly
                    if hasattr(test_info, "content"):  # It's a TestInfo object
                        has_content = bool(test_info.content)
                        test_status = (
                            test_info.status if hasattr(test_info, "status") else None
                        )
                        test_path = (
                            test_info.test_path
                            if hasattr(test_info, "test_path")
                            else None
                        )
                        source_file_path = (
                            test_info.source_file
                            if hasattr(test_info, "source_file")
                            else None
                        )
                    elif isinstance(test_info, dict):  # It's a dictionary
                        has_content = bool(test_info.get("content"))
                        test_status = test_info.get("status")
                        test_path = test_info.get("test_path")
                        source_file_path = test_info.get("source_file")

                    # Count based on extracted info
                    if has_content:
                        tests_generated += 1

                    # Check status (handle both enum and string cases)
                    if test_status:
                        if isinstance(test_status, str):
                            if test_status in ["PASSED", "FIXED"]:
                                passed_tests += 1
                            elif test_status in ["FAILED", "ERROR"]:
                                failed_tests += 1
                            elif test_status == "SKIPPED":
                                skipped_tests += 1
                        else:  # It's a TestStatus enum
                            if test_status in [TestStatus.PASSED, TestStatus.FIXED]:
                                passed_tests += 1
                            elif test_status in [TestStatus.FAILED, TestStatus.ERROR]:
                                failed_tests += 1
                            elif test_status == TestStatus.SKIPPED:
                                skipped_tests += 1

                    # Add to test files list if we have a path
                    if test_path:
                        # Format relative paths
                        rel_path = os.path.relpath(test_path, self.project_directory)
                        rel_source = ""
                        if source_file_path:
                            rel_source = os.path.relpath(
                                source_file_path, self.project_directory
                            )

                        test_files.append(
                            {
                                "path": test_path,
                                "relative_path": rel_path,
                                "status": (
                                    str(test_status) if test_status else "UNKNOWN"
                                ),
                                "source_file": rel_source,
                            }
                        )

            # Get timing
            time_taken = 0
            if start_time and end_time:
                time_taken = end_time - start_time

            # Format errors list
            error_list = []
            if isinstance(errors, list):
                for e in errors:
                    if isinstance(e, dict):
                        error_list.append(
                            {
                                "phase": e.get("phase", "unknown"),
                                "message": e.get("error", "Unknown error"),
                            }
                        )
                    else:
                        # Handle non-dict error objects
                        error_list.append(
                            {
                                "phase": getattr(e, "phase", "unknown"),
                                "message": getattr(e, "error", str(e)),
                            }
                        )

            # Build summary dict
            summary = {
                "status": "success" if not error_list else "error",
                "source_files": total_files,
                "tests_generated": tests_generated,
                "tests_passed": passed_tests,
                "tests_failed": failed_tests,
                "tests_skipped": skipped_tests,
                "time_taken": time_taken,
                "errors": error_list,
                "test_files": test_files,
            }

            # Add cache stats if enabled
            if self.cache_enabled:
                cache_manager = CacheManager(self.project_directory)
                summary["cache_stats"] = cache_manager.get_statistics()

            logger.info(f"Test generation completed with status: {summary['status']}")
            return summary

        except Exception as e:
            logger.exception(f"Error running test generation: {str(e)}")

            return {
                "status": "error",
                "error": str(e),
                "tests_generated": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "time_taken": 0,
                "source_files": 0,
                "errors": [{"phase": "workflow", "message": str(e)}],
            }

    def run_sync(self) -> Dict[str, Any]:
        """
        Synchronous version of run() that creates and manages the event loop.

        Returns:
            Dictionary with results and summary
        """
        # Create or get event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async method
        return loop.run_until_complete(self.run())

    @staticmethod
    def list_supported_languages() -> Dict[str, str]:
        """
        Get a list of supported programming languages.

        Returns:
            Dictionary of language names and descriptions
        """
        return get_supported_languages()

    @staticmethod
    def list_supported_providers() -> Dict[str, Dict[str, Any]]:
        """
        Get a list of supported LLM providers.

        Returns:
            Dictionary of provider information
        """
        return list_providers()


def generate_tests(
    project_directory: str,
    language: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    test_directory: Optional[str] = None,
    api_key: Optional[str] = None,
    excluded_dirs: Optional[List[str]] = None,
    excluded_files: Optional[List[str]] = None,
    verbose: bool = False,
    cache_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Generate tests for a project.

    This is a convenient function that creates a TestAgent and runs it.

    Args:
        project_directory: Path to the project directory
        language: Programming language to use. If None, will be auto-detected.
        llm_provider: LLM provider to use (claude, openai, deepseek, gemini)
        llm_model: Specific model to use for the provider
        test_directory: Directory to store generated tests
        api_key: API key for the LLM provider
        excluded_dirs: Directories to exclude from analysis
        excluded_files: Files to exclude from analysis
        verbose: Whether to enable verbose logging
        cache_enabled: Whether to enable caching

    Returns:
        Dictionary with results and summary
    """
    agent = TestAgent(
        project_directory=project_directory,
        language=language,
        llm_provider=llm_provider,
        llm_model=llm_model,
        test_directory=test_directory,
        api_key=api_key,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        verbose=verbose,
        cache_enabled=cache_enabled,
    )

    return agent.run_sync()


def prompt_api_key(provider: str) -> str:
    """
    Prompt the user for an API key.

    Args:
        provider: Provider name

    Returns:
        API key entered by the user
    """
    api_key = input(f"Enter API key for {provider.capitalize()}: ")
    return api_key.strip()


def prompt_provider_selection() -> str:
    """
    Prompt the user to select an LLM provider.

    Returns:
        Selected provider name
    """
    providers = list_providers()

    print("\nAvailable LLM providers:")
    for i, (name, info) in enumerate(providers.items(), 1):
        print(f"{i}. {name.capitalize()}")

    while True:
        try:
            choice = int(input("\nSelect a provider (enter number): ").strip())
            if 1 <= choice <= len(providers):
                return list(providers.keys())[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Generate tests for a project")

    # Main arguments
    parser.add_argument("project_dir", nargs="?", help="Path to the project directory")
    parser.add_argument(
        "--language", "-l", help="Programming language (auto-detected if not specified)"
    )
    parser.add_argument(
        "--provider", "-p", help="LLM provider (claude, openai, deepseek, gemini)"
    )
    parser.add_argument("--model", "-m", help="Specific model to use (optional)")
    parser.add_argument("--test-dir", "-t", help="Custom test directory (optional)")
    parser.add_argument("--api-key", "-k", help="API key for the LLM provider")

    # Cache and memory options
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cache before running"
    )
    parser.add_argument(
        "--clear-all", action="store_true", help="Clear all caches and settings"
    )

    # Filtering options
    parser.add_argument(
        "--exclude-dir",
        "-e",
        action="append",
        help="Directory to exclude (can be used multiple times)",
    )
    parser.add_argument(
        "--exclude-file",
        "-x",
        action="append",
        help="File to exclude (can be used multiple times)",
    )
    parser.add_argument("--files", "-f", nargs="+", help="Specific files to process")

    # Output options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    parser.add_argument("--log-file", help="Path to save log file")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )

    # Utility commands
    parser.add_argument(
        "--list-languages", action="store_true", help="List supported languages"
    )
    parser.add_argument(
        "--list-providers", action="store_true", help="List supported LLM providers"
    )
    parser.add_argument(
        "--save-key", action="store_true", help="Save API key for a provider"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(
        level=args.log_level.upper(),
        log_file=args.log_file,
        use_colors=True,
    )

    # Handle utility commands
    if args.list_languages:
        languages = get_supported_languages()
        print("Supported languages:")
        for name, desc in languages.items():
            print(f"- {name.capitalize()}: {desc}")
        return 0

    if args.list_providers:
        providers = list_providers()
        print("Supported LLM providers:")
        for name, info in providers.items():
            print(f"- {name.capitalize()}:")
            print(f"  - Default model: {info['default_model']}")
            print(f"  - Available models: {', '.join(info['available_models'])}")
        return 0

    if args.save_key:
        if args.provider:
            provider = args.provider
        else:
            provider = prompt_provider_selection()

        if args.api_key:
            api_key = args.api_key
        else:
            api_key = prompt_api_key(provider)

        save_api_key(provider, api_key)
        print(f"API key saved for {provider.capitalize()}")
        return 0

    # Handle cache clearing
    if args.clear_cache or args.clear_all:
        project_dir = args.project_dir if args.clear_cache else None
        scope = "all"

        from .memory import CacheManager

        cache_manager = CacheManager(project_dir or "")
        result = cache_manager.clear_cache(scope)

        print(f"Cleared {result['hashes']} file hashes")
        print(f"Cleared {result['analysis']} analysis entries")
        print(f"Cleared {result['template']} template entries")
        print("Cache cleared successfully")

        if args.clear_all and not args.project_dir:
            print("All caches cleared")

            # Also clear config if clearing all
            from pathlib import Path

            config_file = Path.home() / ".test_agent" / "config.json"
            if config_file.exists():
                try:
                    config_file.unlink()
                    print("Configuration cleared")
                except IOError:
                    print("Failed to clear configuration")

        if not args.project_dir:
            # If just clearing cache with no project, exit
            return 0

    # Check project directory
    if not args.project_dir:
        parser.print_help()
        print("\nError: Project directory is required")
        return 1

    project_dir = os.path.abspath(args.project_dir)

    if not os.path.exists(project_dir):
        print(f"Error: Project directory not found: {project_dir}")
        return 1

    if not os.path.isdir(project_dir):
        print(f"Error: {project_dir} is not a directory")
        return 1

    # Get language (auto-detect or prompt if not specified)
    language = args.language
    if not language:
        # Try to auto-detect
        print(f"Detecting language for {project_dir}...")
        language = detect_language(project_dir)

        if not language:
            print("Could not auto-detect language")
            language = TestAgent.list_supported_languages()
            if len(language) > 0:
                language = list(language.keys())[0]
            else:
                print("No supported languages found!")
                return 1
        else:
            print(f"Detected language: {language}")

    # Run the test generation
    try:
        print(f"\nGenerating tests for {language.capitalize()} project: {project_dir}")
        if args.provider:
            print(f"Using {args.provider.capitalize()} as the LLM provider")

        print("\nThis may take a few minutes. The agent will:")
        print("1. Analyze your code files")
        print("2. Generate appropriate tests")
        print("3. Execute and verify the tests")
        print("4. Fix any failing tests")
        print("5. Provide a summary of testing coverage")

        print("\nWorking...")

        result = generate_tests(
            project_directory=project_dir,
            language=language,
            llm_provider=args.provider,
            llm_model=args.model,
            test_directory=args.test_dir,
            api_key=args.api_key,
            excluded_dirs=args.exclude_dir,
            excluded_files=args.exclude_file,
            verbose=args.verbose,
            cache_enabled=not args.no_cache,
        )

        # Print summary
        print("\n=== Test Generation Complete ===")
        print(f"Status: {result['status']}")

        if "source_files" in result:
            print(f"Source files analyzed: {result['source_files']}")

        print(f"Tests generated: {result['tests_generated']}")
        print(f"Tests passed: {result['tests_passed']}")
        print(f"Tests failed: {result['tests_failed']}")

        if "tests_skipped" in result:
            print(f"Tests skipped: {result['tests_skipped']}")

        print(f"Time taken: {result.get('time_taken', 0):.2f} seconds")

        # Show errors if any
        if result["errors"]:
            print("\nErrors encountered:")
            for i, error in enumerate(result["errors"][:5], 1):
                print(f"{i}. {error['message']} (in {error['phase']} phase)")

            if len(result["errors"]) > 5:
                print(f"... and {len(result['errors']) - 5} more errors")

        # Show test files
        if "test_files" in result and result["test_files"]:
            print("\nGenerated test files:")

            # Group by status for better readability
            passed_files = []
            failed_files = []

            for test_file in result["test_files"]:
                if test_file["status"] in ["PASSED", "FIXED"]:
                    passed_files.append(test_file["relative_path"])
                else:
                    failed_files.append(test_file["relative_path"])

            # Print passed files first
            for path in sorted(passed_files):
                print(f"✅ {path}")

            # Then print failed files
            for path in sorted(failed_files):
                print(f"❌ {path}")

        return 0 if result["status"] == "success" else 1

    except Exception as e:
        logger.exception(f"Error generating tests: {str(e)}")
        print(f"\nError generating tests: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
