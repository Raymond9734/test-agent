# test_agent/cli.py

import os
import sys
import argparse
import logging
from typing import Optional, Dict

from test_agent.main import generate_tests
from .language import get_supported_languages, detect_language
from .llm import list_providers
from .utils.api_utils import (
    get_api_key,
    save_api_key,
    get_last_provider,
    save_last_provider,
)

# Configure logging
logger = logging.getLogger("test_agent.cli")


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


def prompt_language_selection() -> str:
    """
    Prompt the user to select a programming language.

    Returns:
        Selected language name
    """
    languages = get_supported_languages()

    print("\nAvailable languages:")
    for i, (name, _) in enumerate(languages.items(), 1):
        print(f"{i}. {name.capitalize()}")

    while True:
        try:
            choice = int(input("\nSelect a language (enter number): ").strip())
            if 1 <= choice <= len(languages):
                return list(languages.keys())[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def setup_logging(verbose: bool, log_file: Optional[str] = None):
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable verbose logging
        log_file: Optional path to log file
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def clear_cache(
    project_dir: Optional[str] = None, scope: str = "all"
) -> Dict[str, int]:
    """
    Clear the cache for a project.

    Args:
        project_dir: Project directory (if None, clears all projects)
        scope: Cache scope to clear ("all", "analysis", "template", "hashes")

    Returns:
        Dictionary with count of entries cleared by cache type
    """
    from .memory import CacheManager

    if scope == "all":
        cache_type = None
    else:
        cache_type = scope

    # Create cache manager for the project or an empty string for global cache
    cache_manager = CacheManager(project_dir or "")

    # Clear cache
    result = cache_manager.clear_cache(cache_type)

    return result


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
    setup_logging(args.verbose, args.log_file)

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

        result = clear_cache(project_dir, scope)

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
            language = prompt_language_selection()
        else:
            print(f"Detected language: {language}")

    # Get provider and API key
    provider = args.provider
    if not provider:
        # Check for last used provider
        last_provider = get_last_provider()

        if last_provider:
            use_last = input(
                f"Use previous LLM provider ({last_provider.capitalize()})? (y/n): "
            )
            if use_last.strip().lower() == "y":
                provider = last_provider
            else:
                provider = prompt_provider_selection()
        else:
            provider = prompt_provider_selection()

        # Save the selected provider
        save_last_provider(provider)

    api_key = args.api_key
    if not api_key:
        # Try to load from saved settings or environment
        api_key = get_api_key(provider)

        if not api_key:
            api_key = prompt_api_key(provider)

            # Ask if user wants to save the key
            save_key = input("Do you want to save this API key for future use? (y/n): ")
            if save_key.strip().lower() == "y":
                save_api_key(provider, api_key)

    # Run the test generation
    try:
        print(f"\nGenerating tests for {language.capitalize()} project: {project_dir}")
        print(f"Using {provider.capitalize()} as the LLM provider")

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
            llm_provider=provider,
            llm_model=args.model,
            test_directory=args.test_dir,
            api_key=api_key,
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
