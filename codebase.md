# .gitignore

```
# Python bytecode
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Distribution / packaging
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Python virtual environments
env/
venv/
ENV/
env.bak/
venv.bak/
.env
.venv
test_agent_venv/

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml
*.cover
.hypothesis/
.tox/
nosetests.xml
coverage/
.coverage.*
junit-*.xml

# Go specific
*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out
/vendor/
/Godeps/
/gopath/
coverage.out

# Temporary files created by the test_agent
.test_agent_cache/
.test_agent_memory/
*.bak.*

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store
.spyderproject
.spyproject
.ropeproject
.project
.pydevproject
.settings/
.classpath
*.sublime-workspace
*.sublime-project

# API keys and secrets
.env
*.pem
config.json
.test_agent/config.json

# Log files
*.log
logs/
```

# requirements.txt

```txt
aiohappyeyeballs==2.6.1
aiohttp==3.11.18
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.9.0
asttokens==3.0.0
attrs==25.3.0
certifi==2025.4.26
charset-normalizer==3.4.2
decorator==5.2.1
executing==2.2.0
frozenlist==1.6.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.10
ipython==9.2.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
jsonpatch==1.33
jsonpointer==3.0.0
langchain-core==0.3.60
langgraph==0.4.5
langgraph-checkpoint==2.0.26
langgraph-prebuilt==0.1.8
langgraph-sdk==0.1.69
langsmith==0.3.42
matplotlib-inline==0.1.7
multidict==6.4.4
orjson==3.10.18
ormsgpack==1.9.1
packaging==24.2
parso==0.8.4
pexpect==4.9.0
prompt_toolkit==3.0.51
propcache==0.3.1
ptyprocess==0.7.0
pure_eval==0.2.3
pydantic==2.11.4
pydantic_core==2.33.2
Pygments==2.19.1
PyYAML==6.0.2
requests==2.32.3
requests-toolbelt==1.0.0
sniffio==1.3.1
stack-data==0.6.3
tenacity==9.1.2
traitlets==5.14.3
typing-inspection==0.4.0
typing_extensions==4.13.2
urllib3==2.4.0
wcwidth==0.2.13
xxhash==3.5.0
yarl==1.20.0
zstandard==0.23.0

```

# setup.py

```py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="test-agent",
    version="0.1.0",
    description="An LLM-powered test generation agent",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "test-agent=test_agent.main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

```

# test_agent.egg-info/dependency_links.txt

```txt


```

# test_agent.egg-info/entry_points.txt

```txt
[console_scripts]
test-agent = test_agent.main:main

```

# test_agent.egg-info/PKG-INFO

```
Metadata-Version: 2.4
Name: test-agent
Version: 0.1.0
Summary: An LLM-powered test generation agent
Author: Your Name
Author-email: your.email@example.com
Classifier: Development Status :: 3 - Alpha
Classifier: Intended Audience :: Developers
Classifier: Programming Language :: Python :: 3
Classifier: Programming Language :: Python :: 3.8
Classifier: Programming Language :: Python :: 3.9
Classifier: Programming Language :: Python :: 3.10
Requires-Python: >=3.8
Requires-Dist: aiohappyeyeballs==2.6.1
Requires-Dist: aiohttp==3.11.18
Requires-Dist: aiosignal==1.3.2
Requires-Dist: annotated-types==0.7.0
Requires-Dist: anyio==4.9.0
Requires-Dist: asttokens==3.0.0
Requires-Dist: attrs==25.3.0
Requires-Dist: certifi==2025.4.26
Requires-Dist: charset-normalizer==3.4.2
Requires-Dist: decorator==5.2.1
Requires-Dist: executing==2.2.0
Requires-Dist: frozenlist==1.6.0
Requires-Dist: h11==0.16.0
Requires-Dist: httpcore==1.0.9
Requires-Dist: httpx==0.28.1
Requires-Dist: idna==3.10
Requires-Dist: ipython==9.2.0
Requires-Dist: ipython_pygments_lexers==1.1.1
Requires-Dist: jedi==0.19.2
Requires-Dist: jsonpatch==1.33
Requires-Dist: jsonpointer==3.0.0
Requires-Dist: langchain-core==0.3.60
Requires-Dist: langgraph==0.4.5
Requires-Dist: langgraph-checkpoint==2.0.26
Requires-Dist: langgraph-prebuilt==0.1.8
Requires-Dist: langgraph-sdk==0.1.69
Requires-Dist: langsmith==0.3.42
Requires-Dist: matplotlib-inline==0.1.7
Requires-Dist: multidict==6.4.4
Requires-Dist: orjson==3.10.18
Requires-Dist: ormsgpack==1.9.1
Requires-Dist: packaging==24.2
Requires-Dist: parso==0.8.4
Requires-Dist: pexpect==4.9.0
Requires-Dist: prompt_toolkit==3.0.51
Requires-Dist: propcache==0.3.1
Requires-Dist: ptyprocess==0.7.0
Requires-Dist: pure_eval==0.2.3
Requires-Dist: pydantic==2.11.4
Requires-Dist: pydantic_core==2.33.2
Requires-Dist: Pygments==2.19.1
Requires-Dist: PyYAML==6.0.2
Requires-Dist: requests==2.32.3
Requires-Dist: requests-toolbelt==1.0.0
Requires-Dist: sniffio==1.3.1
Requires-Dist: stack-data==0.6.3
Requires-Dist: tenacity==9.1.2
Requires-Dist: traitlets==5.14.3
Requires-Dist: typing-inspection==0.4.0
Requires-Dist: typing_extensions==4.13.2
Requires-Dist: urllib3==2.4.0
Requires-Dist: wcwidth==0.2.13
Requires-Dist: xxhash==3.5.0
Requires-Dist: yarl==1.20.0
Requires-Dist: zstandard==0.23.0
Dynamic: author
Dynamic: author-email
Dynamic: classifier
Dynamic: requires-dist
Dynamic: requires-python
Dynamic: summary

```

# test_agent.egg-info/requires.txt

```txt
aiohappyeyeballs==2.6.1
aiohttp==3.11.18
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.9.0
asttokens==3.0.0
attrs==25.3.0
certifi==2025.4.26
charset-normalizer==3.4.2
decorator==5.2.1
executing==2.2.0
frozenlist==1.6.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.10
ipython==9.2.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
jsonpatch==1.33
jsonpointer==3.0.0
langchain-core==0.3.60
langgraph==0.4.5
langgraph-checkpoint==2.0.26
langgraph-prebuilt==0.1.8
langgraph-sdk==0.1.69
langsmith==0.3.42
matplotlib-inline==0.1.7
multidict==6.4.4
orjson==3.10.18
ormsgpack==1.9.1
packaging==24.2
parso==0.8.4
pexpect==4.9.0
prompt_toolkit==3.0.51
propcache==0.3.1
ptyprocess==0.7.0
pure_eval==0.2.3
pydantic==2.11.4
pydantic_core==2.33.2
Pygments==2.19.1
PyYAML==6.0.2
requests==2.32.3
requests-toolbelt==1.0.0
sniffio==1.3.1
stack-data==0.6.3
tenacity==9.1.2
traitlets==5.14.3
typing-inspection==0.4.0
typing_extensions==4.13.2
urllib3==2.4.0
wcwidth==0.2.13
xxhash==3.5.0
yarl==1.20.0
zstandard==0.23.0

```

# test_agent.egg-info/SOURCES.txt

```txt
setup.py
test_agent/__init__.py
test_agent/cli.py
test_agent/config.py
test_agent/main.py
test_agent.egg-info/PKG-INFO
test_agent.egg-info/SOURCES.txt
test_agent.egg-info/dependency_links.txt
test_agent.egg-info/entry_points.txt
test_agent.egg-info/requires.txt
test_agent.egg-info/top_level.txt
test_agent/language/__init__.py
test_agent/language/base.py
test_agent/language/detector.py
test_agent/language/go/__init__.py
test_agent/language/go/adapter.py
test_agent/language/go/parser.py
test_agent/language/python/__init__.py
test_agent/language/python/adapter.py
test_agent/language/python/parser.py
test_agent/llm/__init__.py
test_agent/llm/base.py
test_agent/llm/claude.py
test_agent/llm/deepseek.py
test_agent/llm/gemini.py
test_agent/llm/openai.py
test_agent/llm/prompts/__init__.py
test_agent/llm/prompts/language_detection.py
test_agent/llm/prompts/test_fixing.py
test_agent/llm/prompts/test_generation.py
test_agent/memory/__init__.py
test_agent/memory/cache.py
test_agent/memory/conversation.py
test_agent/memory/settings.py
test_agent/tools/__init__.py
test_agent/tools/environment_tools.py
test_agent/tools/file_tools.py
test_agent/tools/test_tools.py
test_agent/utils/__init__.py
test_agent/utils/api_utils.py
test_agent/utils/logging.py
test_agent/utils/security.py
test_agent/workflow/__init__.py
test_agent/workflow/graph.py
test_agent/workflow/state.py
test_agent/workflow/nodes/__init__.py
test_agent/workflow/nodes/complete.py
test_agent/workflow/nodes/error.py
test_agent/workflow/nodes/file_analysis.py
test_agent/workflow/nodes/initialization.py
test_agent/workflow/nodes/language_detection.py
test_agent/workflow/nodes/project_analysis.py
test_agent/workflow/nodes/test_execution.py
test_agent/workflow/nodes/test_fixing.py
test_agent/workflow/nodes/test_generation.py
test_agent/workflow/nodes/test_path.py
```

# test_agent.egg-info/top_level.txt

```txt
test_agent

```

# test_agent/__init__.py

```py
# test_agent/__init__.py

from .main import TestAgent, generate_tests
from .language import get_supported_languages, detect_language
from .llm import list_providers

__all__ = [
    "TestAgent",
    "generate_tests",
    "get_supported_languages",
    "detect_language",
    "list_providers",
]

__version__ = "0.1.0"

```

# test_agent/cli.py

```py
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

```

# test_agent/config.py

```py
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

```

# test_agent/language/__init__.py

```py
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


def register_adapter(extensions: list, adapter_class: Type[LanguageAdapter]) -> None:
    """
    Register a custom language adapter.

    Args:
        extensions: List of file extensions that this adapter handles
        adapter_class: The adapter class to register
    """
    registry.register(extensions, adapter_class)

```

# test_agent/language/base.py

```py
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

```

# test_agent/language/detector.py

```py
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

```

# test_agent/language/go/__init__.py

```py

```

# test_agent/language/go/adapter.py

```py
# test_agent/language/go/adapter.py

import os
import re
from typing import List, Dict, Any, Optional

from ..base import LanguageAdapter, registry


class GoAdapter(LanguageAdapter):
    """Go language adapter with enhanced analysis and project detection"""

    def __init__(self):
        """Initialize the Go adapter"""
        self._project_roots_cache = {}
        self._test_patterns_cache = {}
        self._module_name_cache = {}

    @property
    def language_name(self) -> str:
        """Return the name of the language"""
        return "go"

    @property
    def file_extensions(self) -> List[str]:
        """Return the file extensions for this language"""
        return [".go"]

    @property
    def test_file_prefix(self) -> str:
        """
        Return the prefix for test files.
        Go uses a suffix (_test.go) rather than a prefix.
        """
        return ""  # No prefix - Go uses suffix _test.go

    @property
    def test_command(self) -> List[str]:
        """Return the command to run tests"""
        return ["go", "test"]

    def analyze_source_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Go source file to extract structure"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic parsing using regular expressions

            # Extract package name
            package_match = re.search(r"package\s+(\w+)", content)
            package = package_match.group(1) if package_match else "unknown"

            # Extract imports
            imports = []
            import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
            for block in import_blocks:
                for line in block.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Extract quoted import path
                        import_match = re.search(r'"([^"]+)"', line)
                        if import_match:
                            imports.append(import_match.group(1))

            # Also catch single line imports
            single_imports = re.findall(r'import\s+"([^"]+)"', content)
            imports.extend(single_imports)

            # Extract function definitions (not methods)
            function_matches = re.findall(
                r"func\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
            )
            functions = []
            for name, args, ret in function_matches:
                args_clean = self._clean_go_params(args)
                ret_clean = self._clean_go_returns(ret)

                functions.append(
                    {"name": name, "args": args_clean, "returns": ret_clean}
                )

            # Extract struct definitions
            struct_matches = re.findall(
                r"type\s+(\w+)\s+struct\s+{(.*?)}", content, re.DOTALL
            )
            structs = []
            for name, fields_str in struct_matches:
                fields = []
                # Match field names and types - handles multi-line definitions
                field_matches = re.findall(r"(\w+)(?:\s+([*\[\]\w\.]+))", fields_str)
                for field_name, field_type in field_matches:
                    fields.append({"name": field_name, "type": field_type.strip()})

                structs.append({"name": name, "fields": fields})

            # Extract method definitions (functions with receivers)
            method_matches = re.findall(
                r"func\s+\(([\w\s*]+)\)\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
            )
            methods = []
            for receiver, name, args, ret in method_matches:
                receiver_parts = receiver.strip().split()
                receiver_var = receiver_parts[0] if len(receiver_parts) > 0 else ""
                receiver_type = (
                    receiver_parts[-1].replace("*", "")
                    if len(receiver_parts) > 1
                    else receiver_parts[0].replace("*", "")
                )

                args_clean = self._clean_go_params(args)
                ret_clean = self._clean_go_returns(ret)

                methods.append(
                    {
                        "receiver_var": receiver_var,
                        "receiver_type": receiver_type,
                        "name": name,
                        "args": args_clean,
                        "returns": ret_clean,
                    }
                )

            # Extract interfaces
            interface_matches = re.findall(
                r"type\s+(\w+)\s+interface\s+{(.*?)}", content, re.DOTALL
            )
            interfaces = []
            for name, methods_str in interface_matches:
                interface_methods = []

                # Extract method signatures
                method_matches = re.findall(
                    r"(\w+)(?:\((.*?)\))(?:\s*(.*?))?(?:$|\n)",
                    methods_str,
                    re.MULTILINE,
                )
                for method_match in method_matches:
                    method_name = method_match[0]
                    method_args = (
                        self._clean_go_params(method_match[1])
                        if len(method_match) > 1
                        else []
                    )
                    method_returns = (
                        self._clean_go_returns(method_match[2])
                        if len(method_match) > 2
                        else None
                    )

                    interface_methods.append(
                        {
                            "name": method_name,
                            "args": method_args,
                            "returns": method_returns,
                        }
                    )

                interfaces.append({"name": name, "methods": interface_methods})

            # Get module info from go.mod
            module_name = self._get_module_name(file_path)

            return {
                "file": file_path,
                "package": package,
                "module": module_name,
                "imports": imports,
                "functions": functions,
                "structs": structs,
                "methods": methods,
                "interfaces": interfaces,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _clean_go_params(self, params_str: str) -> List[Dict[str, str]]:
        """Clean and parse Go parameters"""
        params = []

        if not params_str.strip():
            return params

        # Split by commas, but handle multiple parameters of the same type
        param_groups = re.findall(r"((?:\w+(?:,\s*\w+)*)\s+[\*\[\]\w\.]+)", params_str)

        for group in param_groups:
            parts = group.split()
            if len(parts) >= 2:
                param_type = parts[-1]
                param_names = parts[0].split(",")

                for name in param_names:
                    name = name.strip()
                    if name:
                        params.append({"name": name, "type": param_type})

        return params

    def _clean_go_returns(self, returns_str: str) -> Optional[List[Dict[str, str]]]:
        """Clean and parse Go return values"""
        returns_str = returns_str.strip()

        if not returns_str:
            return None

        returns = []

        # Handle single unnamed return
        if "(" not in returns_str and "," not in returns_str:
            return [{"type": returns_str.strip()}]

        # Handle named returns or multiple returns
        if returns_str.startswith("("):
            # Remove parentheses
            returns_str = (
                returns_str[1:-1] if returns_str.endswith(")") else returns_str[1:]
            )

        # Split by commas
        for ret in returns_str.split(","):
            ret = ret.strip()
            if ret:
                # Check for named return
                parts = ret.split()
                if len(parts) == 2:
                    returns.append({"name": parts[0], "type": parts[1]})
                else:
                    returns.append({"type": ret})

        return returns

    def analyze_test_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Go test file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract package name
            package_match = re.search(r"package\s+(\w+)", content)
            package = package_match.group(1) if package_match else "unknown"

            # Extract test functions
            test_funcs = re.findall(r"func\s+(Test\w+)\s*\(", content)
            benchmark_funcs = re.findall(r"func\s+(Benchmark\w+)\s*\(", content)
            example_funcs = re.findall(r"func\s+(Example\w+)\s*\(", content)

            # Extract import paths for source package
            imports = []
            import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
            for block in import_blocks:
                for line in block.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Extract quoted import path
                        import_match = re.search(r'"([^"]+)"', line)
                        if import_match:
                            imports.append(import_match.group(1))

            # Also catch single line imports
            single_imports = re.findall(r'import\s+"([^"]+)"', content)
            imports.extend(single_imports)

            # Determine if it uses the _test package convention
            is_test_package = "_test" in package

            # Find source file being tested
            project_root = self._find_project_root(file_path)
            source_file = self.find_corresponding_source(file_path, project_root)

            return {
                "file": file_path,
                "source_file": source_file,
                "package": package,
                "is_test_package": is_test_package,
                "test_functions": test_funcs,
                "benchmark_functions": benchmark_funcs,
                "example_functions": example_funcs,
                "imports": imports,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _get_module_name(self, file_path: str) -> str:
        """
        Extract the Go module name from go.mod file

        Args:
            file_path: Path to any file in the Go project

        Returns:
            Module name or empty string if not found
        """
        # Check cache first
        project_root = self._find_project_root(file_path)
        if project_root in self._module_name_cache:
            return self._module_name_cache[project_root]

        # Find go.mod file
        go_mod_path = os.path.join(project_root, "go.mod")
        if not os.path.exists(go_mod_path):
            self._module_name_cache[project_root] = ""
            return ""

        try:
            with open(go_mod_path, "r") as f:
                content = f.read()

            # Extract module name
            module_match = re.search(r"module\s+([^\s]+)", content)
            if module_match:
                module_name = module_match.group(1)
                self._module_name_cache[project_root] = module_name
                return module_name
        except Exception:
            pass

        self._module_name_cache[project_root] = ""
        return ""

    def detect_project_structure(self, project_dir: str) -> Dict[str, Any]:
        """Detect project structure, test patterns, and framework information"""
        # Check cache first
        project_dir = os.path.abspath(project_dir)
        if project_dir in self._test_patterns_cache:
            return self._test_patterns_cache[project_dir]

        # Go projects typically have tests in the same directory as source files
        # But some larger projects might have a /test directory
        test_directories = []
        test_files = []

        # Look for test directories
        for root, dirs, _ in os.walk(project_dir):
            for dir_name in dirs:
                if dir_name.lower() in ["test", "tests", "testing"]:
                    test_directories.append(os.path.join(root, dir_name))

        # Look for test files
        for root, _, files in os.walk(project_dir):
            for file_name in files:
                if file_name.endswith("_test.go"):
                    test_files.append(os.path.join(root, file_name))

        # Analyze test location patterns
        location_pattern = "same_directory"  # Default for Go

        if test_files:
            # Count location patterns
            location_counts = {
                "same_directory": 0,
                "tests_directory": 0,
            }

            # Also check for test package pattern
            package_patterns = {
                "same_package": 0,
                "package_test": 0,
            }

            for test_file in test_files:
                # Check location pattern
                test_dir = os.path.dirname(test_file)
                dir_name = os.path.basename(test_dir).lower()

                if dir_name in ["test", "tests", "testing"]:
                    location_counts["tests_directory"] += 1
                else:
                    location_counts["same_directory"] += 1

                    analysis = self.analyze_test_file(test_file)
                    if analysis.get("is_test_package"):
                        package_patterns["package_test"] += 1
                    else:
                        package_patterns["same_package"] += 1

            # Determine primary patterns
            if location_counts:
                location_pattern = max(location_counts.items(), key=lambda x: x[1])[0]

            # Determine package pattern
            package_pattern = "same_package"  # Default
            if package_patterns["package_test"] > package_patterns["same_package"]:
                package_pattern = "package_test"

        # Get module name
        module_name = self._get_module_name(project_dir)

        # Create result
        result = {
            "test_directories": test_directories,
            "test_files": test_files,
            "location_pattern": location_pattern,
            "package_pattern": package_pattern,
            "module": module_name,
        }

        # Cache result
        self._test_patterns_cache[project_dir] = result

        return result

    def generate_test_path(
        self,
        source_file: str,
        test_directory: str,
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the proper path for a test file given a source file and pattern"""
        # Ensure absolute paths
        source_file = os.path.abspath(source_file)
        test_directory = os.path.abspath(test_directory)

        # First, check if test already exists
        project_root = self._find_project_root(source_file)
        existing_test = self.find_corresponding_test(source_file, project_root)
        if existing_test and os.path.exists(existing_test):
            return existing_test

        # Use provided pattern or detect it
        if pattern is None:
            pattern = self.detect_project_structure(project_root)

        # Extract filename components
        source_name = os.path.basename(source_file)
        source_base, source_ext = os.path.splitext(source_name)

        # For Go, the test file is always named {source_base}_test.go
        test_filename = f"{source_base}_test{source_ext}"

        # Generate path based on location pattern
        location_pattern = pattern.get("location_pattern", "same_directory")

        if location_pattern == "same_directory":
            # Go standard is tests in the same directory
            return os.path.join(os.path.dirname(source_file), test_filename)
        else:  # tests_directory
            # Use test_directory - but need to match package structure
            rel_path = os.path.relpath(os.path.dirname(source_file), project_root)
            test_path = os.path.join(test_directory, rel_path)
            os.makedirs(test_path, exist_ok=True)
            return os.path.join(test_path, test_filename)

    def generate_test_template(
        self,
        source_file: str,
        analysis: Dict[str, Any],
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a test template based on project patterns and file analysis"""
        # Use provided pattern or detect it
        if pattern is None:
            project_dir = self._find_project_root(source_file)
            pattern = self.detect_project_structure(project_dir)

        package_name = analysis.get("package", "main")
        package_pattern = pattern.get("package_pattern", "same_package")

        # Determine the package for the test file
        if package_pattern == "package_test":
            test_package = f"{package_name}_test"
        else:
            test_package = package_name

        # Check if we need to import the source package
        import_source = package_pattern == "package_test"

        module_name = pattern.get("module", "")
        source_import = ""
        if import_source and module_name:
            # If module name is available, use it to construct the import path
            project_dir = self._find_project_root(source_file)
            rel_source_dir = os.path.dirname(os.path.relpath(source_file, project_dir))
            if rel_source_dir:
                source_import = f'\n\t"{module_name}/{rel_source_dir}"'
            else:
                source_import = f'\n\t"{module_name}"'

        template = f"""package {test_package}

import (
\t"testing"{source_import}
)

// Generated test file for {os.path.basename(source_file)}
"""

        # Add tests for standalone functions
        for func in analysis.get("functions", []):
            template += f"""
func Test{func['name']}(t *testing.T) {{
\t// TODO: Write test for {func['name']}
\tt.Run("{func['name']}", func(t *testing.T) {{
\t\t// TODO: Add test cases
\t\t// result := {package_name if import_source else ""}.{func['name']}()
\t\t// if result != expected {{
\t\t//     t.Errorf("Expected %v, got %v", expected, result)
\t\t// }}
\t}})
}}
"""

        # Add tests for methods
        for method in analysis.get("methods", []):
            receiver_type = method.get("receiver_type", "")
            method_name = method.get("name", "")

            if receiver_type and method_name:
                template += f"""
func Test{receiver_type}_{method_name}(t *testing.T) {{
\t// TODO: Write test for {receiver_type}.{method_name}
\tt.Run("{method_name}", func(t *testing.T) {{
\t\t// Create an instance of {receiver_type}
\t\t// instance := {package_name + "." if import_source else ""}{receiver_type}{{}}
\t\t
\t\t// TODO: Add test cases
\t\t// result := instance.{method_name}()
\t\t// if result != expected {{
\t\t//     t.Errorf("Expected %v, got %v", expected, result)
\t\t// }}
\t}})
}}
"""

        # Add table-driven test example if there are functions or methods
        if analysis.get("functions") or analysis.get("methods"):
            template += """
// Example of table-driven tests
func TestTableDriven(t *testing.T) {
\t// Define test cases
\ttests := []struct {
\t\tname     string
\t\tinput    string
\t\texpected string
\t}{
\t\t{"test case 1", "input1", "expected1"},
\t\t{"test case 2", "input2", "expected2"},
\t}

\t// Run test cases
\tfor _, tc := range tests {
\t\tt.Run(tc.name, func(t *testing.T) {
\t\t\t// TODO: Call the function/method being tested
\t\t\t// result := YourFunction(tc.input)
\t\t\t
\t\t\t// if result != tc.expected {
\t\t\t//     t.Errorf("Expected %v, got %v", tc.expected, result)
\t\t\t// }
\t\t})
\t}
}
"""

        return template

    def check_is_test_file(self, file_path: str) -> bool:
        """Check if a file is a Go test file"""
        return file_path.endswith("_test.go")

    def find_corresponding_source(
        self, test_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the source file that a test file is testing"""
        if not self.check_is_test_file(test_file):
            return None

        # Go test files are usually in the same directory as the source file
        # with the naming pattern: source.go -> source_test.go
        file_name = os.path.basename(test_file)

        # Extract source file name: remove _test.go suffix
        if file_name.endswith("_test.go"):
            source_name = file_name[:-8] + ".go"  # Replace _test.go with .go

            # Check in the same directory
            source_path = os.path.join(os.path.dirname(test_file), source_name)
            if os.path.exists(source_path):
                return source_path

        # If source not found and test is in a test directory, check parent directory
        test_dir = os.path.dirname(test_file)
        test_dir_name = os.path.basename(test_dir).lower()
        if test_dir_name in ["test", "tests", "testing"]:
            parent_dir = os.path.dirname(test_dir)
            source_path = os.path.join(parent_dir, source_name)
            if os.path.exists(source_path):
                return source_path

        # If still not found, search the project
        for root, _, files in os.walk(project_dir):
            if source_name in files:
                return os.path.join(root, source_name)

        return None

    def find_corresponding_test(
        self, source_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the test file for a given source file"""
        # Skip if it's already a test file
        if self.check_is_test_file(source_file):
            return None

        file_name = os.path.basename(source_file)
        source_base, source_ext = os.path.splitext(file_name)
        source_dir = os.path.dirname(source_file)

        # Generate test file name (Go standard pattern)
        test_name = f"{source_base}_test{source_ext}"

        # Check in the same directory (Go standard)
        test_path = os.path.join(source_dir, test_name)
        if os.path.exists(test_path):
            return test_path

        # Check in a test directory with the same relative path
        test_dirs = []

        # Look for test directories in project
        for root, dirs, _ in os.walk(project_dir):
            for dir_name in dirs:
                if dir_name.lower() in ["test", "tests", "testing"]:
                    test_dirs.append(os.path.join(root, dir_name))

        # Check each test directory
        rel_source_dir = os.path.relpath(source_dir, project_dir)
        for test_dir in test_dirs:
            if os.path.isdir(test_dir):
                # Check for direct match in test dir
                test_path = os.path.join(test_dir, test_name)
                if os.path.exists(test_path):
                    return test_path

                # Check with matching directory structure
                mirror_dir = os.path.join(test_dir, rel_source_dir)
                if os.path.isdir(mirror_dir):
                    test_path = os.path.join(mirror_dir, test_name)
                    if os.path.exists(test_path):
                        return test_path

        return None

    def get_environment_location(self) -> str:
        """
        Get the recommended location for the build environment.
        For Go, we don't really need a special environment as 'go test' handles it.
        """
        # No special environment needed for Go
        return ""


# Register the adapter
registry.register([".go"], GoAdapter)

```

# test_agent/language/go/parser.py

```py
# test_agent/language/go/parser.py

import os
import logging
import re
import subprocess
from typing import Dict, List, Any, Set

# Configure logging
logger = logging.getLogger(__name__)


class GoParser:
    """
    Parser for Go source code.

    Provides methods to extract structure and content from Go files
    using regular expressions and Go's built-in tools when available.
    """

    def __init__(self):
        """Initialize the Go parser."""
        self._parse_cache = {}  # Cache parsed file structures
        self._go_installed = self._check_go_installed()

    def _check_go_installed(self) -> bool:
        """
        Check if Go is installed and available.

        Returns:
            True if Go is installed, False otherwise
        """
        try:
            subprocess.run(
                ["go", "version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning(
                "Go is not installed or not in PATH. Using regex-based parsing only."
            )
            return False

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Go file and extract its structure.

        Args:
            file_path: Path to the Go file to parse

        Returns:
            Dictionary with parsed file structure
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Check cache first
            if file_path in self._parse_cache:
                return self._parse_cache[file_path]

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to use Go's built-in tooling if available
            if self._go_installed:
                try:
                    ast_result = self._parse_with_go_ast(file_path)
                    if ast_result and not ast_result.get("error"):
                        self._parse_cache[file_path] = ast_result
                        return ast_result
                except Exception as e:
                    logger.debug(
                        f"Error using Go AST parser: {str(e)}. Falling back to regex parser."
                    )

            # Fall back to regex-based parsing
            result = self._parse_with_regex(content, file_path)
            self._parse_cache[file_path] = result
            return result

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return {"error": str(e)}

    def parse_content(self, content: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Parse Go code content directly.

        Args:
            content: Go code as string
            filename: Optional virtual filename for the content

        Returns:
            Dictionary with parsed content structure
        """
        try:
            # Use regex-based parsing for content
            return self._parse_with_regex(content, filename)

        except Exception as e:
            logger.error(f"Error parsing content: {str(e)}")
            return {"error": str(e)}

    def _parse_with_go_ast(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Go file using Go's built-in AST tools.

        Args:
            file_path: Path to the Go file

        Returns:
            Dictionary with parsed file structure
        """
        # Create a temporary Go file to extract the AST
        temp_dir = os.path.dirname(os.path.abspath(file_path))
        ast_tool_path = os.path.join(temp_dir, "_temp_ast_tool.go")

        # Simple Go program to print the AST as JSON
        ast_tool_content = """
package main

import (
    "encoding/json"
    "fmt"
    "go/ast"
    "go/parser"
    "go/token"
    "os"
)

type SimpleFunc struct {
    Name       string   `json:"name"`
    DocString  string   `json:"docstring,omitempty"`
    Parameters []string `json:"parameters,omitempty"`
    Results    []string `json:"results,omitempty"`
    IsMethod   bool     `json:"is_method"`
    Receiver   string   `json:"receiver,omitempty"`
}

type SimpleStruct struct {
    Name      string   `json:"name"`
    DocString string   `json:"docstring,omitempty"`
    Fields    []string `json:"fields,omitempty"`
}

type SimpleInterface struct {
    Name      string      `json:"name"`
    DocString string      `json:"docstring,omitempty"`
    Methods   []SimpleFunc `json:"methods,omitempty"`
}

type SimpleFile struct {
    Package    string           `json:"package"`
    Imports    []string         `json:"imports,omitempty"`
    Functions  []SimpleFunc     `json:"functions,omitempty"`
    Structs    []SimpleStruct   `json:"structs,omitempty"`
    Interfaces []SimpleInterface `json:"interfaces,omitempty"`
}

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: go run ast_tool.go <file.go>")
        os.Exit(1)
    }

    filePath := os.Args[1]
    fset := token.NewFileSet()
    f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
    if err != nil {
        fmt.Printf(`{"error": "%s"}`, err)
        os.Exit(1)
    }

    result := SimpleFile{
        Package: f.Name.Name,
    }

    // Extract imports
    for _, i := range f.Imports {
        path := i.Path.Value
        result.Imports = append(result.Imports, path)
    }

    // Extract declarations
    for _, decl := range f.Decls {
        switch d := decl.(type) {
        case *ast.FuncDecl:
            // Get function/method info
            fn := SimpleFunc{
                Name:      d.Name.Name,
                IsMethod:  d.Recv != nil,
            }

            // Extract doc string
            if d.Doc != nil {
                fn.DocString = d.Doc.Text()
            }

            // Extract receiver for methods
            if d.Recv != nil && len(d.Recv.List) > 0 {
                // Get receiver type
                expr := d.Recv.List[0].Type
                if star, ok := expr.(*ast.StarExpr); ok {
                    expr = star.X  // Dereference pointer
                }
                if ident, ok := expr.(*ast.Ident); ok {
                    fn.Receiver = ident.Name
                }
            }

            // Extract parameters
            if d.Type.Params != nil {
                for _, param := range d.Type.Params.List {
                    typeStr := ""
                    // Convert type to string (simplified)
                    switch t := param.Type.(type) {
                    case *ast.Ident:
                        typeStr = t.Name
                    case *ast.StarExpr:
                        if ident, ok := t.X.(*ast.Ident); ok {
                            typeStr = "*" + ident.Name
                        }
                    }
                    
                    // Add parameters
                    for _, name := range param.Names {
                        paramStr := name.Name + " " + typeStr
                        fn.Parameters = append(fn.Parameters, paramStr)
                    }
                }
            }

            // Extract return values
            if d.Type.Results != nil {
                for _, result := range d.Type.Results.List {
                    typeStr := ""
                    // Convert type to string (simplified)
                    switch t := result.Type.(type) {
                    case *ast.Ident:
                        typeStr = t.Name
                    case *ast.StarExpr:
                        if ident, ok := t.X.(*ast.Ident); ok {
                            typeStr = "*" + ident.Name
                        }
                    }
                    
                    // Add named return or just type
                    if len(result.Names) > 0 {
                        for _, name := range result.Names {
                            resultStr := name.Name + " " + typeStr
                            fn.Results = append(fn.Results, resultStr)
                        }
                    } else {
                        fn.Results = append(fn.Results, typeStr)
                    }
                }
            }

            result.Functions = append(result.Functions, fn)

        case *ast.GenDecl:
            for _, spec := range d.Specs {
                switch s := spec.(type) {
                case *ast.TypeSpec:
                    // Check if it's a struct
                    if structType, ok := s.Type.(*ast.StructType); ok {
                        st := SimpleStruct{
                            Name: s.Name.Name,
                        }
                        
                        // Extract doc string
                        if d.Doc != nil {
                            st.DocString = d.Doc.Text()
                        }
                        
                        // Extract fields
                        if structType.Fields != nil {
                            for _, field := range structType.Fields.List {
                                typeStr := ""
                                // Convert type to string (simplified)
                                switch t := field.Type.(type) {
                                case *ast.Ident:
                                    typeStr = t.Name
                                case *ast.StarExpr:
                                    if ident, ok := t.X.(*ast.Ident); ok {
                                        typeStr = "*" + ident.Name
                                    }
                                }
                                
                                // Add fields
                                if len(field.Names) > 0 {
                                    for _, name := range field.Names {
                                        fieldStr := name.Name + " " + typeStr
                                        st.Fields = append(st.Fields, fieldStr)
                                    }
                                } else {
                                    // Embedded field
                                    st.Fields = append(st.Fields, typeStr)
                                }
                            }
                        }
                        
                        result.Structs = append(result.Structs, st)
                    }
                    
                    // Check if it's an interface
                    if interfaceType, ok := s.Type.(*ast.InterfaceType); ok {
                        iface := SimpleInterface{
                            Name: s.Name.Name,
                        }
                        
                        // Extract doc string
                        if d.Doc != nil {
                            iface.DocString = d.Doc.Text()
                        }
                        
                        // Extract methods
                        if interfaceType.Methods != nil {
                            for _, method := range interfaceType.Methods.List {
                                if len(method.Names) > 0 {
                                    // It's a method
                                    fnType, ok := method.Type.(*ast.FuncType)
                                    if ok {
                                        fn := SimpleFunc{
                                            Name:     method.Names[0].Name,
                                            IsMethod: true,
                                        }
                                        
                                        // Extract parameters
                                        if fnType.Params != nil {
                                            for _, param := range fnType.Params.List {
                                                typeStr := ""
                                                // Get parameter type (simplified)
                                                switch t := param.Type.(type) {
                                                case *ast.Ident:
                                                    typeStr = t.Name
                                                case *ast.StarExpr:
                                                    if ident, ok := t.X.(*ast.Ident); ok {
                                                        typeStr = "*" + ident.Name
                                                    }
                                                }
                                                
                                                // Add parameters
                                                if len(param.Names) > 0 {
                                                    for _, name := range param.Names {
                                                        paramStr := name.Name + " " + typeStr
                                                        fn.Parameters = append(fn.Parameters, paramStr)
                                                    }
                                                } else {
                                                    fn.Parameters = append(fn.Parameters, typeStr)
                                                }
                                            }
                                        }
                                        
                                        // Extract results
                                        if fnType.Results != nil {
                                            for _, result := range fnType.Results.List {
                                                typeStr := ""
                                                // Get result type (simplified)
                                                switch t := result.Type.(type) {
                                                case *ast.Ident:
                                                    typeStr = t.Name
                                                case *ast.StarExpr:
                                                    if ident, ok := t.X.(*ast.Ident); ok {
                                                        typeStr = "*" + ident.Name
                                                    }
                                                }
                                                
                                                // Add results
                                                if len(result.Names) > 0 {
                                                    for _, name := range result.Names {
                                                        resultStr := name.Name + " " + typeStr
                                                        fn.Results = append(fn.Results, resultStr)
                                                    }
                                                } else {
                                                    fn.Results = append(fn.Results, typeStr)
                                                }
                                            }
                                        }
                                        
                                        iface.Methods = append(iface.Methods, fn)
                                    }
                                }
                            }
                        }
                        
                        result.Interfaces = append(result.Interfaces, iface)
                    }
                }
            }
        }
    }

    // Output as JSON
    jsonData, err := json.MarshalIndent(result, "", "  ")
    if err != nil {
        fmt.Printf(`{"error": "%s"}`, err)
        os.Exit(1)
    }
    
    fmt.Println(string(jsonData))
}
"""

        try:
            # Write the temporary Go file
            with open(ast_tool_path, "w") as f:
                f.write(ast_tool_content)

            # Run the Go program to extract AST
            process = subprocess.run(
                ["go", "run", ast_tool_path, file_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the JSON output
            import json

            ast_data = json.loads(process.stdout)

            if os.path.exists(ast_tool_path):
                os.remove(ast_tool_path)

            # Convert to our standard format
            result = {
                "file": file_path,
                "package": ast_data.get("package", ""),
                "imports": [],
                "functions": [],
                "structs": [],
                "interfaces": [],
            }

            # Process imports
            for imp in ast_data.get("imports", []):
                # Clean up quotes
                cleaned = imp.strip("\"'")
                result["imports"].append({"path": cleaned})

            # Process functions
            for func in ast_data.get("functions", []):
                func_info = {
                    "name": func.get("name", ""),
                    "docstring": func.get("docstring", ""),
                    "parameters": [],
                    "returns": [],
                    "is_method": func.get("is_method", False),
                    "receiver": func.get("receiver", ""),
                }

                # Process parameters
                for param in func.get("parameters", []):
                    parts = param.split(" ", 1)
                    if len(parts) == 2:
                        func_info["parameters"].append(
                            {
                                "name": parts[0],
                                "type": parts[1],
                            }
                        )

                # Process returns
                for ret in func.get("results", []):
                    parts = ret.split(" ", 1)
                    if len(parts) == 2:
                        func_info["returns"].append(
                            {
                                "name": parts[0],
                                "type": parts[1],
                            }
                        )
                    else:
                        func_info["returns"].append(
                            {
                                "type": ret,
                            }
                        )

                result["functions"].append(func_info)

            # Process structs
            for struct in ast_data.get("structs", []):
                struct_info = {
                    "name": struct.get("name", ""),
                    "docstring": struct.get("docstring", ""),
                    "fields": [],
                }

                # Process fields
                for field in struct.get("fields", []):
                    parts = field.split(" ", 1)
                    if len(parts) == 2:
                        struct_info["fields"].append(
                            {
                                "name": parts[0],
                                "type": parts[1],
                            }
                        )
                    else:
                        # Embedded field
                        struct_info["fields"].append(
                            {
                                "type": field,
                                "embedded": True,
                            }
                        )

                result["structs"].append(struct_info)

            # Process interfaces
            for iface in ast_data.get("interfaces", []):
                iface_info = {
                    "name": iface.get("name", ""),
                    "docstring": iface.get("docstring", ""),
                    "methods": [],
                }

                # Process methods
                for method in iface.get("methods", []):
                    method_info = {
                        "name": method.get("name", ""),
                        "parameters": [],
                        "returns": [],
                    }

                    # Process parameters
                    for param in method.get("parameters", []):
                        parts = param.split(" ", 1)
                        if len(parts) == 2:
                            method_info["parameters"].append(
                                {
                                    "name": parts[0],
                                    "type": parts[1],
                                }
                            )
                        else:
                            method_info["parameters"].append(
                                {
                                    "type": param,
                                }
                            )

                    # Process returns
                    for ret in method.get("results", []):
                        parts = ret.split(" ", 1)
                        if len(parts) == 2:
                            method_info["returns"].append(
                                {
                                    "name": parts[0],
                                    "type": parts[1],
                                }
                            )
                        else:
                            method_info["returns"].append(
                                {
                                    "type": ret,
                                }
                            )

                    iface_info["methods"].append(method_info)

                result["interfaces"].append(iface_info)

            return result

        except subprocess.CalledProcessError as e:
            logger.debug(f"Go AST extraction failed: {e.stderr}")
            raise
        finally:

            if os.path.exists(ast_tool_path):
                os.remove(ast_tool_path)

    def _parse_with_regex(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Parse Go code using regular expressions.

        Args:
            content: Go code content
            file_path: Path to the file (for reference)

        Returns:
            Dictionary with parsed file structure
        """
        result = {
            "file": file_path,
            "package": "",
            "imports": [],
            "functions": [],
            "structs": [],
            "interfaces": [],
        }

        # Extract package name
        package_match = re.search(r"package\s+(\w+)", content)
        if package_match:
            result["package"] = package_match.group(1)

        # Extract imports
        import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
        for block in import_blocks:
            for line in block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    # Extract quoted import path
                    import_match = re.search(r'"([^"]+)"', line)
                    if import_match:
                        import_path = import_match.group(1)
                        result["imports"].append({"path": import_path})

        # Also catch single line imports
        single_imports = re.findall(r'import\s+"([^"]+)"', content)
        for import_path in single_imports:
            result["imports"].append({"path": import_path})

        # Extract function definitions (not methods)
        function_matches = re.findall(
            r"func\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
        )
        for name, args, ret in function_matches:
            # Parse arguments
            params = self._parse_go_params(args)

            # Parse return values
            returns = self._parse_go_returns(ret)

            # Add to results
            result["functions"].append(
                {
                    "name": name,
                    "parameters": params,
                    "returns": returns,
                    "is_method": False,
                }
            )

        # Extract method definitions
        method_matches = re.findall(
            r"func\s+\(([\w\s*]+)\)\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
        )
        for receiver, name, args, ret in method_matches:
            # Parse receiver
            receiver_parts = receiver.strip().split()
            receiver_var = receiver_parts[0] if len(receiver_parts) > 0 else ""
            receiver_type = (
                receiver_parts[-1].replace("*", "")
                if len(receiver_parts) > 1
                else receiver_parts[0].replace("*", "")
            )

            # Parse arguments
            params = self._parse_go_params(args)

            # Parse return values
            returns = self._parse_go_returns(ret)

            # Add to results
            result["functions"].append(
                {
                    "name": name,
                    "parameters": params,
                    "returns": returns,
                    "is_method": True,
                    "receiver": receiver_type,
                    "receiver_var": receiver_var,
                }
            )

        # Extract struct definitions
        struct_matches = re.findall(
            r"type\s+(\w+)\s+struct\s+{(.*?)}", content, re.DOTALL
        )
        for name, fields_str in struct_matches:
            fields = []

            # Match field names and types
            field_matches = re.findall(r"(\w+)(?:\s+([*\[\]\w\.]+))", fields_str)
            for field_name, field_type in field_matches:
                fields.append(
                    {
                        "name": field_name,
                        "type": field_type.strip(),
                    }
                )

            result["structs"].append(
                {
                    "name": name,
                    "fields": fields,
                }
            )

        # Extract interface definitions
        interface_matches = re.findall(
            r"type\s+(\w+)\s+interface\s+{(.*?)}", content, re.DOTALL
        )
        for name, methods_str in interface_matches:
            methods = []

            # Extract method signatures
            method_matches = re.findall(
                r"(\w+)(?:\((.*?)\))(?:\s*(.*?))?(?:$|\n)", methods_str, re.MULTILINE
            )

            for method_match in method_matches:
                method_name = method_match[0]

                # Parse parameters
                method_params = []
                if len(method_match) > 1:
                    method_params = self._parse_go_params(method_match[1])

                # Parse return values
                method_returns = []
                if len(method_match) > 2:
                    method_returns = self._parse_go_returns(method_match[2])

                methods.append(
                    {
                        "name": method_name,
                        "parameters": method_params,
                        "returns": method_returns,
                    }
                )

            result["interfaces"].append(
                {
                    "name": name,
                    "methods": methods,
                }
            )

        return result

    def _parse_go_params(self, params_str: str) -> List[Dict[str, str]]:
        """
        Parse Go function/method parameters.

        Args:
            params_str: Parameter string (e.g., "a int, b string")

        Returns:
            List of parameter dictionaries
        """
        params = []

        if not params_str.strip():
            return params

        # Split by commas, but handle multiple parameters of the same type
        param_groups = re.findall(r"((?:\w+(?:,\s*\w+)*)\s+[\*\[\]\w\.]+)", params_str)

        for group in param_groups:
            parts = group.split()
            if len(parts) >= 2:
                param_type = parts[-1]
                param_names = parts[0].split(",")

                for name in param_names:
                    name = name.strip()
                    if name:
                        params.append({"name": name, "type": param_type})

        return params

    def _parse_go_returns(self, returns_str: str) -> List[Dict[str, str]]:
        """
        Parse Go function/method return values.

        Args:
            returns_str: Return string (e.g., "int, error")

        Returns:
            List of return value dictionaries
        """
        returns_str = returns_str.strip()

        if not returns_str:
            return []

        returns = []

        # Handle single unnamed return
        if "(" not in returns_str and "," not in returns_str:
            return [{"type": returns_str.strip()}]

        # Handle named returns or multiple returns
        if returns_str.startswith("("):
            # Remove parentheses
            returns_str = (
                returns_str[1:-1] if returns_str.endswith(")") else returns_str[1:]
            )

        # Split by commas
        for ret in returns_str.split(","):
            ret = ret.strip()
            if ret:
                # Check for named return
                parts = ret.split()
                if len(parts) == 2:
                    returns.append({"name": parts[0], "type": parts[1]})
                else:
                    returns.append({"type": ret})

        return returns

    def find_dependencies(self, content: str) -> Set[str]:
        """
        Find external package dependencies in Go code.

        Args:
            content: Go code content

        Returns:
            Set of dependency package names
        """
        dependencies = set()

        # Extract import paths
        import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
        for block in import_blocks:
            for line in block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    # Extract quoted import path
                    import_match = re.search(r'"([^"]+)"', line)
                    if import_match:
                        import_path = import_match.group(1)
                        # Extract package name (last component of path)
                        parts = import_path.split("/")
                        if parts:
                            dependencies.add(parts[-1])

        # Also catch single line imports
        single_imports = re.findall(r'import\s+"([^"]+)"', content)
        for import_path in single_imports:
            parts = import_path.split("/")
            if parts:
                dependencies.add(parts[-1])

        # Remove standard library packages
        std_libs = {
            "fmt",
            "os",
            "io",
            "strings",
            "strconv",
            "time",
            "math",
            "bytes",
            "path",
            "net",
            "http",
            "encoding",
            "json",
            "regexp",
            "context",
            "sync",
            "errors",
            "log",
            "flag",
            "testing",
            "sort",
            "bufio",
            "io/ioutil",
            "path/filepath",
        }

        return dependencies - std_libs

    def extract_test_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract test functions from a Go test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of test function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        test_functions = []

        # Extract test functions (Test*)
        for func in analysis.get("functions", []):
            if func["name"].startswith("Test"):
                test_functions.append(func)

        return test_functions

    def extract_benchmark_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract benchmark functions from a Go test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of benchmark function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        benchmark_functions = []

        # Extract benchmark functions (Benchmark*)
        for func in analysis.get("functions", []):
            if func["name"].startswith("Benchmark"):
                benchmark_functions.append(func)

        return benchmark_functions

    def extract_example_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract example functions from a Go test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of example function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        example_functions = []

        # Extract example functions (Example*)
        for func in analysis.get("functions", []):
            if func["name"].startswith("Example"):
                example_functions.append(func)

        return example_functions

    def detect_test_pattern(self, file_path: str) -> str:
        """
        Detect if a Go test file uses table-driven tests.

        Args:
            file_path: Path to the test file

        Returns:
            Pattern name: "table_driven" or "standard"
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for table-driven test patterns
            table_patterns = [
                r"tests\s*:=\s*\[\]struct",  # tests := []struct{...}
                r"testCases\s*:=\s*\[\]struct",  # testCases := []struct{...}
                r"cases\s*:=\s*\[\]struct",  # cases := []struct{...}
                r"for\s*,\s*tc\s*:=\s*range\s+",  # for _, tc := range ...
                r"for\s*,\s*tt\s*:=\s*range\s+",  # for _, tt := range ...
                r"for\s*,\s*test\s*:=\s*range\s+",  # for _, test := range ...
            ]

            for pattern in table_patterns:
                if re.search(pattern, content):
                    return "table_driven"

            return "standard"

        except Exception as e:
            logger.error(f"Error detecting test pattern in {file_path}: {str(e)}")
            return "standard"


# Create a singleton instance
go_parser = GoParser()

```

# test_agent/language/python/__init__.py

```py

```

# test_agent/language/python/adapter.py

```py
# test_agent/language/python/adapter.py

import os
import ast
import tempfile
import logging

from typing import List, Dict, Any, Optional


from ..base import LanguageAdapter, registry

# Configure logging
logger = logging.getLogger(__name__)


class PythonAdapter(LanguageAdapter):
    """Python language adapter with advanced AST-based analysis"""

    def __init__(self):
        """Initialize the Python adapter"""
        self._project_roots_cache = {}
        self._test_patterns_cache = {}

    @property
    def language_name(self) -> str:
        """Return the name of the language"""
        return "python"

    @property
    def file_extensions(self) -> List[str]:
        """Return the file extensions for this language"""
        return [".py"]

    @property
    def test_file_prefix(self) -> str:
        """Return the prefix for test files"""
        return "test_"

    @property
    def test_command(self) -> List[str]:
        """Return the command to run tests"""
        return ["python", "-m", "pytest"]

    def analyze_source_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python source file using AST"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file into an AST
            tree = ast.parse(content)

            # Extract information
            classes = []
            functions = []
            imports = []
            module_docstring = ast.get_docstring(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    class_docstring = ast.get_docstring(node)

                    for m in node.body:
                        if isinstance(m, ast.FunctionDef):
                            method_docstring = ast.get_docstring(m)
                            method_args = self._extract_function_args(m)
                            method_returns = self._extract_function_returns(m)

                            methods.append(
                                {
                                    "name": m.name,
                                    "args": method_args,
                                    "returns": method_returns,
                                    "docstring": method_docstring,
                                    "lineno": m.lineno,
                                    "decorators": [
                                        d.id
                                        for d in m.decorator_list
                                        if isinstance(d, ast.Name)
                                    ],
                                }
                            )

                    # Skip internal/private classes
                    if not node.name.startswith("_"):
                        classes.append(
                            {
                                "name": node.name,
                                "methods": methods,
                                "docstring": class_docstring,
                                "lineno": node.lineno,
                                "decorators": [
                                    d.id
                                    for d in node.decorator_list
                                    if isinstance(d, ast.Name)
                                ],
                            }
                        )

                elif isinstance(node, ast.FunctionDef):
                    # Only include top-level functions
                    if isinstance(node.parent, ast.Module):
                        # Skip internal/private functions for testing
                        if not node.name.startswith("_"):
                            docstring = ast.get_docstring(node)
                            args = self._extract_function_args(node)
                            returns = self._extract_function_returns(node)

                            functions.append(
                                {
                                    "name": node.name,
                                    "args": args,
                                    "returns": returns,
                                    "docstring": docstring,
                                    "lineno": node.lineno,
                                    "decorators": [
                                        d.id
                                        for d in node.decorator_list
                                        if isinstance(d, ast.Name)
                                    ],
                                }
                            )

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(
                                {
                                    "module": name.name,
                                    "name": name.name.split(".")[-1],
                                    "alias": name.asname,
                                    "lineno": node.lineno,
                                }
                            )
                    else:  # ImportFrom
                        module = node.module or ""
                        for name in node.names:
                            imports.append(
                                {
                                    "module": module,
                                    "name": name.name,
                                    "alias": name.asname,
                                    "lineno": node.lineno,
                                }
                            )

            # Extract type annotations
            type_annotations = self._extract_type_annotations(tree)

            return {
                "file": file_path,
                "module_docstring": module_docstring,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "type_annotations": type_annotations,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _extract_function_args(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function arguments with type annotations"""
        args = []

        for arg in node.args.args:
            arg_info = {"name": arg.arg, "type": None}

            # Extract type annotation if present
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_info["type"] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_info["type"] = self._extract_attribute(arg.annotation)
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_info["type"] = self._extract_subscript(arg.annotation)

            args.append(arg_info)

        # Handle keyword arguments
        if node.args.kwarg:
            args.append(
                {"name": node.args.kwarg.arg, "type": "dict", "is_kwargs": True}
            )

        # Handle *args
        if node.args.vararg:
            args.append(
                {"name": node.args.vararg.arg, "type": "tuple", "is_vararg": True}
            )

        return args

    def _extract_function_returns(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract function return type annotation"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return self._extract_attribute(node.returns)
            elif isinstance(node.returns, ast.Subscript):
                return self._extract_subscript(node.returns)

        # Try to find return annotation from return statements
        return_values = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Name):
                    return_values.append(child.value.id)
                elif isinstance(child.value, ast.Constant):
                    return_values.append(type(child.value.value).__name__)

        if return_values:
            return ", ".join(set(return_values))

        return None

    def _extract_attribute(self, node: ast.Attribute) -> str:
        """Extract attribute name from AST node"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._extract_attribute(node.value)}.{node.attr}"
        return node.attr

    def _extract_subscript(self, node: ast.Subscript) -> str:
        """Extract subscript expression from AST node"""
        value = ""
        if isinstance(node.value, ast.Name):
            value = node.value.id
        elif isinstance(node.value, ast.Attribute):
            value = self._extract_attribute(node.value)

        # Handle different Python versions
        if hasattr(node, "slice"):
            if isinstance(node.slice, ast.Index):
                if hasattr(node.slice, "value"):
                    slice_str = self._extract_slice_value(node.slice.value)
                    return f"{value}[{slice_str}]"
            elif isinstance(node.slice, ast.Name):
                return f"{value}[{node.slice.id}]"
            elif isinstance(node.slice, ast.Constant):
                return f"{value}[{node.slice.value}]"
            elif isinstance(node.slice, ast.Tuple):
                elts = []
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Name):
                        elts.append(elt.id)
                    elif isinstance(elt, ast.Constant):
                        elts.append(str(elt.value))
                return f"{value}[{', '.join(elts)}]"

        return f"{value}[...]"

    def _extract_slice_value(self, node: ast.AST) -> str:
        """Extract string representation of a slice value"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Tuple):
            elts = []
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    elts.append(elt.id)
                elif isinstance(elt, ast.Attribute):
                    elts.append(self._extract_attribute(elt))
            return ", ".join(elts)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "..."

    def _extract_type_annotations(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract variable type annotations from AST"""
        annotations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and node.annotation:
                if isinstance(node.target, ast.Name):
                    anno_type = None
                    if isinstance(node.annotation, ast.Name):
                        anno_type = node.annotation.id
                    elif isinstance(node.annotation, ast.Attribute):
                        anno_type = self._extract_attribute(node.annotation)
                    elif isinstance(node.annotation, ast.Subscript):
                        anno_type = self._extract_subscript(node.annotation)

                    if anno_type:
                        annotations.append(
                            {
                                "name": node.target.id,
                                "type": anno_type,
                                "lineno": node.lineno,
                            }
                        )

        return annotations

    def analyze_test_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python test file to extract test cases and framework"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file into an AST
            tree = ast.parse(content)

            test_functions = []
            test_classes = []
            imports = []
            framework = self._detect_test_framework_from_content(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        docstring = ast.get_docstring(node)
                        test_functions.append(
                            {
                                "name": node.name,
                                "docstring": docstring,
                                "lineno": node.lineno,
                            }
                        )

                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith("Test"):
                        class_docstring = ast.get_docstring(node)
                        methods = []

                        for m in node.body:
                            if isinstance(m, ast.FunctionDef) and (
                                m.name.startswith("test_")
                                or (
                                    framework == "unittest"
                                    and m.name.startswith("test")
                                )
                            ):
                                method_docstring = ast.get_docstring(m)
                                methods.append(
                                    {
                                        "name": m.name,
                                        "docstring": method_docstring,
                                        "lineno": m.lineno,
                                    }
                                )

                        test_classes.append(
                            {
                                "name": node.name,
                                "methods": methods,
                                "docstring": class_docstring,
                                "lineno": node.lineno,
                            }
                        )

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    else:  # ImportFrom
                        module = node.module or ""
                        for name in node.names:
                            imports.append(f"{module}.{name.name}")

            # Find source file being tested
            project_root = self._find_project_root(file_path)
            source_file = self.find_corresponding_source(file_path, project_root)

            return {
                "file": file_path,
                "source_file": source_file,
                "test_functions": test_functions,
                "test_classes": test_classes,
                "imports": imports,
                "framework": framework,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _detect_test_framework_from_content(self, content: str) -> str:
        """Detect which test framework is being used in the content"""
        pytest_indicators = [
            "import pytest",
            "from pytest",
            "@pytest",
            "pytest.fixture",
            "pytest.mark",
            "def test_",
        ]

        unittest_indicators = [
            "import unittest",
            "from unittest",
            "TestCase",
            "self.assert",
            "setUp",
            "tearDown",
        ]

        pytest_score = sum(1 for indicator in pytest_indicators if indicator in content)
        unittest_score = sum(
            1 for indicator in unittest_indicators if indicator in content
        )

        return "pytest" if pytest_score >= unittest_score else "unittest"

    def detect_project_structure(self, project_dir: str) -> Dict[str, Any]:
        """Detect project structure, test patterns, and framework information"""
        # Check cache first
        project_dir = os.path.abspath(project_dir)
        if project_dir in self._test_patterns_cache:
            return self._test_patterns_cache[project_dir]

        # Analyze project structure
        test_directories = []
        test_files = []

        # Look for test directories
        for root, dirs, _ in os.walk(project_dir):
            for dir_name in dirs:
                if dir_name.lower() in ["test", "tests", "testing"]:
                    test_directories.append(os.path.join(root, dir_name))

        # Look for test files
        for root, _, files in os.walk(project_dir):
            for file_name in files:
                if self.check_is_test_file(file_name):
                    test_files.append(os.path.join(root, file_name))

        # Analyze test location patterns
        location_pattern = "tests_directory"  # Default
        naming_convention = "test_prefix"  # Default

        if test_files:
            # Count location patterns
            location_counts = {
                "same_directory": 0,
                "tests_directory": 0,
                "tests_subdirectory": 0,
                "mirror_under_tests": 0,
            }

            # Count naming patterns
            naming_counts = {"test_prefix": 0, "suffix_test": 0}

            for test_file in test_files:
                # Check location pattern
                rel_path = os.path.relpath(test_file, project_dir)
                parts = rel_path.split(os.sep)

                if parts[0].lower() in ["tests", "test"]:
                    if len(parts) > 2:
                        location_counts["mirror_under_tests"] += 1
                    else:
                        location_counts["tests_directory"] += 1
                else:
                    dir_name = os.path.basename(os.path.dirname(test_file)).lower()
                    if dir_name in ["tests", "test"]:
                        location_counts["tests_subdirectory"] += 1
                    else:
                        location_counts["same_directory"] += 1

                # Check naming pattern
                file_name = os.path.basename(test_file)
                if file_name.startswith("test_"):
                    naming_counts["test_prefix"] += 1
                elif "_test" in file_name:
                    naming_counts["suffix_test"] += 1

            # Determine primary patterns
            if location_counts:
                location_pattern = max(location_counts.items(), key=lambda x: x[1])[0]

            if naming_counts:
                naming_convention = max(naming_counts.items(), key=lambda x: x[1])[0]

        # Detect test framework
        framework = "pytest"  # Default
        for test_file in test_files[:5]:  # Check first 5 test files

            with open(test_file, "r") as f:
                content = f.read()
                detected_framework = self._detect_test_framework_from_content(content)
                if detected_framework == "unittest":
                    framework = "unittest"
                    break

        # Determine primary test directory
        primary_test_dir = None
        if test_directories:
            # Preference for test directories at project root
            for test_dir in test_directories:
                if os.path.dirname(test_dir) == project_dir:
                    primary_test_dir = test_dir
                    break

            # If no test directory at project root, use the first one
            if not primary_test_dir:
                primary_test_dir = test_directories[0]
        else:
            # Default to tests/ at project root
            primary_test_dir = os.path.join(project_dir, "tests")

        # Create result
        result = {
            "test_directories": test_directories,
            "test_files": test_files,
            "location_pattern": location_pattern,
            "naming_convention": naming_convention,
            "framework": framework,
            "primary_test_dir": primary_test_dir,
        }

        # Cache result
        self._test_patterns_cache[project_dir] = result

        return result

    def generate_test_path(
        self,
        source_file: str,
        test_directory: str,
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the proper path for a test file given a source file and pattern"""
        # Ensure absolute paths
        source_file = os.path.abspath(source_file)
        test_directory = os.path.abspath(test_directory)

        # First, check if test already exists
        project_root = self._find_project_root(source_file)
        existing_test = self.find_corresponding_test(source_file, project_root)
        if existing_test and os.path.exists(existing_test):
            return existing_test

        # Use provided pattern or detect it
        if pattern is None:
            pattern = self.detect_project_structure(project_root)

        # Extract filename components
        source_name = os.path.basename(source_file)
        source_base, source_ext = os.path.splitext(source_name)

        # Determine test filename based on naming convention
        naming_convention = pattern.get("naming_convention", "test_prefix")

        if naming_convention == "test_prefix":
            test_filename = f"test_{source_name}"
        else:  # suffix_test
            test_filename = f"{source_base}_test{source_ext}"

        # Generate path based on location pattern
        location_pattern = pattern.get("location_pattern", "tests_directory")

        if location_pattern == "same_directory":
            return os.path.join(os.path.dirname(source_file), test_filename)

        elif location_pattern == "tests_subdirectory":
            test_dir = os.path.join(os.path.dirname(source_file), "tests")
            os.makedirs(test_dir, exist_ok=True)
            return os.path.join(test_dir, test_filename)

        elif location_pattern == "mirror_under_tests":
            # Get path relative to project root
            rel_path = os.path.relpath(source_file, project_root)
            rel_dir = os.path.dirname(rel_path)

            # Create mirror directory under test_directory
            mirror_dir = os.path.join(test_directory, rel_dir)
            os.makedirs(mirror_dir, exist_ok=True)
            return os.path.join(mirror_dir, test_filename)

        else:  # tests_directory or fallback
            # Check for module-specific directory in test_directory
            module_name = os.path.basename(os.path.dirname(source_file))
            module_test_dir = os.path.join(test_directory, module_name)

            if os.path.exists(module_test_dir) and os.path.isdir(module_test_dir):
                return os.path.join(module_test_dir, test_filename)

            # Use test_directory directly
            os.makedirs(test_directory, exist_ok=True)
            return os.path.join(test_directory, test_filename)

    def generate_test_template(
        self,
        source_file: str,
        analysis: Dict[str, Any],
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a test template based on source analysis and project patterns"""
        # Use provided pattern or detect it
        if pattern is None:
            project_dir = self._find_project_root(source_file)
            pattern = self.detect_project_structure(project_dir)

        module_name = os.path.basename(source_file).replace(".py", "")
        framework = pattern.get("framework", "pytest")
        naming_convention = pattern.get("naming_convention", "test_prefix")

        if framework == "unittest":
            return self._get_unittest_template(module_name, analysis, naming_convention)
        else:
            return self._get_pytest_template(module_name, analysis, naming_convention)

    def _get_pytest_template(
        self, module_name: str, analysis: Dict[str, Any], naming_convention: str
    ) -> str:
        """Generate a pytest-style test template"""
        template = f"""# Generated test file for {module_name}
import pytest
from {module_name} import *

"""
        # Add test functions for classes
        for cls in analysis.get("classes", []):
            if naming_convention in ["test_classes", "mixed"]:
                template += f"""
class Test{cls['name']}:
    def test_{cls['name'].lower()}_initialization(self):
        \"\"\"Test that {cls['name']} can be initialized.\"\"\"
        instance = {cls['name']}()
        assert instance is not None
"""
                # Add tests for methods
                for method in cls.get("methods", []):
                    if method["name"] != "__init__":
                        template += f"""
    def test_{method['name']}(self):
        \"\"\"Test the {method['name']} method of {cls['name']}.\"\"\"
        instance = {cls['name']}()
        # TODO: Add proper test for {method['name']}
        # assert instance.{method['name']}() == expected_result
        pass
"""
            else:
                # Add function-style tests for class methods
                template += f"""
def test_{cls['name'].lower()}_initialization():
    \"\"\"Test that {cls['name']} can be initialized.\"\"\"
    instance = {cls['name']}()
    assert instance is not None
"""
                # Add tests for methods
                for method in cls.get("methods", []):
                    if method["name"] != "__init__":
                        template += f"""
def test_{cls['name'].lower()}_{method['name']}():
    \"\"\"Test the {method['name']} method of {cls['name']}.\"\"\"
    instance = {cls['name']}()
    # TODO: Add proper test for {method['name']}
    # assert instance.{method['name']}() == expected_result
    pass
"""

        # Add test functions for standalone functions
        for func in analysis.get("functions", []):
            template += f"""
def test_{func['name']}():
    \"\"\"Test the {func['name']} function.\"\"\"
    # TODO: Add proper test for {func['name']}
    # result = {func['name']}()
    # assert result == expected_value
    pass
"""

        return template

    def _get_unittest_template(
        self, module_name: str, analysis: Dict[str, Any], naming_convention: str
    ) -> str:
        """Generate a unittest-style test template"""
        template = f"""# Generated test file for {module_name}
import unittest
from {module_name} import *

"""
        # Add test classes for classes
        for cls in analysis.get("classes", []):
            template += f"""
class Test{cls['name']}(unittest.TestCase):
    def test_{cls['name'].lower()}_initialization(self):
        \"\"\"Test that {cls['name']} can be initialized.\"\"\"
        instance = {cls['name']}()
        self.assertIsNotNone(instance)
"""
            # Add tests for methods
            for method in cls.get("methods", []):
                if method["name"] != "__init__":
                    template += f"""
    def test_{method['name']}(self):
        \"\"\"Test the {method['name']} method of {cls['name']}.\"\"\"
        instance = {cls['name']}()
        # TODO: Add proper test for {method['name']}
        # result = instance.{method['name']}()
        # self.assertEqual(result, expected_result)
        pass
"""

        # Add test class for standalone functions
        if analysis.get("functions"):
            template += f"""
class Test{module_name.capitalize()}Functions(unittest.TestCase):"""

            # Add test methods for standalone functions
            for func in analysis.get("functions", []):
                template += f"""
    def test_{func['name']}(self):
        \"\"\"Test the {func['name']} function.\"\"\"
        # TODO: Add proper test for {func['name']}
        # result = {func['name']}()
        # self.assertEqual(result, expected_value)
        pass
"""

        template += """

if __name__ == '__main__':
    unittest.main()
"""

        return template

    def check_is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file"""
        file_name = os.path.basename(file_path)

        if os.path.basename(file_path) == "__init__.py":
            return True

        # Common test file patterns for Python
        return (
            file_name.startswith("test_")
            or file_name.endswith("_test.py")
            or "test" in os.path.dirname(file_path).lower()
        )

    def find_corresponding_source(
        self, test_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the source file that a test file is testing"""
        if not self.check_is_test_file(test_file):
            return None

        file_name = os.path.basename(test_file)
        # file_ext = os.path.splitext(file_name)[1]

        # Remove test prefix/suffix based on language conventions
        if file_name.startswith("test_"):
            source_name = file_name[5:]  # Remove "test_"
        elif file_name.endswith("_test.py"):
            source_name = file_name[:-8] + ".py"  # Remove "_test.py" and add ".py"
        else:
            # Can't determine source file name
            return None

        # First look in the same directory
        test_dir = os.path.dirname(test_file)
        potential_source = os.path.join(test_dir, source_name)
        if os.path.exists(potential_source):
            return potential_source

        # Then look in parent directory if test is in a test directory
        test_dir_name = os.path.basename(test_dir).lower()
        if test_dir_name in ["test", "tests", "testing"]:
            parent_dir = os.path.dirname(test_dir)
            potential_source = os.path.join(parent_dir, source_name)
            if os.path.exists(potential_source):
                return potential_source

        # Look for source files in the project with the same base name
        source_base = os.path.splitext(source_name)[0]
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file == source_name:
                    return os.path.join(root, file)
                elif os.path.splitext(file)[0] == source_base and file.endswith(".py"):
                    return os.path.join(root, file)

        return None

    def find_corresponding_test(
        self, source_file: str, project_dir: str
    ) -> Optional[str]:
        """
        Find the test file for a given source file.

        Args:
            source_file: Path to the source file
            project_dir: Root directory of the project

        Returns:
            Optional[str]: Path to the corresponding test file, or None if not found
        """
        # Skip if it's already a test file
        if self.check_is_test_file(source_file):
            return None

        file_name = os.path.basename(source_file)
        file_base, file_ext = os.path.splitext(file_name)
        source_dir = os.path.dirname(source_file)

        # Ensure we're only looking for .py test files
        if file_ext != ".py":
            logger.debug(f"Skipping non-Python file: {source_file}")
            return None

        # Generate possible test file names (ensuring they end with .py)
        test_names = [
            f"test_{file_name}",  # test_file.py
            f"{file_base}_test.py",  # file_test.py
        ]

        # Check in different locations based on common patterns
        potential_locations = [
            source_dir,  # Same directory
            os.path.join(source_dir, "tests"),  # tests/ subdirectory
            os.path.join(source_dir, "test"),  # test/ subdirectory
            os.path.join(project_dir, "tests"),  # tests/ in project root
            os.path.join(project_dir, "test"),  # test/ in project root
        ]

        # Add mirror pattern: directory structure under tests/
        rel_path = os.path.relpath(source_dir, project_dir)
        if rel_path != ".":
            potential_locations.append(os.path.join(project_dir, "tests", rel_path))
            potential_locations.append(os.path.join(project_dir, "test", rel_path))

        # Check each potential location
        for test_name in test_names:
            for location in potential_locations:
                if os.path.exists(location):
                    test_path = os.path.join(location, test_name)
                    if os.path.exists(test_path):
                        logger.debug(f"Found test file at {test_path}")
                        return test_path

        # If no exact match found, try a more comprehensive search but exclude __pycache__
        logger.debug(
            f"No direct match found, performing comprehensive search for {file_base}"
        )

        # Define directories to exclude
        excluded_dirs = [
            "__pycache__",
            ".git",
            ".idea",
            ".vscode",
            "node_modules",
            "venv",
            "env",
        ]

        for root, dirs, files in os.walk(project_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            # Only check directories with "test" in the name
            if "test" in root.lower() or "tests" in root.lower():
                for file in files:
                    # Ensure file is a Python file
                    if not file.endswith(".py"):
                        continue

                    # Check if file name matches test patterns
                    if (file.startswith("test_") and file_base in file) or (
                        file.endswith("_test.py") and file_base in file
                    ):
                        try:
                            test_file_path = os.path.join(root, file)
                            logger.debug(
                                f"Checking potential test file: {test_file_path}"
                            )

                            # Try different encodings when reading files
                            for encoding in [
                                "utf-8",
                                "latin-1",
                                "cp1252",
                                "iso-8859-1",
                            ]:
                                try:
                                    with open(
                                        test_file_path, "r", encoding=encoding
                                    ) as f:
                                        content = f.read()
                                        # Look for imports or references to the source file
                                        if file_base in content or file_name in content:
                                            logger.debug(
                                                f"Found matching test file: {test_file_path}"
                                            )
                                            return test_file_path
                                    # If we get here, the file was read successfully
                                    break
                                except UnicodeDecodeError:
                                    # Try the next encoding
                                    continue
                                except Exception as e:
                                    # Any other error, just log and continue
                                    logger.debug(f"Error reading file {file}: {str(e)}")
                        except Exception as e:
                            logger.debug(f"Error processing file {file}: {str(e)}")

        logger.debug(f"No corresponding test found for {source_file}")
        return None

    def get_environment_location(self) -> str:
        """
        Get the recommended location for the virtual environment
        based on the project structure
        """
        # Use system temp directory for virtual environment
        base_dir = tempfile.gettempdir()
        venv_dir = os.path.join(base_dir, "test_agent_venv")
        return venv_dir


# Add parent references to AST nodes for better analysis
# This modifies the ast module to track parent relationships
def _add_parent_references(tree):
    """Add parent references to all nodes in the AST"""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return tree


# Patch ast.parse to add parent references
original_parse = ast.parse


def patched_parse(*args, **kwargs):
    """Patched version of ast.parse that adds parent references"""
    tree = original_parse(*args, **kwargs)
    return _add_parent_references(tree)


# Apply the patch
ast.parse = patched_parse

# Register the adapter
registry.register([".py"], PythonAdapter)

```

# test_agent/language/python/parser.py

```py
# test_agent/language/python/parser.py

import os
import ast
import logging
from typing import Dict, List, Any, Optional, Set, Union

# Configure logging
logger = logging.getLogger(__name__)


class PythonParser:
    """
    Parser for Python source code.

    Provides methods to extract structure and content from Python files
    using AST (Abstract Syntax Tree) analysis.
    """

    def __init__(self):
        """Initialize the Python parser."""
        self._ast_cache = {}  # Cache parsed AST trees

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Python file and extract its structure.

        Args:
            file_path: Path to the Python file to parse

        Returns:
            Dictionary with parsed file structure
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = self._get_ast(file_path, content)

            # Extract file structure
            return self._analyze_ast(tree, file_path)

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {str(e)}")
            return {"error": f"Syntax error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return {"error": str(e)}

    def parse_content(self, content: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Parse Python code content directly.

        Args:
            content: Python code as string
            filename: Optional virtual filename for the content

        Returns:
            Dictionary with parsed content structure
        """
        try:
            # Parse AST
            tree = ast.parse(content, filename=filename)

            # Add parent references to AST nodes
            tree = self._add_parent_references(tree)

            # Extract structure
            return self._analyze_ast(tree, filename)

        except SyntaxError as e:
            logger.error(f"Syntax error in content: {str(e)}")
            return {"error": f"Syntax error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error parsing content: {str(e)}")
            return {"error": str(e)}

    def _get_ast(self, file_path: str, content: str) -> ast.Module:
        """
        Get AST for file content with caching.

        Args:
            file_path: Path to the file (used as cache key)
            content: File content to parse

        Returns:
            AST Module node
        """
        # Check cache first
        if file_path in self._ast_cache:
            return self._ast_cache[file_path]

        # Parse content
        tree = ast.parse(content, filename=file_path)

        # Add parent references to AST nodes
        tree = self._add_parent_references(tree)

        # Cache the result
        self._ast_cache[file_path] = tree

        return tree

    def _add_parent_references(self, tree: ast.Module) -> ast.Module:
        """
        Add parent references to AST nodes for easier traversal.

        Args:
            tree: AST tree to modify

        Returns:
            Modified AST tree with parent references
        """
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        return tree

    def _analyze_ast(self, tree: ast.Module, file_path: str) -> Dict[str, Any]:
        """
        Analyze AST to extract code structure.

        Args:
            tree: AST tree to analyze
            file_path: Path to the file being analyzed

        Returns:
            Dictionary with code structure
        """
        # Initialize result
        result = {
            "file": file_path,
            "type": "module",
            "imports": [],
            "classes": [],
            "functions": [],
            "variables": [],
            "module_docstring": None,
        }

        # Get module docstring
        result["module_docstring"] = ast.get_docstring(tree)

        # Process top-level nodes
        for node in tree.body:
            # Extract imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = self._extract_imports(node)
                result["imports"].extend(imports)

            # Extract classes
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class(node)
                result["classes"].append(class_info)

            # Extract functions
            elif isinstance(node, ast.FunctionDef):
                func_info = self._extract_function(node)
                result["functions"].append(func_info)

            # Extract variables
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                vars_info = self._extract_variables(node)
                result["variables"].extend(vars_info)

        return result

    def _extract_imports(
        self, node: Union[ast.Import, ast.ImportFrom]
    ) -> List[Dict[str, Any]]:
        """
        Extract import information from import nodes.

        Args:
            node: Import or ImportFrom AST node

        Returns:
            List of import dictionaries
        """
        imports = []

        if isinstance(node, ast.Import):
            # Regular imports (import x, import y)
            for name in node.names:
                imports.append(
                    {
                        "type": "import",
                        "name": name.name,
                        "alias": name.asname,
                        "lineno": node.lineno,
                    }
                )
        else:  # ImportFrom
            # From imports (from x import y)
            module = node.module or ""
            for name in node.names:
                imports.append(
                    {
                        "type": "import_from",
                        "module": module,
                        "name": name.name,
                        "alias": name.asname,
                        "lineno": node.lineno,
                        "level": node.level,  # Relative import level
                    }
                )

        return imports

    def _extract_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Extract class information from a ClassDef node.

        Args:
            node: ClassDef AST node

        Returns:
            Dictionary with class information
        """
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "class_variables": [],
            "bases": [],
            "decorators": [],
            "lineno": node.lineno,
        }

        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info["bases"].append(base.id)
            elif isinstance(base, ast.Attribute):
                class_info["bases"].append(self._extract_attribute(base))

        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                class_info["decorators"].append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Name
            ):
                class_info["decorators"].append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                class_info["decorators"].append(self._extract_attribute(decorator))

        # Extract class body elements
        for item in node.body:
            # Methods
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function(item, is_method=True)
                class_info["methods"].append(method_info)

            # Class variables
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                vars_info = self._extract_variables(item, class_var=True)
                class_info["class_variables"].extend(vars_info)

        return class_info

    def _extract_function(
        self, node: ast.FunctionDef, is_method: bool = False
    ) -> Dict[str, Any]:
        """
        Extract function/method information from a FunctionDef node.

        Args:
            node: FunctionDef AST node
            is_method: Whether this function is a class method

        Returns:
            Dictionary with function information
        """
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": None,
            "decorators": [],
            "lineno": node.lineno,
            "is_method": is_method,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

        # Extract arguments
        for arg in node.args.args:
            arg_info = {"name": arg.arg, "type": None}
            if arg.annotation:
                arg_info["type"] = self._extract_annotation(arg.annotation)
            func_info["args"].append(arg_info)

        # Extract *args argument
        if node.args.vararg:
            vararg = {"name": f"*{node.args.vararg.arg}", "type": None}
            if node.args.vararg.annotation:
                vararg["type"] = self._extract_annotation(node.args.vararg.annotation)
            func_info["args"].append(vararg)

        # Extract **kwargs argument
        if node.args.kwarg:
            kwarg = {"name": f"**{node.args.kwarg.arg}", "type": None}
            if node.args.kwarg.annotation:
                kwarg["type"] = self._extract_annotation(node.args.kwarg.annotation)
            func_info["args"].append(kwarg)

        # Extract return type annotation
        if node.returns:
            func_info["returns"] = self._extract_annotation(node.returns)

        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                func_info["decorators"].append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Name
            ):
                func_info["decorators"].append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                func_info["decorators"].append(self._extract_attribute(decorator))

        return func_info

    def _extract_variables(
        self, node: Union[ast.Assign, ast.AnnAssign], class_var: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract variable assignments.

        Args:
            node: Assign or AnnAssign AST node
            class_var: Whether this is a class variable

        Returns:
            List of variable dictionaries
        """
        variables = []

        if isinstance(node, ast.Assign):
            # Extract targets
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_info = {
                        "name": target.id,
                        "type": None,
                        "lineno": node.lineno,
                        "class_var": class_var,
                    }
                    variables.append(var_info)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            var_info = {
                                "name": elt.id,
                                "type": None,
                                "lineno": node.lineno,
                                "class_var": class_var,
                            }
                            variables.append(var_info)

        elif isinstance(node, ast.AnnAssign):
            # Annotated assignment (x: type = value)
            if isinstance(node.target, ast.Name):
                var_info = {
                    "name": node.target.id,
                    "type": self._extract_annotation(node.annotation),
                    "lineno": node.lineno,
                    "class_var": class_var,
                }
                variables.append(var_info)

        return variables

    def _extract_annotation(self, node: ast.AST) -> Optional[str]:
        """
        Extract type annotation from an AST node.

        Args:
            node: AST node containing type annotation

        Returns:
            String representation of the type annotation
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Subscript):
            value = self._extract_annotation(node.value)

            # Handle subscript (e.g., List[int])
            if hasattr(node, "slice"):
                # Python 3.9+
                if isinstance(node.slice, ast.Index):
                    if hasattr(node.slice, "value"):
                        # Python 3.8-
                        slice_str = self._extract_subscript_slice(node.slice.value)
                        return f"{value}[{slice_str}]"
                    else:
                        # Python 3.9+
                        slice_str = self._extract_subscript_slice(node.slice)
                        return f"{value}[{slice_str}]"
                else:
                    # Direct slice in Python 3.9+
                    slice_str = self._extract_subscript_slice(node.slice)
                    return f"{value}[{slice_str}]"

            return f"{value}[...]"
        elif isinstance(node, ast.Constant):
            # Python 3.8+ uses Constant
            return repr(node.value)
        elif isinstance(node, (ast.Str, ast.Num, ast.Bytes, ast.NameConstant)):
            # Python 3.7 and earlier
            if isinstance(node, ast.Str):
                return repr(node.s)
            elif isinstance(node, ast.Num):
                return repr(node.n)
            elif isinstance(node, ast.Bytes):
                return repr(node.s)
            elif isinstance(node, ast.NameConstant):
                return repr(node.value)

        # Default for unknown annotation types
        return "Any"

    def _extract_attribute(self, node: ast.Attribute) -> str:
        """
        Extract full attribute name (e.g., module.submodule.attr).

        Args:
            node: Attribute AST node

        Returns:
            Full attribute name as string
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._extract_attribute(node.value)}.{node.attr}"
        return node.attr

    def _extract_subscript_slice(self, node: ast.AST) -> str:
        """
        Extract the slice part of a subscript (e.g., the 'int' in List[int]).

        Args:
            node: AST node for the slice part

        Returns:
            String representation of the slice
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Tuple):
            elements = []
            for elt in node.elts:
                elements.append(self._extract_annotation(elt))
            return ", ".join(elements)
        elif isinstance(node, ast.Subscript):
            # Nested subscript (e.g., Dict[str, List[int]])
            value = self._extract_annotation(node.value)
            slice_str = self._extract_subscript_slice(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Constant):
            # Python 3.8+
            return repr(node.value)
        elif isinstance(node, (ast.Str, ast.Num, ast.Bytes, ast.NameConstant)):
            # Python 3.7 and earlier
            if isinstance(node, ast.Str):
                return repr(node.s)
            elif isinstance(node, ast.Num):
                return repr(node.n)
            elif isinstance(node, ast.Bytes):
                return repr(node.s)
            elif isinstance(node, ast.NameConstant):
                return repr(node.value)
        elif isinstance(node, ast.Index):
            # Python 3.8-
            return self._extract_subscript_slice(node.value)

        return "..."

    def find_dependencies(self, tree: ast.Module) -> Set[str]:
        """
        Find external package dependencies in the AST.

        Args:
            tree: AST Module node

        Returns:
            Set of dependency package names
        """
        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Extract root package (e.g., 'pandas' from 'pandas.DataFrame')
                    root_package = name.name.split(".")[0]
                    dependencies.add(root_package)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Extract root package
                    root_package = node.module.split(".")[0]
                    dependencies.add(root_package)

        # Remove standard library modules
        std_libs = {
            "os",
            "sys",
            "re",
            "math",
            "random",
            "datetime",
            "time",
            "json",
            "csv",
            "collections",
            "itertools",
            "functools",
            "typing",
            "pathlib",
            "io",
            "shutil",
            "glob",
            "argparse",
            "logging",
            "unittest",
            "abc",
            "ast",
            "copy",
            "hashlib",
            "pickle",
            "tempfile",
            "warnings",
            "zipfile",
            "importlib",
        }

        return dependencies - std_libs

    def extract_test_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract test functions from a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of test function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        test_functions = []

        # Extract top-level test functions
        for func in analysis.get("functions", []):
            if func["name"].startswith("test_"):
                test_functions.append(func)

        # Extract test methods in test classes
        for cls in analysis.get("classes", []):
            if cls["name"].startswith("Test"):
                for method in cls.get("methods", []):
                    if method["name"].startswith("test_"):
                        # Add class context to the method
                        method["class"] = cls["name"]
                        test_functions.append(method)

        return test_functions

    def extract_test_classes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract test classes from a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of test class dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        test_classes = []

        # Extract test classes
        for cls in analysis.get("classes", []):
            if cls["name"].startswith("Test"):
                # Only include classes with test methods
                has_test_methods = any(
                    method["name"].startswith("test_")
                    for method in cls.get("methods", [])
                )

                if has_test_methods:
                    test_classes.append(cls)

        return test_classes

    def detect_test_framework(self, file_path: str) -> str:
        """
        Detect the test framework used in a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            Detected framework name ("pytest", "unittest", or "unknown")
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return "unknown"

        # Check imports for pytest or unittest
        imports = analysis.get("imports", [])

        for imp in imports:
            if imp.get("name") == "pytest" or imp.get("module") == "pytest":
                return "pytest"
            if imp.get("name") == "unittest" or imp.get("module") == "unittest":
                return "unittest"

        # Check class inheritance for unittest.TestCase
        for cls in analysis.get("classes", []):
            if "TestCase" in cls.get("bases", []):
                return "unittest"

        # Check for pytest fixtures
        for func in analysis.get("functions", []):
            if "fixture" in func.get("decorators", []):
                return "pytest"

        # Check for pytest-style test functions
        has_pytest_tests = any(
            func["name"].startswith("test_") for func in analysis.get("functions", [])
        )

        if has_pytest_tests:
            return "pytest"

        # Check for unittest-style test methods
        for cls in analysis.get("classes", []):
            has_unittest_tests = any(
                method["name"].startswith("test") for method in cls.get("methods", [])
            )

            if has_unittest_tests and "TestCase" in cls.get("bases", []):
                return "unittest"

        return "unknown"


# Create a singleton instance
python_parser = PythonParser()

```

# test_agent/llm/__init__.py

```py
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

```

# test_agent/llm/base.py

```py
# test_agent/llm/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
import logging

# Define a logger for the module
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement,
    allowing the test agent to work with different LLM backends.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Returns the name of the provider.

        Returns:
            str: Provider name (e.g., 'openai', 'claude', 'deepseek', 'gemini')
        """
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Returns a list of available models for this provider.

        Returns:
            List[str]: List of model names
        """
        pass

    @property
    def default_model(self) -> str:
        """
        Returns the default model for this provider.

        Returns:
            str: Default model name
        """
        return self.available_models[0] if self.available_models else ""

    @abstractmethod
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validates that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_llm(self, **kwargs) -> Any:
        """
        Returns a configured LLM instance for use with LangGraph.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: LLM instance
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities.

        Args:
            prompt: The prompt to send to the LLM
            tools: List of tools available to the LLM
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Response containing text and/or tool calls
        """
        pass

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for the specific provider.

        This method allows customizing the tool format for each provider.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for this provider
        """
        # Default implementation (OpenAI-like format)
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize tool input from the LLM to a consistent format.

        Different LLMs may format tool calls differently, this method
        ensures they're normalized to a consistent format for the agent.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in provider-specific format

        Returns:
            Any: Normalized tool input
        """
        # Default implementation (passthrough)
        return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize tool output to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # Default implementation (convert to string)
        if isinstance(output_data, str):
            return output_data
        return str(output_data)

```

# test_agent/llm/claude.py

```py
# test_agent/llm/claude.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """
    Provider for Anthropic's Claude models.
    """

    def __init__(self):
        """Initialize the Claude provider."""
        self._api_key = None
        self._models = {
            "claude-3-5-sonnet-20240620": {
                "context_window": 200000,
                "description": "Claude 3.5 Sonnet - Anthropic's latest mid-range model",
            },
            "claude-3-opus-20240229": {
                "context_window": 200000,
                "description": "Claude 3 Opus - Anthropic's most powerful model",
            },
            "claude-3-sonnet-20240229": {
                "context_window": 200000,
                "description": "Claude 3 Sonnet - Balanced performance and speed",
            },
            "claude-3-haiku-20240307": {
                "context_window": 200000,
                "description": "Claude 3 Haiku - Anthropic's fastest model",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "claude"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available Claude models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default Claude model."""
        return "claude-3-5-sonnet-20240620"  # Using the latest model as default

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # Claude API keys typically start with 'sk-ant-'
        return bool(
            api_key and isinstance(api_key, str) and api_key.startswith("sk-ant-")
        )

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured Claude LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: Claude LLM instance
        """
        try:
            # Try to import Anthropic libraries
            from langchain_anthropic import ChatAnthropic

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY or provide api_key parameter."
                )

            self._api_key = api_key

            # Configure callbacks for streaming if enabled
            callbacks = []
            if streaming:
                try:
                    from langchain.callbacks.streaming_stdout import (
                        StreamingStdOutCallbackHandler,
                    )

                    callbacks.append(StreamingStdOutCallbackHandler())
                except ImportError:
                    logger.warning(
                        "StreamingStdOutCallbackHandler not available, disabling streaming"
                    )

            return ChatAnthropic(
                model=model,
                temperature=temperature,
                anthropic_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )
        except ImportError as e:
            logger.error(f"Error importing Anthropic libraries: {e}")
            raise ImportError(
                "Anthropic integration not available. Please install langchain-anthropic package."
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Generate text using Claude's API directly.

        Args:
            prompt: The prompt to send to Claude
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Claude-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key")
            or self._api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("Anthropic API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        if system_prompt:
            data["system"] = system_prompt

        if max_tokens:
            data["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Claude API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["content"][0]["text"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using Claude.

        Args:
            prompt: The prompt to send to Claude
            tools: List of tools available to Claude
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional Claude-specific parameters

        Returns:
            Dict[str, Any]: Claude response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key")
            or self._api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("Anthropic API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Format tools for Claude's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "tools": formatted_tools,
        }

        if system_prompt:
            data["system"] = system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Claude API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for Claude's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for Claude
        """
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize Claude's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in Claude's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # Handle the simple list format [tool_name, args]
            if isinstance(input_data, list) and len(input_data) >= 2:
                return input_data[1]

            # Handle Claude's advanced tool_use object format
            if isinstance(input_data, dict):
                # Check for special tool_use format
                if input_data.get("type") == "tool_use":
                    # Extract from partial_json if available
                    if "partial_json" in input_data:
                        try:
                            # Parse the partial_json string
                            partial_data = json.loads(input_data["partial_json"])

                            # Handle the __arg1 pattern
                            if "__arg1" in partial_data:
                                # Parse the nested JSON string
                                return json.loads(partial_data["__arg1"])

                            return partial_data
                        except json.JSONDecodeError:
                            # If not valid JSON, return it as is
                            return input_data["partial_json"]

                    # Try input field if partial_json isn't available
                    if "input" in input_data and input_data["input"]:
                        return input_data["input"]

                # Check if we have "responded" field with list of tool_use objects
                if "responded" in input_data and isinstance(
                    input_data["responded"], list
                ):
                    for item in input_data["responded"]:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            # Recursively process this tool_use object
                            return self.normalize_tool_input(tool_name, item)

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing Claude input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize Claude's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        try:
            # If it's already a string, return it
            if isinstance(output_data, str):
                return output_data

            # If it's a list with one element that's a string, return that
            if (
                isinstance(output_data, list)
                and len(output_data) == 1
                and isinstance(output_data[0], str)
            ):
                return output_data[0]

            # If it's a list with objects that have 'text' keys, extract and join them
            if isinstance(output_data, list):
                texts = []
                for item in output_data:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                if texts:
                    return "".join(texts)

            # Last resort - convert to JSON string
            return json.dumps(output_data)

        except Exception as e:
            logger.error(f"Error normalizing Claude output: {str(e)}")
            # Return a string representation as a fallback
            return str(output_data)

```

# test_agent/llm/deepseek.py

```py
# test_agent/llm/deepseek.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class DeepSeekProvider(LLMProvider):
    """
    Provider for DeepSeek models.
    """

    def __init__(self):
        """Initialize the DeepSeek provider."""
        self._api_key = None
        self._models = {
            "deepseek-chat": {
                "context_window": 32768,
                "description": "DeepSeek Chat - General purpose model",
            },
            "deepseek-coder": {
                "context_window": 32768,
                "description": "DeepSeek Coder - Optimized for code generation",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "deepseek"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available DeepSeek models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default DeepSeek model."""
        return "deepseek-coder"  # Using coder model as default for test generation

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # Basic validation - DeepSeek API keys follow a specific format
        return bool(api_key and isinstance(api_key, str) and len(api_key) > 20)

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured DeepSeek LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: DeepSeek LLM instance
        """
        try:
            # Import DeepSeek conditionally to avoid import errors if not installed
            try:
                from langchain_deepseek import ChatDeepSeek
            except ImportError:
                raise ImportError(
                    "DeepSeek integration not available. Please install langchain-deepseek package."
                )

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")

            if not api_key:
                raise ValueError(
                    "DeepSeek API key is required. Set DEEPSEEK_API_KEY or provide api_key parameter."
                )

            self._api_key = api_key

            # Configure callbacks for streaming if enabled
            callbacks = []
            if streaming:
                try:
                    from langchain.callbacks.streaming_stdout import (
                        StreamingStdOutCallbackHandler,
                    )

                    callbacks.append(StreamingStdOutCallbackHandler())
                except ImportError:
                    logger.warning(
                        "StreamingStdOutCallbackHandler not available, disabling streaming"
                    )

            return ChatDeepSeek(
                model=model,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                callbacks=callbacks if streaming else None,
            )
        except Exception as e:
            logger.error(f"Error initializing DeepSeek LLM: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Generate text using DeepSeek's API directly.

        Args:
            prompt: The prompt to send to DeepSeek
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise ValueError("DeepSeek API key is required.")

        # Set environment variable for consistency
        os.environ["DEEPSEEK_API_KEY"] = api_key
        self._api_key = api_key

        model = kwargs.get("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        # DeepSeek API URL may vary - using a placeholder
        api_url = "https://api.deepseek.com/v1/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"DeepSeek API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using DeepSeek.

        Args:
            prompt: The prompt to send to DeepSeek
            tools: List of tools available to DeepSeek
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            Dict[str, Any]: DeepSeek response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise ValueError("DeepSeek API key is required.")

        os.environ["DEEPSEEK_API_KEY"] = api_key
        self._api_key = api_key

        model = kwargs.get("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Format tools for DeepSeek's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": formatted_tools,
        }

        # DeepSeek API URL may vary - using a placeholder
        api_url = "https://api.deepseek.com/v1/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"DeepSeek API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for DeepSeek's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for DeepSeek
        """
        # DeepSeek uses a format similar to OpenAI
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize DeepSeek's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in DeepSeek's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # Handle different input formats
            if isinstance(input_data, dict):
                # If there's a direct arguments field, use it
                if "arguments" in input_data:
                    args = input_data["arguments"]
                    if isinstance(args, str):
                        return json.loads(args)

                    return args

                # If it has action_input, it's probably our standard format
                if "action_input" in input_data:
                    return input_data["action_input"]

                # For direct parameter dictionaries, return as is
                return input_data

            # Handle string input that might be JSON
            if isinstance(input_data, str):
                return json.loads(input_data)

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing DeepSeek input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize DeepSeek's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # For simple strings
        if isinstance(output_data, str):
            return output_data

        # For dictionaries and other objects, convert to JSON
        return json.dumps(output_data)

```

# test_agent/llm/gemini.py

```py
# test_agent/llm/gemini.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Provider for Google's Gemini models.
    """

    def __init__(self):
        """Initialize the Gemini provider."""
        self._api_key = None
        self._models = {
            "gemini-1.0-pro-latest": {
                "context_window": 32768,
                "description": "Gemini 1.0 Pro - Google's general purpose model",
            },
            "gemini-1.5-pro-latest": {
                "context_window": 1000000,
                "description": "Gemini 1.5 Pro - Google's advanced model with large context window",
            },
            "gemini-1.5-flash-latest": {
                "context_window": 1000000,
                "description": "Gemini 1.5 Flash - Faster, more economical model",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "gemini"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available Gemini models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default Gemini model."""
        return "gemini-1.5-pro-latest"  # Using the latest model as default

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # Basic validation for Gemini API keys
        return bool(api_key and isinstance(api_key, str) and len(api_key) > 20)

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured Gemini LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: Gemini LLM instance
        """
        try:
            # Try to import Google libraries
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "Google Gemini integration not available. Please install langchain-google-genai package."
                )

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY")

            if not api_key:
                raise ValueError(
                    "Google API key is required. Set GOOGLE_API_KEY or provide api_key parameter."
                )

            self._api_key = api_key

            # Configure callbacks for streaming if enabled
            callbacks = []
            if streaming:
                try:
                    from langchain.callbacks.streaming_stdout import (
                        StreamingStdOutCallbackHandler,
                    )

                    callbacks.append(StreamingStdOutCallbackHandler())
                except ImportError:
                    logger.warning(
                        "StreamingStdOutCallbackHandler not available, disabling streaming"
                    )

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )
        except ImportError as e:
            logger.error(f"Error importing Google libraries: {e}")
            raise ImportError(
                "Google Gemini integration not available. Please install langchain-google-genai package."
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Generate text using Gemini's API directly.

        Args:
            prompt: The prompt to send to Gemini
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Gemini-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Google API key is required.")

        model = kwargs.get("model", self.default_model)

        # Gemini API URL with API key
        api_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # Format content according to Gemini API
        content = []
        if system_prompt:
            content.append({"role": "system", "parts": [{"text": system_prompt}]})

        content.append({"role": "user", "parts": [{"text": prompt}]})

        data = {"contents": content, "generationConfig": {"temperature": temperature}}

        if max_tokens:
            data["generationConfig"]["maxOutputTokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Gemini API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using Gemini.

        Args:
            prompt: The prompt to send to Gemini
            tools: List of tools available to Gemini
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional Gemini-specific parameters

        Returns:
            Dict[str, Any]: Gemini response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Google API key is required.")

        model = kwargs.get("model", self.default_model)

        # Gemini API URL with API key
        api_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # Format content according to Gemini API
        content = []
        if system_prompt:
            content.append({"role": "system", "parts": [{"text": system_prompt}]})

        content.append({"role": "user", "parts": [{"text": prompt}]})

        # Format tools for Gemini's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "contents": content,
            "generationConfig": {"temperature": temperature},
            "tools": formatted_tools,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Gemini API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for Gemini's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for Gemini
        """
        # Gemini uses a format similar to OpenAI with some differences
        return {
            "functionDeclarations": [
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
            ]
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize Gemini's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in Gemini's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # Handle Gemini's function calling format
            if isinstance(input_data, dict):
                # Check for functionCall format
                if "functionCall" in input_data:
                    function_call = input_data["functionCall"]
                    if "args" in function_call:
                        return function_call["args"]

                # Check for our standard format
                if "action_input" in input_data:
                    return input_data["action_input"]

                # If none of the above, return as is
                return input_data

            # For string input, try to parse JSON
            if isinstance(input_data, str):
                try:
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    return input_data

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing Gemini input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize Gemini's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # For simple strings
        if isinstance(output_data, str):
            return output_data

        # For dictionaries and other objects, convert to JSON

        return json.dumps(output_data)

```

# test_agent/llm/openai.py

```py
# test_agent/llm/openai.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI's models.
    """

    def __init__(self):
        """Initialize the OpenAI provider."""
        self._api_key = None
        self._models = {
            "gpt-4o": {
                "context_window": 128000,
                "description": "GPT-4o - Most powerful general model from OpenAI",
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "description": "GPT-4 Turbo - Balanced performance and speed",
            },
            "gpt-4": {
                "context_window": 8192,
                "description": "GPT-4 - Capable large language model from OpenAI",
            },
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "description": "GPT-3.5 Turbo - Fast, economical model",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "openai"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available OpenAI models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default OpenAI model."""
        return "gpt-4o"  # Using the latest model as default

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # OpenAI API keys typically start with 'sk-'
        return bool(api_key and isinstance(api_key, str) and api_key.startswith("sk-"))

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured OpenAI LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: OpenAI LLM instance
        """
        try:
            # Try to import OpenAI libraries
            from langchain_openai import ChatOpenAI

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY or provide api_key parameter."
                )

            self._api_key = api_key

            # Configure callbacks for streaming if enabled
            callbacks = []
            if streaming:
                try:
                    from langchain.callbacks.streaming_stdout import (
                        StreamingStdOutCallbackHandler,
                    )

                    callbacks.append(StreamingStdOutCallbackHandler())
                except ImportError:
                    logger.warning(
                        "StreamingStdOutCallbackHandler not available, disabling streaming"
                    )

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )
        except ImportError as e:
            logger.error(f"Error importing OpenAI libraries: {e}")
            raise ImportError(
                "OpenAI integration not available. Please install langchain-openai package."
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Generate text using OpenAI's API directly.

        Args:
            prompt: The prompt to send to OpenAI
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"OpenAI API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using OpenAI.

        Args:
            prompt: The prompt to send to OpenAI
            tools: List of tools available to OpenAI
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Dict[str, Any]: OpenAI response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Format tools for OpenAI's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": formatted_tools,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"OpenAI API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for OpenAI's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for OpenAI
        """
        # OpenAI format is our standard format
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize OpenAI's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in OpenAI's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # If input_data is already a dictionary, return it
            if isinstance(input_data, dict):
                return input_data

            # If it's a string, try to parse it as JSON
            if isinstance(input_data, str):
                try:
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    pass

            # Handle nested structures
            if isinstance(input_data, dict):
                if "arguments" in input_data:
                    try:
                        # Parse arguments if it's a string
                        if isinstance(input_data["arguments"], str):
                            return json.loads(input_data["arguments"])
                        return input_data["arguments"]
                    except json.JSONDecodeError:
                        pass

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing OpenAI input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize OpenAI's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # OpenAI expects tool outputs to be strings
        if isinstance(output_data, str):
            return output_data

        # Convert non-string outputs to JSON
        try:
            return json.dumps(output_data)
        except Exception as e:
            logger.error(f"Error normalizing OpenAI output: {str(e)}")
            return str(output_data)

```

# test_agent/llm/prompts/__init__.py

```py
# test_agent/llm/prompts/__init__.py

from .language_detection import LANGUAGE_DETECTION_PROMPT
from .test_generation import TEST_GENERATION_PROMPT
from .test_fixing import TEST_FIXING_PROMPT

__all__ = ["LANGUAGE_DETECTION_PROMPT", "TEST_GENERATION_PROMPT", "TEST_FIXING_PROMPT"]

```

# test_agent/llm/prompts/language_detection.py

```py
# test_agent/llm/prompts/language_detection.py

LANGUAGE_DETECTION_PROMPT = """
You are an expert programmer tasked with identifying the programming language used in a project.

Project directory: {project_dir}

I will provide you with information about the files in this project, and you need to determine the primary programming language used.

File extensions found: {extensions}
Key files: {key_files}

Based on this information, which programming language is most likely used in this project?

Respond with just the language name (e.g., "python", "go", "javascript", etc.) and a brief explanation.
"""

```

# test_agent/llm/prompts/test_fixing.py

```py
# test_agent/llm/prompts/test_fixing.py

TEST_FIXING_PROMPT = """
I need help fixing a failing test. I'll provide:
1. The test content
2. The error output
3. An analysis of the error

Test file: {test_file}
Source file: {source_file}

Error output:
{error_output}

Error analysis:
- Error type: {error_type}
- Error message: {error_message}
- Has syntax error: {has_syntax_error}
- Has import error: {has_import_error}
- Has assertion error: {has_assertion_error}
- Has exception: {has_exception}

Current test content:
\`\`\`
{test_content}
\`\`\`

Please fix the test based on the error output. Return ONLY the corrected test code without explanations.
"""

IMPORT_ERROR_FIXING_PROMPT = """
This test is failing due to an import error. The missing module is: {missing_module}

Please fix the import statement or provide a valid alternative approach. Consider:
1. Correcting the import path
2. Using a mock or stub if appropriate
3. Adding a dependency check or skip test if the module is optional

Return ONLY the corrected code without explanations.
"""

ASSERTION_ERROR_FIXING_PROMPT = """
This test is failing due to an assertion error. The assertion that failed is:

{assertion_error}

Please modify the test to fix this assertion. Consider:
1. Checking if the expected value is correct
2. Updating the assertion to match the actual behavior if it's valid
3. Fixing any setup code that might be affecting the test

Return ONLY the corrected code without explanations.
"""

```

# test_agent/llm/prompts/test_generation.py

```py
# test_agent/llm/prompts/test_generation.py

TEST_GENERATION_PROMPT = """
You are an expert test engineer tasked with creating a comprehensive test for a source file.

I'll provide you with:
1. The source file path
2. A basic test template
3. Analysis of the source file

Your task is to enhance the template to create a complete, working test file. Follow these guidelines:
- Focus on testing the public interface of functions and classes
- Use good test design practices (arrange-act-assert pattern)
- Include edge cases and error scenarios
- Don't modify imports or basic structure - just fill in the test implementations
- Keep your tests focused, testing one thing at a time
- Use appropriate test fixtures/mocks where needed

Source file: {source_file}

Source file analysis: {file_analysis}

Basic test template:
\`\`\`
{template}
\`\`\`

Enhance this template into a complete test file. Return ONLY the complete test code without explanations.
"""

PYTHON_TEST_DOCSTRING_PROMPT = """
Write a clear docstring for the test function {test_function} that tests {target_function}.
The docstring should explain:
1. What functionality is being tested
2. Any edge cases or specific scenarios being tested
3. Expected behavior

Use a clear, concise format following Python docstring conventions.
"""

GO_TEST_COMMENT_PROMPT = """
Write a clear comment for the test function {test_function} that tests {target_function}.
The comment should explain:
1. What functionality is being tested
2. Any edge cases or specific scenarios being tested
3. Expected behavior

Use a clear, concise format following Go comment conventions.
"""

```

# test_agent/main.py

```py
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

```

# test_agent/memory/__init__.py

```py
# test_agent/memory/__init__.py

from .conversation import ConversationMemory, MemoryManager
from .cache import CacheManager

__all__ = ["ConversationMemory", "MemoryManager", "CacheManager"]
```

# test_agent/memory/cache.py

```py
# test_agent/memory/cache.py

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Set

# Configure logging
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages file analysis and test generation caching.

    This class provides persistence for expensive operations to avoid redundant processing.
    It tracks file hashes to detect changes and invalidate cache entries as needed.
    """

    def __init__(self, project_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize the cache manager.

        Args:
            project_dir: Root directory of the project
            cache_dir: Optional directory to store cache. If None, uses a temp directory.
        """
        self.project_dir = project_dir
        self.project_hash = self._hash_project_path(project_dir)

        # Set up cache directory
        if cache_dir is None:
            import tempfile

            cache_dir = os.path.join(tempfile.gettempdir(), "test_agent_cache")

        self.cache_dir = os.path.join(cache_dir, self.project_hash)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Different cache types
        self.file_hashes: Dict[str, Dict[str, Any]] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.template_cache: Dict[str, Dict[str, Any]] = {}

        # Cache file paths
        self.hashes_path = os.path.join(self.cache_dir, "file_hashes.json")
        self.analysis_path = os.path.join(self.cache_dir, "analysis_cache.json")
        self.template_path = os.path.join(self.cache_dir, "template_cache.json")

        # Load existing cache if available
        self._load_cache()

    def _hash_project_path(self, project_path: str) -> str:
        """
        Create a hash of the project path to use as a unique identifier

        Args:
            project_path: Path to the project directory

        Returns:
            MD5 hash of the normalized project path
        """
        # Normalize path to handle different OS path separators
        normalized_path = os.path.normpath(os.path.abspath(project_path))
        return hashlib.md5(normalized_path.encode()).hexdigest()[:8]

    def _load_cache(self) -> None:
        """Load all cache files from disk"""
        # Load file hashes
        if os.path.exists(self.hashes_path):
            try:
                with open(self.hashes_path, "r") as f:
                    self.file_hashes = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.file_hashes)} file hashes from {self.hashes_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading file hashes: {e}")
                self.file_hashes = {}

        # Load analysis cache
        if os.path.exists(self.analysis_path):
            try:
                with open(self.analysis_path, "r") as f:
                    self.analysis_cache = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.analysis_cache)} analysis entries from {self.analysis_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading analysis cache: {e}")
                self.analysis_cache = {}

        # Load template cache
        if os.path.exists(self.template_path):
            try:
                with open(self.template_path, "r") as f:
                    self.template_cache = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.template_cache)} template entries from {self.template_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading template cache: {e}")
                self.template_cache = {}

    def _save_cache(self, cache_type: str) -> None:
        """
        Save a specific cache type to disk

        Args:
            cache_type: Type of cache to save ("hashes", "analysis", "template")
        """
        try:
            if cache_type == "hashes":
                with open(self.hashes_path, "w") as f:
                    json.dump(self.file_hashes, f, indent=2)
                    logger.debug(
                        f"Saved {len(self.file_hashes)} file hashes to {self.hashes_path}"
                    )

            elif cache_type == "analysis":
                with open(self.analysis_path, "w") as f:
                    json.dump(self.analysis_cache, f, indent=2)
                    logger.debug(
                        f"Saved {len(self.analysis_cache)} analysis entries to {self.analysis_path}"
                    )

            elif cache_type == "template":
                with open(self.template_path, "w") as f:
                    json.dump(self.template_cache, f, indent=2)
                    logger.debug(
                        f"Saved {len(self.template_cache)} template entries to {self.template_path}"
                    )

            else:
                logger.warning(f"Unknown cache type: {cache_type}")

        except IOError as e:
            logger.error(f"Error saving {cache_type} cache: {e}")

    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute MD5 hash for a file

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash of the file contents
        """
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except IOError as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def is_file_changed(self, file_path: str) -> bool:
        """
        Check if a file has changed since last processing

        Args:
            file_path: Path to the file

        Returns:
            True if the file has changed or wasn't processed before, False otherwise
        """
        if not os.path.exists(file_path):
            return False

        # Normalize path for consistency
        file_path = os.path.abspath(file_path)

        # Check if file is in hash cache
        if file_path not in self.file_hashes:
            return True

        # Get current hash and compare with cached hash
        current_hash = self.compute_file_hash(file_path)
        return current_hash != self.file_hashes[file_path]["hash"]

    def update_file_hash(self, file_path: str) -> None:
        """
        Update hash for a file after processing

        Args:
            file_path: Path to the file
        """
        # Normalize path for consistency
        file_path = os.path.abspath(file_path)

        current_hash = self.compute_file_hash(file_path)
        if current_hash:
            self.file_hashes[file_path] = {
                "hash": current_hash,
                "last_processed": time.time(),
            }
            self._save_cache("hashes")

    def get_changed_files(self, files: List[str]) -> List[str]:
        """
        Get list of files that have changed since last processing

        Args:
            files: List of file paths to check

        Returns:
            List of file paths that have changed
        """
        return [f for f in files if self.is_file_changed(f)]

    def get_analysis_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis for a file

        Args:
            file_path: Path to the file

        Returns:
            Cached analysis or None if not in cache or file has changed
        """
        # Normalize path for consistency
        file_path = os.path.abspath(file_path)

        # Return None if file has changed
        if self.is_file_changed(file_path):
            return None

        # Return cached analysis if available
        return self.analysis_cache.get(file_path)

    def set_analysis_cache(self, file_path: str, analysis: Dict[str, Any]) -> None:
        """
        Cache analysis for a file

        Args:
            file_path: Path to the file
            analysis: Analysis data to cache
        """
        # Normalize path for consistency
        file_path = os.path.abspath(file_path)

        # Update file hash
        self.update_file_hash(file_path)

        # Store analysis in cache
        self.analysis_cache[file_path] = analysis
        self._save_cache("analysis")

    def get_template_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached template for a file

        Args:
            file_path: Path to the file

        Returns:
            Cached template or None if not in cache or file has changed
        """
        # Normalize path for consistency
        file_path = os.path.abspath(file_path)

        # Return None if file has changed
        if self.is_file_changed(file_path):
            return None

        # Return cached template if available
        return self.template_cache.get(file_path)

    def set_template_cache(self, file_path: str, template: Dict[str, Any]) -> None:
        """
        Cache template for a file

        Args:
            file_path: Path to the file
            template: Template data to cache
        """
        # Normalize path for consistency
        file_path = os.path.abspath(file_path)

        # Update file hash
        self.update_file_hash(file_path)

        # Store template in cache
        self.template_cache[file_path] = template
        self._save_cache("template")

    def clean_nonexistent_files(self) -> int:
        """
        Remove entries for files that no longer exist

        Returns:
            Number of entries removed
        """
        # Find nonexistent files in all caches
        nonexistent_files: Set[str] = set()

        for file_path in list(self.file_hashes.keys()):
            if not os.path.exists(file_path):
                nonexistent_files.add(file_path)

        # Remove entries from all caches
        count = 0
        for file_path in nonexistent_files:
            self.file_hashes.pop(file_path, None)
            self.analysis_cache.pop(file_path, None)
            self.template_cache.pop(file_path, None)
            count += 1

        # Save all caches if there were changes
        if count > 0:
            self._save_cache("hashes")
            self._save_cache("analysis")
            self._save_cache("template")

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cache

        Returns:
            Dictionary with cache statistics
        """
        # Count files in each cache
        cached_files = len(self.file_hashes)
        analysis_entries = len(self.analysis_cache)
        template_entries = len(self.template_cache)

        # Calculate cache size
        cache_size = 0
        for path in [self.hashes_path, self.analysis_path, self.template_path]:
            if os.path.exists(path):
                cache_size += os.path.getsize(path)

        # Group by last processed time
        now = time.time()
        last_day = 0
        last_week = 0
        last_month = 0

        for file_info in self.file_hashes.values():
            last_processed = file_info.get("last_processed", 0)
            age_days = (now - last_processed) / (60 * 60 * 24)

            if age_days <= 1:
                last_day += 1
            if age_days <= 7:
                last_week += 1
            if age_days <= 30:
                last_month += 1

        return {
            "cached_files": cached_files,
            "analysis_entries": analysis_entries,
            "template_entries": template_entries,
            "total_size_bytes": cache_size,
            "processed_last_day": last_day,
            "processed_last_week": last_week,
            "processed_last_month": last_month,
            "cache_location": self.cache_dir,
        }

    def clear_cache(self, cache_type: Optional[str] = None) -> Dict[str, int]:
        """
        Clear cache entries

        Args:
            cache_type: Type of cache to clear ("hashes", "analysis", "template")
                        If None, clears all caches

        Returns:
            Dictionary with count of entries cleared by cache type
        """
        result = {"hashes": 0, "analysis": 0, "template": 0}

        if cache_type is None or cache_type == "hashes":
            result["hashes"] = len(self.file_hashes)
            self.file_hashes = {}
            self._save_cache("hashes")

        if cache_type is None or cache_type == "analysis":
            result["analysis"] = len(self.analysis_cache)
            self.analysis_cache = {}
            self._save_cache("analysis")

        if cache_type is None or cache_type == "template":
            result["template"] = len(self.template_cache)
            self.template_cache = {}
            self._save_cache("template")

        return result

```

# test_agent/memory/conversation.py

```py
# test_agent/memory/conversation.py

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history and provides persistence.

    This class handles saving and loading conversation turns to/from
    a file, with rotation to prevent files from growing too large.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_turns: int = 100,
        filename: str = "conversation_history.json",
    ):
        """
        Initialize the conversation memory.

        Args:
            storage_dir: Directory to store the conversation history.
                         If None, uses a temp directory.
            max_turns: Maximum number of turns to keep in memory
            filename: Name of the conversation history file
        """
        # Set up storage directory
        if storage_dir is None:
            import tempfile

            storage_dir = os.path.join(tempfile.gettempdir(), "test_agent_memory")

        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.history_path = os.path.join(storage_dir, filename)
        self.max_turns = max_turns
        self.turns = []

        # Load existing history if available
        self._load_history()

    def _load_history(self) -> None:
        """Load conversation history from disk"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self.turns = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.turns)} conversation turns from {self.history_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading conversation history: {e}")
                # Backup the corrupted file
                if os.path.exists(self.history_path):
                    backup_path = f"{self.history_path}.bak.{int(time.time())}"
                    try:
                        os.rename(self.history_path, backup_path)
                        logger.info(
                            f"Backed up corrupted history file to {backup_path}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to backup history file: {e}")

                # Start with empty history
                self.turns = []

    def _save_history(self) -> None:
        """Save conversation history to disk"""
        try:
            # Ensure directory exists
            os.makedirs(self.storage_dir, exist_ok=True)

            with open(self.history_path, "w") as f:
                json.dump(self.turns, f, indent=2)
                logger.debug(
                    f"Saved {len(self.turns)} conversation turns to {self.history_path}"
                )
        except IOError as e:
            logger.error(f"Error saving conversation history: {e}")

    def add_turn(self, role: str, content: str) -> None:
        """
        Add a turn to the conversation history.

        Args:
            role: Role of the speaker ("system", "user", "assistant")
            content: Content of the message
        """
        turn = {"role": role, "content": content, "timestamp": time.time()}

        self.turns.append(turn)

        # Rotate if necessary
        if len(self.turns) > self.max_turns:
            # Keep last max_turns turns
            self.turns = self.turns[-self.max_turns :]

        # Save to disk
        self._save_history()

    def get_history(self, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.

        Args:
            max_turns: Maximum number of recent turns to return.
                       If None, returns all turns up to max_turns.

        Returns:
            List of conversation turns
        """
        if max_turns is None:
            return self.turns

        return self.turns[-max_turns:]

    def clear(self) -> None:
        """Clear the conversation history"""
        self.turns = []

        # Remove the history file if it exists
        if os.path.exists(self.history_path):
            try:
                os.remove(self.history_path)
                logger.info(f"Removed conversation history file: {self.history_path}")
            except Exception as e:
                logger.error(f"Failed to remove history file: {e}")

    def format_for_context(self, max_turns: Optional[int] = None) -> str:
        """
        Format the conversation history for inclusion in an LLM prompt.

        Args:
            max_turns: Maximum number of recent turns to include

        Returns:
            Formatted conversation history
        """
        history = self.get_history(max_turns)

        if not history:
            return ""

        formatted = []
        for turn in history:
            role = turn["role"]
            content = turn["content"]

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")

        return "\n\n".join(formatted)

    def get_turns_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get conversation turns for a specific role.

        Args:
            role: Role to filter by ("system", "user", "assistant")

        Returns:
            List of turns with the specified role
        """
        return [turn for turn in self.turns if turn["role"] == role]


class MemoryManager:
    """
    Central manager for different types of memory in the test agent.
    """

    def __init__(self, project_dir: str):
        """
        Initialize the memory manager.

        Args:
            project_dir: Project directory path
        """
        # Create a hash of the project path to use as part of the storage directory
        self.project_dir = project_dir
        self.project_hash = self._hash_project_path(project_dir)

        # Set up storage directory in temporary location
        import tempfile

        storage_base = os.path.join(tempfile.gettempdir(), "test_agent_memory")
        self.storage_dir = os.path.join(storage_base, self.project_hash)
        os.makedirs(self.storage_dir, exist_ok=True)

        # Initialize different memory types
        self.conversation = ConversationMemory(
            storage_dir=self.storage_dir, filename="conversation_history.json"
        )

        # Store decisions (important choices made during test generation)
        self.decisions = {}
        self.decisions_path = os.path.join(self.storage_dir, "decisions.json")
        self._load_decisions()

    def _hash_project_path(self, project_path: str) -> str:
        """
        Create a hash of the project path to use as a unique identifier

        Args:
            project_path: Path to the project directory

        Returns:
            MD5 hash of the normalized project path
        """
        import hashlib

        # Normalize path to handle different OS path separators
        normalized_path = os.path.normpath(os.path.abspath(project_path))
        return hashlib.md5(normalized_path.encode()).hexdigest()[:8]

    def _load_decisions(self) -> None:
        """Load decisions from disk"""
        if os.path.exists(self.decisions_path):
            try:
                with open(self.decisions_path, "r") as f:
                    self.decisions = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.decisions)} decisions from {self.decisions_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading decisions: {e}")
                self.decisions = {}

    def _save_decisions(self) -> None:
        """Save decisions to disk"""
        try:
            with open(self.decisions_path, "w") as f:
                json.dump(self.decisions, f, indent=2)
                logger.debug(
                    f"Saved {len(self.decisions)} decisions to {self.decisions_path}"
                )
        except IOError as e:
            logger.error(f"Error saving decisions: {e}")

    def record_decision(self, key: str, value: Any) -> None:
        """
        Record an important decision.

        Args:
            key: Decision identifier
            value: Decision value
        """
        self.decisions[key] = {"value": value, "timestamp": time.time()}
        self._save_decisions()

    def get_decision(self, key: str) -> Optional[Any]:
        """
        Get a recorded decision.

        Args:
            key: Decision identifier

        Returns:
            Decision value or None if not found
        """
        if key in self.decisions:
            return self.decisions[key]["value"]
        return None

    def clear_decisions(self) -> None:
        """Clear all recorded decisions"""
        self.decisions = {}

        # Remove the decisions file if it exists
        if os.path.exists(self.decisions_path):
            try:
                os.remove(self.decisions_path)
                logger.info(f"Removed decisions file: {self.decisions_path}")
            except Exception as e:
                logger.error(f"Failed to remove decisions file: {e}")

    def clear_all(self) -> None:
        """Clear all memory"""
        self.clear_decisions()
        self.conversation.clear()

        # Optionally remove the entire storage directory
        try:
            import shutil

            shutil.rmtree(self.storage_dir)
            logger.info(f"Removed memory storage directory: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to remove memory storage directory: {e}")

```

# test_agent/memory/settings.py

```py
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
                    "location_pattern": "tests_directory",
                    "naming_convention": "test_prefix",
                },
                "go": {
                    "location_pattern": "same_directory",
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

```

# test_agent/tools/__init__.py

```py
# test_agent/tools/__init__.py

from .environment_tools import (
    setup_environment,
    install_dependencies,
    cleanup_environment,
)
from .file_tools import find_files, read_file_content, write_file, create_directory
from .test_tools import run_test_command, parse_test_results, check_test_coverage

__all__ = [
    "setup_environment",
    "install_dependencies",
    "cleanup_environment",
    "find_files",
    "read_file_content",
    "write_file",
    "create_directory",
    "run_test_command",
    "parse_test_results",
    "check_test_coverage",
]

```

# test_agent/tools/environment_tools.py

```py
# test_agent/tools/environment_tools.py

import os
import shutil
import logging
import tempfile
import subprocess
from typing import List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def setup_environment(
    language: str, project_dir: str, temp_dir: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Set up a test environment for a specific language.

    Args:
        language: Programming language ("python", "go", etc.)
        project_dir: Project directory
        temp_dir: Optional temporary directory to use (if None, creates one)

    Returns:
        Tuple of (success flag, environment path, message)
    """
    if temp_dir is None:
        temp_dir = os.path.join(
            tempfile.gettempdir(),
            f"test_agent_{language}_{os.path.basename(project_dir)}",
        )

    # Create the temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    logger.info(f"Setting up {language} environment in {temp_dir}")

    if language.lower() == "python":
        return _setup_python_environment(project_dir, temp_dir)
    elif language.lower() == "go":
        return _setup_go_environment(project_dir, temp_dir)
    else:
        return False, temp_dir, f"Unsupported language: {language}"


def _setup_python_environment(project_dir: str, env_dir: str) -> Tuple[bool, str, str]:
    """
    Set up a Python virtual environment.

    Args:
        project_dir: Project directory
        env_dir: Environment directory

    Returns:
        Tuple of (success flag, environment path, message)
    """
    try:
        # Create virtual environment
        import venv

        venv.create(env_dir, with_pip=True, clear=True)

        # Get path to pip
        if os.name == "nt":  # Windows
            pip_path = os.path.join(env_dir, "Scripts", "pip")
            # python_path = os.path.join(env_dir, "Scripts", "python")
        else:  # Unix/Linux/Mac
            pip_path = os.path.join(env_dir, "bin", "pip")
            # python_path = os.path.join(env_dir, "bin", "python")

        # Install pytest
        logger.info(f"Installing pytest in {env_dir}")
        subprocess.run(
            [pip_path, "install", "-U", "pytest"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check for requirements file and install if present
        requirements_file = os.path.join(project_dir, "requirements.txt")
        if os.path.exists(requirements_file):
            logger.info(f"Installing dependencies from {requirements_file}")
            try:
                subprocess.run(
                    [pip_path, "install", "-r", requirements_file],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing project requirements: {e}")
                # Continue even if requirements installation fails
                return (
                    True,
                    env_dir,
                    f"Environment created but some dependencies failed: {e}",
                )

        # Check for setup.py or pyproject.toml
        setup_py = os.path.join(project_dir, "setup.py")
        pyproject_toml = os.path.join(project_dir, "pyproject.toml")

        if os.path.exists(setup_py):
            logger.info(f"Installing project in development mode from {setup_py}")
            try:
                subprocess.run(
                    [pip_path, "install", "-e", project_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing project in development mode: {e}")
        elif os.path.exists(pyproject_toml):
            logger.info(f"Installing project from {pyproject_toml}")
            try:
                subprocess.run(
                    [pip_path, "install", "-e", project_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error installing project from pyproject.toml: {e}")

        return True, env_dir, f"Python environment created at {env_dir}"

    except Exception as e:
        logger.error(f"Error setting up Python environment: {e}")
        return False, env_dir, f"Failed to create Python environment: {str(e)}"


def _setup_go_environment(project_dir: str, env_dir: str) -> Tuple[bool, str, str]:
    """
    Set up a Go environment.

    Args:
        project_dir: Project directory
        env_dir: Environment directory

    Returns:
        Tuple of (success flag, environment path, message)
    """
    try:
        # Check if Go is installed
        try:
            subprocess.run(
                ["go", "version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Go is not installed or not in PATH")
            return False, env_dir, "Go is not installed or not in PATH"

        # Create a temporary GOPATH
        go_path = os.path.join(env_dir, "gopath")
        os.makedirs(go_path, exist_ok=True)

        # Set environment variables
        env = os.environ.copy()
        env["GOPATH"] = go_path

        # Check if the project has go.mod
        go_mod = os.path.join(project_dir, "go.mod")
        if os.path.exists(go_mod):
            # Download dependencies
            logger.info(f"Downloading Go dependencies for {project_dir}")
            try:
                subprocess.run(
                    ["go", "mod", "download"],
                    check=True,
                    cwd=project_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error downloading Go dependencies: {e}")
                return (
                    True,
                    env_dir,
                    f"Go environment created but dependency download failed: {e}",
                )

        return True, env_dir, f"Go environment created at {env_dir}"

    except Exception as e:
        logger.error(f"Error setting up Go environment: {e}")
        return False, env_dir, f"Failed to create Go environment: {str(e)}"


def install_dependencies(
    environment_path: str, dependencies: List[str], language: str
) -> Tuple[bool, str]:
    """
    Install dependencies in the test environment.

    Args:
        environment_path: Path to the environment
        dependencies: List of dependencies to install
        language: Programming language

    Returns:
        Tuple of (success flag, message)
    """
    if not dependencies:
        return True, "No dependencies to install"

    logger.info(f"Installing {len(dependencies)} dependencies for {language}")

    try:
        if language.lower() == "python":
            # Get path to pip
            if os.name == "nt":  # Windows
                pip_path = os.path.join(environment_path, "Scripts", "pip")
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(environment_path, "bin", "pip")

            # Install dependencies
            for dependency in dependencies:
                try:
                    subprocess.run(
                        [pip_path, "install", dependency],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {dependency}: {e}")
                    return False, f"Failed to install {dependency}: {e}"

            return (
                True,
                f"Successfully installed {len(dependencies)} Python dependencies",
            )

        elif language.lower() == "go":
            # Set environment variables
            env = os.environ.copy()
            env["GOPATH"] = os.path.join(environment_path, "gopath")

            # Install dependencies using go get
            for dependency in dependencies:
                try:
                    subprocess.run(
                        ["go", "get", dependency],
                        check=True,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {dependency}: {e}")
                    return False, f"Failed to install {dependency}: {e}"

            return True, f"Successfully installed {len(dependencies)} Go dependencies"

        else:
            return False, f"Dependency installation not implemented for {language}"

    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False, f"Error installing dependencies: {str(e)}"


def cleanup_environment(environment_path: str) -> Tuple[bool, str]:
    """
    Clean up a test environment.

    Args:
        environment_path: Path to the environment

    Returns:
        Tuple of (success flag, message)
    """
    if not os.path.exists(environment_path):
        return True, f"Environment {environment_path} does not exist"

    try:
        logger.info(f"Cleaning up environment at {environment_path}")
        shutil.rmtree(environment_path, ignore_errors=True)
        return True, f"Successfully cleaned up environment at {environment_path}"
    except Exception as e:
        logger.error(f"Error cleaning up environment: {e}")
        return False, f"Error cleaning up environment: {str(e)}"

```

# test_agent/tools/file_tools.py

```py
# test_agent/tools/file_tools.py

import os

import logging
import fnmatch
from typing import List, Dict, Any, Optional, Tuple

# from ..utils.security import is_safe_path, sanitize_filename

# Configure logging
logger = logging.getLogger(__name__)


def find_files(
    root_dir: str,
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    recursive: bool = True,
    include_hidden: bool = False,
) -> List[str]:
    """
    Find files matching patterns in a directory.

    Args:
        root_dir: Root directory to search in
        patterns: List of glob patterns to match (e.g., "*.py")
        exclude_patterns: List of glob patterns to exclude
        recursive: Whether to search recursively
        include_hidden: Whether to include hidden files/directories

    Returns:
        List of matching file paths
    """
    if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
        logger.error(f"Directory does not exist: {root_dir}")
        return []

    patterns = patterns or ["*"]
    exclude_patterns = exclude_patterns or []

    # Common directories to exclude
    default_exclude_dirs = [
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        ".venv",
        ".env",
    ]

    matching_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories if not including hidden
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        # Skip common exclude directories
        dirnames[:] = [d for d in dirnames if d not in default_exclude_dirs]

        # Process files in this directory
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # Skip hidden files if not including hidden
                if not include_hidden and filename.startswith("."):
                    continue

                # Check if file should be excluded
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, root_dir)

                if any(
                    fnmatch.fnmatch(rel_path, ex_pattern)
                    for ex_pattern in exclude_patterns
                ):
                    continue

                matching_files.append(file_path)

        # Stop recursion if not recursive
        if not recursive:
            break

    return matching_files


def read_file_content(
    file_path: str, encoding: str = "utf-8"
) -> Tuple[bool, str, Optional[str]]:
    """
    Read content from a file safely.

    Args:
        file_path: Path to the file
        encoding: File encoding

    Returns:
        Tuple of (success flag, message, file content or None)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", None

    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}", None

    # Try with different encodings
    encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    # First try the requested encoding
    if encoding not in encodings_to_try:
        encodings_to_try.insert(0, encoding)

    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            return (
                True,
                f"Successfully read file with encoding {enc}: {file_path}",
                content,
            )
        except UnicodeDecodeError:
            # Try the next encoding
            continue
        except Exception as e:
            return False, f"Failed to read file: {str(e)}", None

    # If we get here, none of the encodings worked
    # Try to read as binary and detect encoding
    try:
        import chardet

        with open(file_path, "rb") as f:
            binary_content = f.read()
        detected = chardet.detect(binary_content)
        detected_encoding = detected["encoding"] or "utf-8"

        with open(file_path, "r", encoding=detected_encoding) as f:
            content = f.read()
        return (
            True,
            f"Successfully read file with detected encoding {detected_encoding}: {file_path}",
            content,
        )
    except Exception as e:
        return False, f"Failed to read file: {str(e)}", None


def write_file(
    file_path: str, content: str, encoding: str = "utf-8", overwrite: bool = True
) -> Tuple[bool, str]:
    """
    Write content to a file safely.

    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding
        overwrite: Whether to overwrite existing file

    Returns:
        Tuple of (success flag, message)
    """
    # Check if file exists and overwrite flag
    if os.path.exists(file_path) and not overwrite:
        return False, f"File already exists and overwrite is False: {file_path}"

    # Ensure directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            return False, f"Failed to create directory {directory}: {str(e)}"

    try:
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        return True, f"Successfully wrote file: {file_path}"
    except Exception as e:
        return False, f"Failed to write file: {str(e)}"


def create_directory(directory_path: str, exist_ok: bool = True) -> Tuple[bool, str]:
    """
    Create a directory safely.

    Args:
        directory_path: Path to the directory
        exist_ok: Whether it's okay if the directory already exists

    Returns:
        Tuple of (success flag, message)
    """
    try:
        os.makedirs(directory_path, exist_ok=exist_ok)
        return True, f"Successfully created directory: {directory_path}"
    except FileExistsError:
        return False, f"Directory already exists: {directory_path}"
    except Exception as e:
        return False, f"Failed to create directory: {str(e)}"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {"exists": False, "path": file_path}

    try:
        stat_info = os.stat(file_path)
        return {
            "exists": True,
            "path": file_path,
            "size": stat_info.st_size,
            "last_modified": stat_info.st_mtime,
            "is_file": os.path.isfile(file_path),
            "is_dir": os.path.isdir(file_path),
            "extension": (
                os.path.splitext(file_path)[1] if os.path.isfile(file_path) else ""
            ),
        }
    except Exception as e:
        return {"exists": True, "path": file_path, "error": str(e)}


def copy_file(
    source_path: str, dest_path: str, overwrite: bool = True
) -> Tuple[bool, str]:
    """
    Copy a file safely.

    Args:
        source_path: Path to the source file
        dest_path: Path to the destination
        overwrite: Whether to overwrite existing file

    Returns:
        Tuple of (success flag, message)
    """
    if not os.path.exists(source_path):
        return False, f"Source file does not exist: {source_path}"

    if not os.path.isfile(source_path):
        return False, f"Source path is not a file: {source_path}"

    if os.path.exists(dest_path) and not overwrite:
        return (
            False,
            f"Destination file already exists and overwrite is False: {dest_path}",
        )

    # Ensure destination directory exists
    dest_dir = os.path.dirname(dest_path)
    if dest_dir and not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            return False, f"Failed to create destination directory {dest_dir}: {str(e)}"

    try:
        import shutil

        shutil.copy2(source_path, dest_path)
        return True, f"Successfully copied {source_path} to {dest_path}"
    except Exception as e:
        return False, f"Failed to copy file: {str(e)}"

```

# test_agent/tools/test_tools.py

```py
# test_agent/tools/test_tools.py

import os
import re
import logging
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


def run_test_command(
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """
    Run a test command and return the results.

    Args:
        command: Command to run as a list of strings
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        Dictionary with test command results
    """
    logger.info(f"Running test command: {' '.join(command)}")

    result = {
        "command": command,
        "success": False,
        "return_code": None,
        "stdout": None,
        "stderr": None,
        "error": None,
        "timed_out": False,
    }

    try:
        # Set up environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Run the command
        if capture_output:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=process_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
        else:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=process_env,
                timeout=timeout,
            )

        result["return_code"] = process.returncode
        result["success"] = process.returncode == 0

    except subprocess.TimeoutExpired as e:
        result["error"] = f"Command timed out after {timeout} seconds"
        result["timed_out"] = True
        logger.warning(f"Test command timed out: {' '.join(command)}")
        logger.error(f"Error: {e}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error running test command: {str(e)}")

    return result


def parse_test_results(test_output: str, format: str = "auto") -> Dict[str, Any]:
    """
    Parse test results from the output of a test command.

    Args:
        test_output: Output from test command
        format: Format of the test output ("auto", "pytest", "junit", "go")

    Returns:
        Dictionary with parsed test results
    """
    # Determine format if auto
    if format == "auto":
        if "pytest" in test_output.lower():
            format = "pytest"
        elif "<testsuites>" in test_output or "<testsuite>" in test_output:
            format = "junit"
        elif "=== RUN" in test_output:
            format = "go"
        else:
            # Default to pytest
            format = "pytest"

    if format == "pytest":
        return _parse_pytest_output(test_output)
    elif format == "junit":
        return _parse_junit_output(test_output)
    elif format == "go":
        return _parse_go_output(test_output)
    else:
        logger.warning(f"Unknown test output format: {format}")
        return {
            "success": False,
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "tests": [],
            "error": f"Unknown test output format: {format}",
        }


def _parse_pytest_output(output: str) -> Dict[str, Any]:
    """
    Parse pytest output.

    Args:
        output: pytest output

    Returns:
        Dictionary with parsed results
    """
    result = {
        "success": False,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests": [],
    }

    # Extract test results
    # Example: "2 passed, 1 failed, 3 skipped"
    summary_match = re.search(
        r"(\d+) passed(?:, )?(\d+)? failed(?:, )?(\d+)? skipped", output
    )
    if summary_match:
        passed = int(summary_match.group(1) or 0)
        failed = int(summary_match.group(2) or 0)
        skipped = int(summary_match.group(3) or 0)

        result["tests_passed"] = passed
        result["tests_failed"] = failed
        result["tests_skipped"] = skipped
        result["tests_total"] = passed + failed + skipped
        result["success"] = failed == 0

    # Extract individual test results
    test_pattern = r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\s+(.+?)(?:\[|$)"
    test_matches = re.finditer(test_pattern, output)

    for match in test_matches:
        status = match.group(1)
        test_name = match.group(2).strip()

        test_result = {
            "name": test_name,
            "status": status.lower(),
            "message": None,
        }

        # Try to extract error message for failed tests
        if status in ["FAILED", "ERROR"]:
            # Look for error message after the test name
            error_match = re.search(
                rf"{re.escape(test_name)}.*?\n(.*?)(?:\n\n|\n=====|$)",
                output,
                re.DOTALL,
            )
            if error_match:
                test_result["message"] = error_match.group(1).strip()

        result["tests"].append(test_result)

    return result


def _parse_junit_output(output: str) -> Dict[str, Any]:
    """
    Parse JUnit XML output.

    Args:
        output: JUnit XML output

    Returns:
        Dictionary with parsed results
    """
    result = {
        "success": False,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests": [],
    }

    try:
        # Parse XML
        root = ET.fromstring(output)

        # Find all test cases
        test_cases = root.findall(".//testcase")

        for test_case in test_cases:
            test_name = test_case.get("name", "")
            class_name = test_case.get("classname", "")
            full_name = f"{class_name}.{test_name}" if class_name else test_name

            # Check status
            failure = test_case.find("failure")
            error = test_case.find("error")
            skipped = test_case.find("skipped")

            if failure is not None:
                status = "failed"
                message = failure.get("message", "") or failure.text
                result["tests_failed"] += 1
            elif error is not None:
                status = "error"
                message = error.get("message", "") or error.text
                result["tests_failed"] += 1
            elif skipped is not None:
                status = "skipped"
                message = skipped.get("message", "") or skipped.text
                result["tests_skipped"] += 1
            else:
                status = "passed"
                message = None
                result["tests_passed"] += 1

            result["tests"].append(
                {
                    "name": full_name,
                    "status": status,
                    "message": message,
                }
            )

        result["tests_total"] = len(test_cases)
        result["success"] = result["tests_failed"] == 0

    except Exception as e:
        logger.error(f"Error parsing JUnit XML: {str(e)}")
        result["error"] = f"Error parsing JUnit XML: {str(e)}"

    return result


def _parse_go_output(output: str) -> Dict[str, Any]:
    """
    Parse Go test output.

    Args:
        output: Go test output

    Returns:
        Dictionary with parsed results
    """
    result = {
        "success": False,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests": [],
    }

    # Extract test results
    # Example: "ok      package/path    0.015s"
    # Example: "FAIL    package/path    0.015s"

    # Track tests
    current_test = None
    tests = {}

    # Parse line by line
    for line in output.splitlines():
        # Check for test start
        start_match = re.match(r"=== RUN\s+(\S+)", line)
        if start_match:
            test_name = start_match.group(1)
            current_test = test_name
            tests[test_name] = {
                "name": test_name,
                "status": "unknown",
                "message": [],
            }
            continue

        # Check for test pass
        pass_match = re.match(r"--- PASS:\s+(\S+)", line)
        if pass_match:
            test_name = pass_match.group(1)
            if test_name in tests:
                tests[test_name]["status"] = "passed"
            result["tests_passed"] += 1
            continue

        # Check for test fail
        fail_match = re.match(r"--- FAIL:\s+(\S+)", line)
        if fail_match:
            test_name = fail_match.group(1)
            if test_name in tests:
                tests[test_name]["status"] = "failed"
            result["tests_failed"] += 1
            continue

        # Check for test skip
        skip_match = re.match(r"--- SKIP:\s+(\S+)", line)
        if skip_match:
            test_name = skip_match.group(1)
            if test_name in tests:
                tests[test_name]["status"] = "skipped"
            result["tests_skipped"] += 1
            continue

        # Collect output for current test
        if current_test and current_test in tests:
            tests[current_test]["message"].append(line)

    # Process tests
    for test_name, test_info in tests.items():
        if test_info["status"] == "unknown":
            # Assume it passed if we didn't see a result
            test_info["status"] = "passed"
            result["tests_passed"] += 1

        # Join message lines
        test_info["message"] = "\n".join(test_info["message"]).strip()

        # Add to results
        result["tests"].append(
            {
                "name": test_info["name"],
                "status": test_info["status"],
                "message": test_info["message"] if test_info["message"] else None,
            }
        )

    # Set totals
    result["tests_total"] = len(tests)
    result["success"] = result["tests_failed"] == 0 and result["tests_total"] > 0

    # Check for overall success/failure
    if "FAIL" in output:
        result["success"] = False

    return result


def check_test_coverage(
    project_dir: str,
    test_dir: Optional[str] = None,
    language: str = "python",
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check test coverage for a project.

    Args:
        project_dir: Project directory
        test_dir: Test directory (if None, uses default for language)
        language: Programming language
        exclude_patterns: Patterns to exclude from coverage

    Returns:
        Dictionary with coverage results
    """
    result = {
        "success": False,
        "total_coverage": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "covered_branches": 0,
        "total_branches": 0,
        "files": [],
        "error": None,
    }

    try:
        if language.lower() == "python":
            return _check_python_coverage(project_dir, test_dir, exclude_patterns)
        elif language.lower() == "go":
            return _check_go_coverage(project_dir, test_dir, exclude_patterns)
        else:
            result["error"] = f"Coverage checking not implemented for {language}"
            return result
    except Exception as e:
        logger.error(f"Error checking coverage: {str(e)}")
        result["error"] = f"Error checking coverage: {str(e)}"
        return result


def _check_python_coverage(
    project_dir: str,
    test_dir: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check test coverage for a Python project using coverage.py.

    Args:
        project_dir: Project directory
        test_dir: Test directory
        exclude_patterns: Patterns to exclude

    Returns:
        Dictionary with coverage results
    """
    result = {
        "success": False,
        "total_coverage": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "covered_branches": 0,
        "total_branches": 0,
        "files": [],
        "error": None,
    }

    try:
        # Check if coverage is installed
        try:
            import coverage
        except ImportError:
            result["error"] = "coverage.py is not installed"
            return result

        # Set up coverage
        cov = coverage.Coverage(
            source=[project_dir],
            omit=exclude_patterns,
        )

        # Determine test command
        if test_dir is None:
            test_dir = os.path.join(project_dir, "tests")

        if not os.path.exists(test_dir):
            result["error"] = f"Test directory not found: {test_dir}"
            return result

        # Run tests with coverage
        cov.start()

        # Import and run pytest
        import pytest

        pytest.main(["-xvs", test_dir])

        cov.stop()
        cov.save()

        # Get coverage data
        coverage_data = cov.get_data()
        total_lines = 0
        covered_lines = 0

        # Get file coverage
        for file_path in coverage_data.measured_files():
            rel_path = os.path.relpath(file_path, project_dir)

            # Skip files in excluded patterns
            if exclude_patterns and any(
                re.match(pattern, rel_path) for pattern in exclude_patterns
            ):
                continue

            # Get line coverage for file
            file_lines = coverage_data.lines(file_path)
            file_missing = coverage_data.missing_lines(file_path)

            file_total_lines = len(file_lines)
            file_covered_lines = file_total_lines - len(file_missing)

            file_coverage = 0.0
            if file_total_lines > 0:
                file_coverage = (file_covered_lines / file_total_lines) * 100.0

            result["files"].append(
                {
                    "path": rel_path,
                    "coverage": file_coverage,
                    "covered_lines": file_covered_lines,
                    "total_lines": file_total_lines,
                }
            )

            total_lines += file_total_lines
            covered_lines += file_covered_lines

        # Calculate overall coverage
        if total_lines > 0:
            result["line_coverage"] = (covered_lines / total_lines) * 100.0
            result["total_coverage"] = result["line_coverage"]

        result["covered_lines"] = covered_lines
        result["total_lines"] = total_lines
        result["success"] = True

    except Exception as e:
        logger.error(f"Error checking Python coverage: {str(e)}")
        result["error"] = f"Error checking Python coverage: {str(e)}"

    return result


def _check_go_coverage(
    project_dir: str,
    test_dir: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check test coverage for a Go project.

    Args:
        project_dir: Project directory
        test_dir: Test directory
        exclude_patterns: Patterns to exclude

    Returns:
        Dictionary with coverage results
    """
    result = {
        "success": False,
        "total_coverage": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "covered_branches": 0,
        "total_branches": 0,
        "files": [],
        "error": None,
    }

    try:
        # Determine test command
        if test_dir is None:
            # Use whole project
            test_dir = "./..."

        # Create coverage output file
        coverage_file = os.path.join(project_dir, "coverage.out")

        # Run go test with coverage
        cmd = ["go", "test", "-coverprofile", coverage_file, test_dir]

        process = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            result["error"] = f"Error running go test: {process.stderr}"
            return result

        # Check if coverage file was created
        if not os.path.exists(coverage_file):
            result["error"] = "Coverage file not created"
            return result

        # Parse coverage file
        with open(coverage_file, "r") as f:
            coverage_data = f.readlines()

        # First line is mode, skip it
        coverage_data = coverage_data[1:]

        # Process each line
        total_statements = 0
        covered_statements = 0

        for line in coverage_data:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            file_path = parts[0]
            coverage_info = parts[1]
            coverage_percent = float(parts[2].rstrip("%"))

            # Extract covered/total from coverage_info (e.g. "10/20")
            coverage_parts = coverage_info.split("/")
            if len(coverage_parts) == 2:
                file_covered = int(coverage_parts[0])
                file_total = int(coverage_parts[1])

                total_statements += file_total
                covered_statements += file_covered

                # Get relative path
                rel_path = os.path.relpath(file_path, project_dir)

                # Skip excluded patterns
                if exclude_patterns and any(
                    re.match(pattern, rel_path) for pattern in exclude_patterns
                ):
                    continue

                result["files"].append(
                    {
                        "path": rel_path,
                        "coverage": coverage_percent,
                        "covered_lines": file_covered,
                        "total_lines": file_total,
                    }
                )

        # Calculate overall coverage
        if total_statements > 0:
            result["line_coverage"] = (covered_statements / total_statements) * 100.0
            result["total_coverage"] = result["line_coverage"]

        result["covered_lines"] = covered_statements
        result["total_lines"] = total_statements
        result["success"] = True

        # Clean up coverage file
        if os.path.exists(coverage_file):
            os.remove(coverage_file)

    except Exception as e:
        logger.error(f"Error checking Go coverage: {str(e)}")
        result["error"] = f"Error checking Go coverage: {str(e)}"

    return result

```

# test_agent/utils/__init__.py

```py
# test_agent/utils/__init__.py

from .logging import setup_logging, get_logger
from .security import mask_api_key, is_safe_path, sanitize_filename
from .api_utils import get_api_key, get_last_provider, save_api_key, save_last_provider

__all__ = [
    "setup_logging",
    "get_logger",
    "mask_api_key",
    "is_safe_path",
    "sanitize_filename",
    "get_api_key",
    "get_last_provider",
    "save_api_key",
    "save_last_provider",
]

```

# test_agent/utils/api_utils.py

```py
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

```

# test_agent/utils/logging.py

```py
# test_agent/utils/logging.py

import os
import sys
import logging
from typing import Optional

# Default logging format
DEFAULT_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file
        log_format: Optional log format string
        date_format: Optional date format string
        use_colors: Whether to use colored output
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Set up format
    log_format = log_format or DEFAULT_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers = []

    # Add console handler with colors if requested
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if use_colors:
        try:
            import colorlog

            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s" + log_format,
                datefmt=date_format,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
            console_handler.setFormatter(color_formatter)
        except ImportError:
            # Fallback to non-colored if colorlog not available
            console_formatter = logging.Formatter(log_format, datefmt=date_format)
            console_handler.setFormatter(console_formatter)
    else:
        console_formatter = logging.Formatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

```

# test_agent/utils/security.py

```py
# test_agent/utils/security.py

import re


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """
    Mask an API key for logging/display purposes.

    Args:
        api_key: API key to mask
        visible_chars: Number of characters to leave visible at start and end

    Returns:
        Masked API key
    """
    if not api_key or len(api_key) <= visible_chars * 2:
        return "***"

    return (
        api_key[:visible_chars]
        + "*" * (len(api_key) - visible_chars * 2)
        + api_key[-visible_chars:]
    )


def is_safe_path(path: str) -> bool:
    """
    Check if a file path is safe (no directory traversal, etc.).

    Args:
        path: File path to check

    Returns:
        True if the path is safe, False otherwise
    """
    # Check for directory traversal
    if ".." in path:
        return False

    # Check for absolute paths
    if path.startswith("/") or path.startswith("\\"):
        return False

    # Check for environment variables
    if "$" in path:
        return False

    # Check for other dangerous patterns
    dangerous_patterns = [
        r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\.|$)",  # Windows reserved names
        r"[<>:|?*]",  # Invalid characters in filenames
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, path, re.IGNORECASE):
            return False

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to make it safe for file operations.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"|?*\\\/]', "_", filename)

    # Replace leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")

    # Ensure it's not a reserved name in Windows
    if re.match(
        r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\.|$)", sanitized, re.IGNORECASE
    ):
        sanitized = "_" + sanitized

    return sanitized

```

# test_agent/workflow/__init__.py

```py
# test_agent/workflow/__init__.py

from .state import WorkflowState, TestStatus, FileInfo, TestInfo
from .graph import TestGenerationGraph

__all__ = ["WorkflowState", "TestStatus", "TestGenerationGraph", "FileInfo", "TestInfo"]

```

# test_agent/workflow/graph.py

```py
# test_agent/workflow/graph.py

import logging
from typing import Callable, Awaitable

# Import conditional to handle both LangGraph and LangChain
try:
    # LangGraph imports
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for LangChain
    from langchain.graphs import StateGraph, END

from .state import WorkflowState
from .nodes import (
    initialize_workflow,
    detect_project_language,
    analyze_project,
    analyze_files,
    generate_test_paths,
    generate_tests,
    execute_tests,
    fix_tests,
    complete_workflow,
    handle_error,
)

# Configure logging
logger = logging.getLogger(__name__)

# Type for node function
NodeFn = Callable[[WorkflowState], Awaitable[WorkflowState]]


class TestGenerationGraph:
    """
    Graph definition for the test generation workflow.
    """

    def __init__(self):
        """Initialize the workflow graph."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the workflow graph with all nodes and edges.

        Returns:
            StateGraph: The constructed workflow graph
        """
        # Create a new StateGraph with WorkflowState
        try:
            # LangGraph with type
            graph = StateGraph(WorkflowState)
        except TypeError:
            # Fallback for older versions, manually handle type validation
            graph = StateGraph()

        # Add all nodes
        graph.add_node("initialization", initialize_workflow)
        graph.add_node("language_detection", detect_project_language)
        graph.add_node("project_analysis", analyze_project)
        graph.add_node("file_analysis", analyze_files)
        graph.add_node("test_path_generation", generate_test_paths)
        graph.add_node("test_generation", generate_tests)
        graph.add_node("test_execution", execute_tests)
        graph.add_node("test_fixing", fix_tests)
        graph.add_node("complete", complete_workflow)
        graph.add_node("error", handle_error)

        # Function to determine next state based on next_phase
        def route_by_next_phase(state):
            next_phase = state.next_phase
            logger.info(f"Routing from {state.current_phase} to {next_phase or 'END'}")

            if next_phase is None:
                return "end"
            return next_phase

        # Add conditional edges for each processing node
        graph.add_conditional_edges(
            "initialization",
            route_by_next_phase,
            {
                "language_detection": "language_detection",
                "project_analysis": "project_analysis",
                "error": "error",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "language_detection",
            route_by_next_phase,
            {"project_analysis": "project_analysis", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "project_analysis",
            route_by_next_phase,
            {"file_analysis": "file_analysis", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "file_analysis",
            route_by_next_phase,
            {
                "test_path_generation": "test_path_generation",
                "error": "error",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "test_path_generation",
            route_by_next_phase,
            {"test_generation": "test_generation", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "test_generation",
            route_by_next_phase,
            {"test_execution": "test_execution", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "test_execution",
            route_by_next_phase,
            {
                "test_fixing": "test_fixing",
                "complete": "complete",
                "error": "error",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "test_fixing",
            route_by_next_phase,
            {"complete": "complete", "error": "error", "end": END},
        )

        # Complete and error should end the workflow
        graph.add_conditional_edges("complete", route_by_next_phase, {"end": END})

        graph.add_conditional_edges("error", route_by_next_phase, {"end": END})

        # Set the entry point
        graph.set_entry_point("initialization")

        # Compile the graph
        try:
            # LangGraph uses compile, LangChain may not
            graph = graph.compile()
        except AttributeError:
            # If compile is not available, just continue
            pass

        return graph

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the workflow with the given initial state.

        Args:
            state: Initial workflow state

        Returns:
            The final workflow state
        """
        logger.info("Starting test generation workflow")

        try:
            # LangGraph/LangChain interface might differ slightly
            try:
                # LangGraph async invoke
                result = await self.graph.ainvoke(state)
            except AttributeError:
                # Fallback to sync invoke
                result = self.graph.invoke(state)

            return result
        except Exception as e:
            logger.exception(f"Error running workflow: {str(e)}")

            # Create error state
            state.errors.append(
                {
                    "phase": "workflow",
                    "error": f"Workflow execution error: {str(e)}",
                    "type": "exception",
                }
            )
            state.is_completed = True
            state.next_phase = None

            return state

```

# test_agent/workflow/nodes/__init__.py

```py
# test_agent/workflow/nodes/__init__.py

from .initialization import initialize_workflow
from .language_detection import detect_project_language
from .project_analysis import analyze_project
from .file_analysis import analyze_files
from .test_path import generate_test_paths
from .test_generation import generate_tests
from .test_execution import execute_tests
from .test_fixing import fix_tests
from .complete import complete_workflow
from .error import handle_error

__all__ = [
    "initialize_workflow",
    "detect_project_language",
    "analyze_project",
    "analyze_files",
    "generate_test_paths",
    "generate_tests",
    "execute_tests",
    "fix_tests",
    "complete_workflow",
    "handle_error",
]

```

# test_agent/workflow/nodes/complete.py

```py
# test_agent/workflow/nodes/complete.py

import os
import logging
import time

from test_agent.workflow import WorkflowState, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


def generate_summary(state: WorkflowState) -> str:
    """
    Generate a summary of the test generation process.

    Args:
        state: Workflow state

    Returns:
        Summary text
    """
    # Calculate counts
    total_files = len(state.project.source_files)
    tests_generated = len([t for t in state.tests.values() if t.content])
    passed_tests = len(
        [t for t in state.tests.values() if t.status == TestStatus.PASSED]
    )
    fixed_tests = len([t for t in state.tests.values() if t.status == TestStatus.FIXED])
    failed_tests = len(
        [t for t in state.tests.values() if t.status == TestStatus.FAILED]
    )
    error_tests = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped_tests = len(
        [t for t in state.tests.values() if t.status == TestStatus.SKIPPED]
    )

    # Calculate timing if available
    time_taken = "unknown"
    if state.start_time and state.end_time:
        time_taken = f"{state.end_time - state.start_time:.2f}s"

    # Generate summary text
    summary = f"""
Test Generation Summary
----------------------

Project: {state.project.root_directory}
Language: {state.project.language}
Test Directory: {state.project.test_directory}

Files & Tests:
- Source files analyzed: {total_files}
- Tests generated: {tests_generated}
- Tests skipped (already existed): {skipped_tests}

Test Results:
- Passed: {passed_tests}
- Fixed: {fixed_tests}
- Failed: {failed_tests}
- Errors: {error_tests}
- Success rate: {(passed_tests + fixed_tests) / max(1, tests_generated - skipped_tests):.1%}

Time taken: {time_taken}
"""

    # Add error summary if there were errors
    if state.errors:
        summary += "\nErrors encountered:\n"
        for i, error in enumerate(state.errors[:5], 1):  # Show first 5 errors
            summary += f"- Error {i}: {error.get('error', 'Unknown error')}\n"

        if len(state.errors) > 5:
            summary += f"- ... and {len(state.errors) - 5} more errors\n"

    # Add warnings summary if there were warnings
    if state.warnings:
        summary += "\nWarnings:\n"
        for i, warning in enumerate(state.warnings[:5], 1):  # Show first 5 warnings
            summary += f"- Warning {i}: {warning.get('error', 'Unknown warning')}\n"

        if len(state.warnings) > 5:
            summary += f"- ... and {len(state.warnings) - 5} more warnings\n"

    # Add list of test files
    summary += "\nGenerated test files:\n"

    # Sort by status for a better overview
    test_files = []

    for test_info in state.tests.values():
        if test_info.status in [TestStatus.PASSED, TestStatus.FIXED]:
            test_files.append((test_info.test_path, "✅"))
        elif test_info.status == TestStatus.FAILED:
            test_files.append((test_info.test_path, "❌"))
        elif test_info.status == TestStatus.ERROR:
            test_files.append((test_info.test_path, "⚠️"))
        elif test_info.status == TestStatus.SKIPPED:
            test_files.append((test_info.test_path, "⏭️"))

    # Sort by path for a consistent output
    test_files.sort()

    # Add files to summary, truncating if too many
    max_files_to_show = 20
    for i, (path, status) in enumerate(test_files[:max_files_to_show], 1):
        rel_path = os.path.relpath(path, state.project.root_directory)
        summary += f"{status} {rel_path}\n"

    if len(test_files) > max_files_to_show:
        summary += f"... and {len(test_files) - max_files_to_show} more test files\n"

    # Add next steps
    summary += f"""
Next Steps:
- Run the tests: cd {state.project.root_directory} && {'python -m pytest' if state.project.language == 'python' else 'go test ./...'}
- Review and improve tests as needed
- Add tests for specific edge cases
"""

    return summary


async def complete_workflow(state: WorkflowState) -> WorkflowState:
    """
    Node to complete the workflow and generate summary.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    logger.info("Completing workflow")

    # Mark as completed
    state.is_completed = True

    # Set end time
    state.end_time = time.time()

    # Generate summary
    summary = generate_summary(state)

    # Log summary
    logger.info("Test generation completed successfully")
    logger.info(f"\n{summary}")

    # Update state
    state.current_phase = "complete"
    state.next_phase = None

    return state

```

# test_agent/workflow/nodes/error.py

```py
# test_agent/workflow/nodes/error.py

import logging
import time

from test_agent.workflow import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


def generate_error_report(state: WorkflowState) -> str:
    """
    Generate a report of errors encountered during the workflow.

    Args:
        state: Workflow state

    Returns:
        Error report text
    """
    # Generate heading
    report = """
Test Generation Error Report
---------------------------

The test generation process encountered errors that prevented completion.
"""

    # Add error details
    if state.errors:
        report += "\nErrors:\n"
        for i, error in enumerate(state.errors, 1):
            phase = error.get("phase", "unknown")
            error_msg = error.get("error", "Unknown error")
            error_type = error.get("type", "unknown")

            report += f"Error {i} (in {phase} phase): {error_msg}\n"
            report += f"  Type: {error_type}\n"

            # Add more details if available
            if "file" in error:
                report += f"  File: {error['file']}\n"
            if "details" in error:
                report += f"  Details: {error['details']}\n"

            report += "\n"
    else:
        report += "\nNo specific errors were recorded.\n"

    # Add warning details
    if state.warnings:
        report += "\nWarnings:\n"
        for i, warning in enumerate(state.warnings, 1):
            phase = warning.get("phase", "unknown")
            error_msg = warning.get("error", "Unknown warning")

            report += f"Warning {i} (in {phase} phase): {error_msg}\n"

            # Add more details if available
            if "file" in warning:
                report += f"  File: {warning['file']}\n"

            report += "\n"

    # Add troubleshooting suggestions
    report += """
Troubleshooting Suggestions:
---------------------------
"""

    # Add specific suggestions based on error types
    language_errors = [e for e in state.errors if e.get("type") == "adapter_not_found"]
    env_errors = [
        e for e in state.errors if e.get("type") == "environment_setup_failed"
    ]
    file_errors = [
        e
        for e in state.errors
        if e.get("type") in ["file_not_found", "permission_denied"]
    ]
    llm_errors = [
        e
        for e in state.errors
        if e.get("type") in ["llm_provider_not_found", "api_key_missing"]
    ]

    if language_errors:
        report += """
- Language Detection Issues:
  * Check if the project contains files with the expected extension
  * Ensure you specified the correct language manually if auto-detection failed
  * The tool currently supports Python and Go primarily
"""

    if env_errors:
        report += """
- Environment Setup Issues:
  * Check if the required tools are installed (Python/Go)
  * Ensure you have sufficient permissions to create virtual environments
  * Try clearing any existing environments and retry
"""

    if file_errors:
        report += """
- File Access Issues:
  * Check if the specified project directory exists and is accessible
  * Ensure you have read/write permissions for the project directory
  * If specific files were mentioned in errors, check their permissions
"""

    if llm_errors:
        report += """
- LLM Provider Issues:
  * Check if you specified a valid LLM provider (claude, openai, deepseek, gemini)
  * Ensure you provided a valid API key for the chosen provider
  * Check if the API key has the necessary permissions and quota
"""

    # Always add general suggestions
    report += """
General Suggestions:
  * Check the logs for more detailed error information
  * Try running with the --verbose flag for additional debugging information
  * Ensure all dependencies are installed: pip install -e .
  * Try using a different LLM provider if available
  * For more help, please report the issue with the full error log
"""

    return report


async def handle_error(state: WorkflowState) -> WorkflowState:
    """
    Node to handle workflow errors.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    logger.error("Workflow encountered errors, generating error report")

    # Set end time
    state.end_time = time.time()

    # Generate error report
    error_report = generate_error_report(state)

    # Log error report
    logger.error(f"\n{error_report}")

    # Update state
    state.current_phase = "error"
    state.next_phase = None
    state.is_completed = True

    return state

```

# test_agent/workflow/nodes/file_analysis.py

```py
# test_agent/workflow/nodes/file_analysis.py

import logging
import time
import asyncio
from typing import Optional

from test_agent.language import get_adapter
from test_agent.memory import CacheManager
from test_agent.workflow import WorkflowState, FileInfo

# Configure logging
logger = logging.getLogger(__name__)


async def analyze_single_file(
    file_info: FileInfo, language_adapter, cache_manager: Optional[CacheManager] = None
) -> FileInfo:
    """
    Analyze a single source file.

    Args:
        file_info: File information
        language_adapter: Language adapter for the file
        cache_manager: Optional cache manager

    Returns:
        Updated file information with analysis
    """
    file_path = file_info.path

    try:
        # Check cache first if available
        if cache_manager:
            cached_analysis = cache_manager.get_analysis_cache(file_path)
            if cached_analysis:
                file_info.analysis = cached_analysis
                return file_info

        # Analyze file
        analysis = language_adapter.analyze_source_file(file_path)

        # Update file info
        file_info.analysis = analysis

        # Update cache if available
        if cache_manager:
            cache_manager.set_analysis_cache(file_path, analysis)

        return file_info

    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {str(e)}")
        file_info.analysis = {"error": str(e)}
        return file_info


async def analyze_files(state: WorkflowState) -> WorkflowState:
    """
    Node to analyze source files.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    # Get language adapter
    language = state.project.language
    language_adapter = get_adapter(language)

    if not language_adapter:
        error_msg = f"No adapter found for language: {language}"
        logger.error(error_msg)
        state.errors.append(
            {"phase": "file_analysis", "error": error_msg, "type": "adapter_not_found"}
        )
        state.next_phase = "error"
        return state

    # Initialize cache manager
    cache_manager = CacheManager(state.project.root_directory)

    # Start timing
    start_time = time.time()

    # Get files to analyze
    files_to_analyze = [
        f for f in state.project.source_files if not f.skip and not f.analysis
    ]

    logger.info(f"Analyzing {len(files_to_analyze)} source files")

    # Process files in parallel using asyncio
    try:
        # Create analysis tasks
        tasks = []
        for file_info in files_to_analyze:
            tasks.append(
                analyze_single_file(file_info, language_adapter, cache_manager)
            )

        # Run tasks concurrently
        if tasks:
            analyzed_files = await asyncio.gather(*tasks)

            # Update state with analyzed files
            for analyzed_file in analyzed_files:
                # Find and update the file in state.project.source_files
                for i, file_info in enumerate(state.project.source_files):
                    if file_info.path == analyzed_file.path:
                        state.project.source_files[i] = analyzed_file
                        break

        # Calculate time taken
        time_taken = time.time() - start_time

        # Update state
        logger.info(f"File analysis complete in {time_taken:.2f}s")
        logger.info(f"Analyzed {len(tasks)} files")

        # Set next phase
        state.current_phase = "file_analysis"
        state.next_phase = "test_path_generation"

    except Exception as e:
        error_msg = f"Error during file analysis: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "file_analysis", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"

    return state

```

# test_agent/workflow/nodes/initialization.py

```py
# test_agent/workflow/nodes/initialization.py

import os
import logging
import time

from test_agent.language import get_adapter
from test_agent.workflow import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


def initialize_workflow(state: WorkflowState) -> WorkflowState:
    """
    Node to initialize the workflow.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    logger.info("Initializing workflow")

    # Set start time
    state.start_time = time.time()

    try:
        # Validate project directory
        if not state.project or not state.project.root_directory:
            error_msg = "No project directory specified"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "initialization",
                    "error": error_msg,
                    "type": "missing_project_directory",
                }
            )
            state.next_phase = "error"
            return state

        project_dir = state.project.root_directory

        if not os.path.exists(project_dir):
            error_msg = f"Project directory does not exist: {project_dir}"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "initialization",
                    "error": error_msg,
                    "type": "project_directory_not_found",
                }
            )
            state.next_phase = "error"
            return state

        if not os.path.isdir(project_dir):
            error_msg = f"Project path is not a directory: {project_dir}"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "initialization",
                    "error": error_msg,
                    "type": "project_path_not_directory",
                }
            )
            state.next_phase = "error"
            return state

        # Validate LLM provider if specified
        if state.llm and state.llm.provider:
            try:
                logger.info(f"Using LLM provider: {state.llm.provider}")
            except ValueError as e:
                error_msg = f"Invalid LLM provider: {str(e)}"
                logger.error(error_msg)
                state.errors.append(
                    {
                        "phase": "initialization",
                        "error": error_msg,
                        "type": "invalid_llm_provider",
                    }
                )
                state.next_phase = "error"
                return state

        # Validate language if specified
        if state.project.language:
            language_adapter = get_adapter(state.project.language)
            if not language_adapter:
                error_msg = f"Unsupported language: {state.project.language}"
                logger.error(error_msg)
                state.errors.append(
                    {
                        "phase": "initialization",
                        "error": error_msg,
                        "type": "unsupported_language",
                    }
                )
                state.next_phase = "error"
                return state

            logger.info(f"Using language: {state.project.language}")

        # Set the starting phase
        state.current_phase = "initialization"

        # Determine next phase
        if state.project.language:
            # Skip language detection if already specified
            logger.info("Language already specified, skipping detection")
            state.next_phase = "project_analysis"
        else:
            # Start with language detection
            logger.info("No language specified, will detect language next")
            state.next_phase = "language_detection"

        # DEBUGGING: Print the state at the end of initialization
        logger.info(
            f"State after initialization: current_phase={state.current_phase}, next_phase={state.next_phase}"
        )

        logger.info("Initialization complete")

        return state

    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "initialization", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"
        return state

    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "initialization", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"
        return state

```

# test_agent/workflow/nodes/language_detection.py

```py
# test_agent/workflow/nodes/language_detection.py

import logging

from test_agent.language import detect_language, get_adapter
from test_agent.workflow import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


async def detect_project_language(state: WorkflowState) -> WorkflowState:
    """
    Node to detect the programming language used in the project.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    project_dir = state.project.root_directory
    logger.info(f"Detecting language for project: {project_dir}")

    # Get language from state if already set
    language = state.project.language

    if language is None:
        try:
            # Auto-detect language
            language = detect_language(project_dir)

            if language:
                logger.info(f"Detected language: {language}")

                # Get language adapter
                language_adapter = get_adapter(language)

                if language_adapter:
                    # Update state with detected language
                    state.project.language = language

                    # Add a decision to the memory
                    if state.memory and hasattr(state.memory, "record_decision"):
                        state.memory.record_decision("detected_language", language)

                    logger.info(f"Language detection successful: {language}")
                else:
                    error_msg = (
                        f"Failed to find adapter for detected language: {language}"
                    )
                    logger.error(error_msg)
                    state.errors.append(
                        {
                            "phase": "language_detection",
                            "error": error_msg,
                            "type": "adapter_not_found",
                        }
                    )
            else:
                error_msg = "Could not detect language automatically"
                logger.error(error_msg)
                state.errors.append(
                    {
                        "phase": "language_detection",
                        "error": error_msg,
                        "type": "detection_failed",
                    }
                )

        except Exception as e:
            error_msg = f"Error during language detection: {str(e)}"
            logger.exception(error_msg)
            state.errors.append(
                {"phase": "language_detection", "error": error_msg, "type": "exception"}
            )

    else:
        logger.info(f"Using language from state: {language}")

    # Determine next phase based on success
    if state.project.language:
        state.current_phase = "language_detection"
        state.next_phase = "project_analysis"
    else:
        state.current_phase = "language_detection"
        state.next_phase = "error"

    return state

```

# test_agent/workflow/nodes/project_analysis.py

```py
# test_agent/workflow/nodes/project_analysis.py

import os
import logging
import time
from typing import List, Optional

from test_agent.language import get_adapter
from test_agent.memory import CacheManager
from test_agent.workflow import WorkflowState, FileInfo

# Configure logging
logger = logging.getLogger(__name__)


def _should_exclude_directory(dir_path: str) -> bool:
    """
    Check if a directory should be excluded from analysis.

    Args:
        dir_path: Directory path

    Returns:
        True if directory should be excluded, False otherwise
    """
    dir_name = os.path.basename(dir_path)

    # Common directories to exclude
    exclude_patterns = [
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
        ".pytest_cache",
        ".coverage",
    ]

    # Check if the directory name matches any exclude patterns
    for pattern in exclude_patterns:
        if dir_name == pattern or dir_name.startswith(pattern):
            logger.debug(f"Excluding directory: {dir_path} (matches pattern {pattern})")
            return True

    # Also check if __pycache__ is in the path
    if "__pycache__" in dir_path:
        logger.debug(f"Excluding directory with __pycache__ in path: {dir_path}")
        return True

    return False


def _should_exclude_file(file_path: str, file_extensions: List[str]) -> bool:
    """
    Check if a file should be excluded from analysis.

    Args:
        file_path: File path
        file_extensions: Valid file extensions to include

    Returns:
        True if file should be excluded, False otherwise
    """
    file_name = os.path.basename(file_path)

    # Skip files in __pycache__ directories
    if "__pycache__" in file_path:
        logger.debug(f"Excluding file in __pycache__: {file_path}")
        return True

    # Don't exclude non-empty __init__.py files
    if file_name == "__init__.py":
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
                    logger.debug(
                        f"Excluding empty or comment-only __init__.py: {file_path}"
                    )
                    return True
                else:
                    # Non-empty __init__.py files should be included
                    logger.debug(f"Including non-empty __init__.py: {file_path}")
                    return False
        except Exception as e:
            logger.debug(f"Error checking __init__.py content: {str(e)}")
            # If we can't read the file, assume it should be included
            return False

    # Check if it's a backup file or hidden file
    if file_name.startswith(".") or file_name.endswith("~"):
        logger.debug(f"Excluding backup/hidden file: {file_path}")
        return True

    # Check if it has a valid extension
    if not any(file_name.endswith(ext) for ext in file_extensions):
        logger.debug(
            f"Excluding file with invalid extension: {file_path} (valid extensions: {file_extensions})"
        )
        return True

    # Ensure Python files end with .py
    if (
        ".py" in file_extensions
        and not file_name.endswith(".py")
        and any(ext in file_name for ext in [".py"])
    ):
        logger.debug(f"Excluding file with partial Python extension: {file_path}")
        return True

    return False


def find_source_files(
    project_dir: str,
    file_extensions: List[str],
    excluded_dirs: Optional[List[str]] = None,
    excluded_files: Optional[List[str]] = None,
) -> List[str]:
    """
    Find all source files in a project directory.

    Args:
        project_dir: Project directory path
        file_extensions: Valid file extensions to include
        excluded_dirs: Optional list of directories to exclude
        excluded_files: Optional list of files to exclude

    Returns:
        List of source file paths
    """
    source_files = []
    excluded_dirs = excluded_dirs or []
    excluded_files = excluded_files or []

    # Normalize excluded_dirs to absolute paths
    excluded_dirs = [
        os.path.abspath(d) if not os.path.isabs(d) else d for d in excluded_dirs
    ]

    # Debug info
    logger.debug(
        f"Searching for files in {project_dir} with extensions: {file_extensions}"
    )
    logger.debug(f"Excluded dirs: {excluded_dirs}")
    logger.debug(f"Excluded files: {excluded_files}")

    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories - improve logic to handle path differences
        filtered_dirs = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            # Skip if the directory path is in excluded_dirs
            if dir_path in excluded_dirs:
                logger.debug(f"Excluding directory: {dir_path} (exact match)")
                continue

            # Skip common directories to exclude
            if _should_exclude_directory(dir_path):
                logger.debug(f"Excluding directory: {dir_path} (pattern match)")
                continue

            filtered_dirs.append(d)

        # Update dirs in place to control walk
        dirs[:] = filtered_dirs

        # Process files
        for file in files:
            file_path = os.path.join(root, file)

            # Skip if file path is in excluded_files
            if file_path in excluded_files:
                logger.debug(f"Excluding file: {file_path} (exact match)")
                continue

            # Skip if file should be excluded based on name/extension
            if _should_exclude_file(file_path, file_extensions):
                logger.debug(f"Excluding file: {file_path} (extension/pattern match)")
                continue

            # If we get here, add the file
            logger.debug(f"Found source file: {file_path}")
            source_files.append(file_path)

    logger.info(f"Found {len(source_files)} source files in {project_dir}")
    if not source_files:
        logger.warning(
            f"No source files found in {project_dir} with extensions {file_extensions}"
        )

    return source_files


async def analyze_project(state: WorkflowState) -> WorkflowState:
    """
    Node to analyze project structure and find source files.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    project_dir = state.project.root_directory
    language = state.project.language

    logger.info(f"Analyzing project: {project_dir} (language: {language})")

    try:
        # Get language adapter
        language_adapter = get_adapter(language)

        if not language_adapter:
            error_msg = f"No adapter found for language: {language}"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "project_analysis",
                    "error": error_msg,
                    "type": "adapter_not_found",
                }
            )
            state.next_phase = "error"
            return state

        # Start timing
        start_time = time.time()

        # Get file extensions from the language adapter
        file_extensions = language_adapter.file_extensions
        logger.info(f"Looking for files with extensions: {file_extensions}")

        # List all files in the project directory
        try:
            all_files = os.listdir(project_dir)
            logger.info(f"Files in project directory (non-recursive): {len(all_files)}")
            for file in all_files[:10]:  # Log first 10 files
                full_path = os.path.join(project_dir, file)
                is_dir = os.path.isdir(full_path)
                logger.debug(f"  {'[DIR]' if is_dir else '[FILE]'} {file}")
            if len(all_files) > 10:
                logger.debug(f"  ... and {len(all_files) - 10} more files/directories")
        except Exception as e:
            logger.warning(f"Failed to list directory contents: {str(e)}")

        # Find source files
        logger.info(f"Searching for source files in {project_dir}")
        logger.info(f"Excluded directories: {state.project.excluded_directories}")
        logger.info(f"Excluded files: {state.project.excluded_files}")

        source_files = find_source_files(
            project_dir,
            file_extensions,
            state.project.excluded_directories,
            state.project.excluded_files,
        )

        logger.info(f"Found {len(source_files)} source files")
        for file in source_files[:10]:  # Log first 10 source files
            logger.debug(f"  Source file: {file}")
        if len(source_files) > 10:
            logger.debug(f"  ... and {len(source_files) - 10} more source files")

        # If no source files found, perform additional checks
        if not source_files:
            logger.warning("No source files found! Performing additional checks...")

            # Try to find Python files recursively using os.walk
            py_files = []
            for root, _, files in os.walk(project_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in file_extensions):
                        py_files.append(os.path.join(root, file))

            logger.warning(f"Direct file search found {len(py_files)} Python files")
            for file in py_files[:10]:
                logger.warning(f"  Python file (direct search): {file}")

            # Check if any files are being excluded by should_skip_file
            if py_files:
                logger.warning(
                    "Checking which files are being excluded by should_skip_file..."
                )
                for file in py_files:
                    if language_adapter.should_skip_file(file):
                        logger.warning(f"  Skipped by should_skip_file: {file}")
                    else:
                        logger.warning(f"  Not skipped, but still not included: {file}")

        # Initialize cache manager for the project
        cache_manager = CacheManager(project_dir)

        # Filter out files that haven't changed since last run
        unchanged_files = 0
        changed_files = []

        for file_path in source_files:
            if cache_manager.is_file_changed(file_path):
                changed_files.append(file_path)
            else:
                unchanged_files += 1

        logger.info(f"Files changed since last run: {len(changed_files)}")
        logger.info(f"Files unchanged since last run: {unchanged_files}")

        # Detect test pattern for the project
        logger.info("Detecting project test pattern...")
        test_pattern = language_adapter.detect_project_structure(project_dir)
        logger.info(f"Detected test pattern: {test_pattern}")

        # Set the primary test directory if not already specified
        if state.project.test_directory is None and test_pattern.get(
            "primary_test_dir"
        ):
            state.project.test_directory = test_pattern.get("primary_test_dir")
            logger.info(
                f"Using detected primary test directory: {state.project.test_directory}"
            )
        elif state.project.test_directory is None:
            # Use default tests directory
            state.project.test_directory = os.path.join(project_dir, "tests")
            logger.info(f"Using default test directory: {state.project.test_directory}")

        # Create test directory if it doesn't exist
        if not os.path.exists(state.project.test_directory):
            logger.info(f"Creating test directory: {state.project.test_directory}")
            os.makedirs(state.project.test_directory, exist_ok=True)

        # Update state with project analysis results
        state.project.patterns = test_pattern
        state.project.source_files = []

        # Process each source file
        logger.info("Processing source files...")
        skipped_files = 0
        for file_path in source_files:
            # Check if file should be skipped
            if language_adapter.should_skip_file(file_path):
                logger.debug(f"Skipping file: {file_path}")
                skipped_files += 1
                continue

            # Add file info to state
            relative_path = os.path.relpath(file_path, project_dir)
            logger.debug(f"Processing file: {relative_path}")

            # Check if file has an existing test
            existing_test = language_adapter.find_corresponding_test(
                file_path, project_dir
            )
            if existing_test:
                logger.debug(f"Found existing test: {existing_test}")

            file_info = FileInfo(
                path=file_path,
                relative_path=relative_path,
                language=language,
                size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                last_modified=(
                    os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                ),
                has_existing_test=existing_test is not None,
                existing_test_path=existing_test,
            )

            state.project.source_files.append(file_info)

        logger.info(f"Skipped {skipped_files} files")

        # Calculate time taken
        time_taken = time.time() - start_time

        logger.info(f"Project analysis complete in {time_taken:.2f}s")
        logger.info(f"Found {len(state.project.source_files)} source files")
        logger.info(f"Test directory: {state.project.test_directory}")

        # Set next phase
        state.current_phase = "project_analysis"
        state.next_phase = "file_analysis"

    except Exception as e:
        error_msg = f"Error during project analysis: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "project_analysis", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"

    return state

```

# test_agent/workflow/nodes/test_execution.py

```py
# test_agent/workflow/nodes/test_execution.py

import os
import logging
import time
import asyncio
import subprocess
import tempfile
from typing import Optional, Tuple

from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


def setup_test_environment(language: str) -> Tuple[bool, str, Optional[str]]:
    """
    Set up the test environment for a specific language.

    Args:
        language: Language name

    Returns:
        Tuple of (success flag, message, environment path or None)
    """
    try:
        if language.lower() == "python":
            # Create a virtual environment for Python
            venv_dir = os.path.join(tempfile.gettempdir(), "test_agent_venv")

            if not os.path.exists(venv_dir):
                logger.info(f"Creating Python virtual environment at {venv_dir}")

                # Use Python's venv module
                import venv

                venv.create(venv_dir, with_pip=True, clear=True)

                # Install pytest
                if os.name == "nt":  # Windows
                    pip_path = os.path.join(venv_dir, "Scripts", "pip")
                else:  # Unix/Linux/Mac
                    pip_path = os.path.join(venv_dir, "bin", "pip")

                subprocess.run(
                    [pip_path, "install", "pytest"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                logger.info("Installed pytest in virtual environment")

            return True, f"Test environment set up at {venv_dir}", venv_dir

        elif language.lower() == "go":
            # Go doesn't need a special environment, it uses go test command
            return True, "Using system Go installation for tests", None

        else:
            return False, f"No test environment setup implemented for {language}", None

    except Exception as e:
        error_msg = f"Error setting up test environment: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, None


async def run_test(
    test_info: TestInfo, language: str, env_path: Optional[str] = None
) -> TestInfo:
    """
    Run a single test.

    Args:
        test_info: Test information
        language: Language name
        env_path: Optional path to the test environment

    Returns:
        Updated test information
    """
    if test_info.status == TestStatus.SKIPPED:
        # Skip existing tests
        return test_info

    if not os.path.exists(test_info.test_path):
        test_info.status = TestStatus.ERROR
        test_info.error_message = f"Test file not found: {test_info.test_path}"
        return test_info

    try:
        # Update status
        test_info.status = TestStatus.RUNNING

        # Get command based on language
        if language.lower() == "python":
            if os.name == "nt":  # Windows
                python_path = os.path.join(env_path, "Scripts", "python")
            else:  # Unix/Linux/Mac
                python_path = os.path.join(env_path, "bin", "python")

            # Set up environment variables
            env = os.environ.copy()

            # Add project directory to PYTHONPATH to allow imports
            project_dir = os.path.dirname(test_info.source_file)
            if "PYTHONPATH" in env:
                if os.name == "nt":  # Windows
                    env["PYTHONPATH"] = f"{project_dir};{env['PYTHONPATH']}"
                else:  # Unix/Linux/Mac
                    env["PYTHONPATH"] = f"{project_dir}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = project_dir

            # Create command
            command = [python_path, "-m", "pytest", "-v", test_info.test_path]

        elif language.lower() == "go":
            # Get directory containing the test file
            # test_dir = os.path.dirname(test_info.test_path)

            # Create command
            command = ["go", "test", "-v", test_info.test_path]

            # Set up environment variables
            env = os.environ.copy()

        else:
            test_info.status = TestStatus.ERROR
            test_info.error_message = f"No test execution implemented for {language}"
            return test_info

        # Run the test
        logger.info(f"Running test: {test_info.test_path}")

        # Execute the command
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(test_info.test_path),
        )

        # Get output
        stdout, stderr = await process.communicate()

        # Store execution result
        test_info.execution_result = (
            f"STDOUT:\n{stdout.decode()}\n\nSTDERR:\n{stderr.decode()}"
        )

        # Update status based on exit code
        if process.returncode == 0:
            test_info.status = TestStatus.PASSED
            logger.info(f"Test passed: {test_info.test_path}")
        else:
            # Check if it's a test failure (assertion failed) or an error
            if "AssertionError" in test_info.execution_result:
                test_info.status = TestStatus.FAILED
                logger.info(f"Test failed (assertions): {test_info.test_path}")
            else:
                test_info.status = TestStatus.ERROR
                test_info.error_message = (
                    f"Test execution error (code {process.returncode})"
                )
                logger.info(f"Test error: {test_info.test_path}")

        return test_info

    except Exception as e:
        test_info.status = TestStatus.ERROR
        test_info.error_message = f"Error running test: {str(e)}"
        logger.error(f"Error running test {test_info.test_path}: {str(e)}")
        return test_info


async def execute_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to execute generated tests.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    logger.info(f"Executing tests for language: {language}")

    # Set up test environment
    success, message, env_path = setup_test_environment(language)

    if not success:
        error_msg = f"Failed to set up test environment: {message}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_execution",
                "error": error_msg,
                "type": "environment_setup_failed",
            }
        )
        state.next_phase = "error"
        return state

    logger.info(message)

    # Start timing
    start_time = time.time()

    # Get tests to execute (skip already executed tests)
    tests_to_execute = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status in [TestStatus.PENDING]
    }

    # Skip tests that don't have content
    tests_to_execute = {
        source_file: test_info
        for source_file, test_info in tests_to_execute.items()
        if test_info.content
    }

    logger.info(f"Executing {len(tests_to_execute)} tests")

    # Run tests in batches to avoid resource issues
    batch_size = 5
    files = list(tests_to_execute.keys())

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {len(batch)} files"
        )

        # Process files in parallel
        tasks = []
        for source_file in batch:
            test_info = tests_to_execute[source_file]
            tasks.append(run_test(test_info, language, env_path))

        # Run batch
        results = await asyncio.gather(*tasks)

        # Update state with results
        for test_info in results:
            state.tests[test_info.source_file] = test_info

        # Add a small delay between batches
        if i + batch_size < len(files):
            await asyncio.sleep(1)

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state
    passed = len([t for t in state.tests.values() if t.status == TestStatus.PASSED])
    failed = len([t for t in state.tests.values() if t.status == TestStatus.FAILED])
    error = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    state.successful_tests = passed
    state.failed_tests = failed + error

    logger.info(f"Test execution complete in {time_taken:.2f}s")
    logger.info(
        f"Passed: {passed}, Failed: {failed}, Error: {error}, Skipped: {skipped}"
    )

    # Set next phase
    state.current_phase = "test_execution"

    # Determine next phase based on results
    if failed > 0 or error > 0:
        # Need to fix tests
        state.next_phase = "test_fixing"
    else:
        # All tests passed or skipped, we're done
        state.next_phase = "complete"

    return state

```

# test_agent/workflow/nodes/test_fixing.py

```py
# test_agent/workflow/nodes/test_fixing.py

import os
import logging
import time
import asyncio
import re
from typing import Dict, Any, Optional

from test_agent.llm import get_provider
from test_agent.workflow import WorkflowState, TestInfo, TestStatus
from .test_execution import run_test

# Configure logging
logger = logging.getLogger(__name__)


def analyze_test_error(execution_result: str) -> Dict[str, Any]:
    """
    Analyze a test execution result to determine the type of error.

    Args:
        execution_result: Test execution output

    Returns:
        Dictionary with error analysis
    """
    result = {
        "has_syntax_error": False,
        "has_import_error": False,
        "has_assertion_error": False,
        "has_exception": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "error_line": None,
        "missing_dependencies": [],
    }

    # Skip if no execution result
    if not execution_result:
        return result

    # Check for common error types
    if "SyntaxError" in execution_result:
        result["has_syntax_error"] = True
        result["error_type"] = "SyntaxError"

        # Extract error details
        syntax_match = re.search(r"SyntaxError: (.*?)(?:\n|$)", execution_result)
        if syntax_match:
            result["error_message"] = syntax_match.group(1)

        # Extract line number
        line_match = re.search(r"line (\d+)", execution_result)
        if line_match:
            result["error_line"] = line_match.group(1)

    elif "ImportError" in execution_result or "ModuleNotFoundError" in execution_result:
        result["has_import_error"] = True
        result["error_type"] = (
            "ImportError"
            if "ImportError" in execution_result
            else "ModuleNotFoundError"
        )

        # Extract error details
        import_match = re.search(
            r"(ImportError|ModuleNotFoundError): (.*?)(?:\n|$)", execution_result
        )
        if import_match:
            result["error_message"] = import_match.group(2)

        # Extract missing module
        if "No module named" in execution_result:
            module_match = re.search(r"No module named '(.*?)'", execution_result)
            if module_match:
                module_name = module_match.group(1)
                result["missing_dependencies"].append(module_name)

    elif "AssertionError" in execution_result:
        result["has_assertion_error"] = True
        result["error_type"] = "AssertionError"

        # Extract error details
        assertion_match = re.search(r"AssertionError: (.*?)(?:\n|$)", execution_result)
        if assertion_match:
            result["error_message"] = assertion_match.group(1)

    elif "Exception" in execution_result or "Error" in execution_result:
        result["has_exception"] = True

        # Try to extract error type
        error_match = re.search(
            r"([A-Za-z]+Error|Exception): (.*?)(?:\n|$)", execution_result
        )
        if error_match:
            result["error_type"] = error_match.group(1)
            result["error_message"] = error_match.group(2)

    return result


async def fix_test(
    test_info: TestInfo,
    language: str,
    llm_provider,
    env_path: Optional[str] = None,
    max_attempts: int = 3,
) -> TestInfo:
    """
    Attempt to fix a failing test.

    Args:
        test_info: Test information
        language: Language name
        llm_provider: LLM provider
        env_path: Optional path to the test environment
        max_attempts: Maximum number of fix attempts

    Returns:
        Updated test information
    """
    # Skip if no execution result or not a failed/error status
    if not test_info.execution_result or test_info.status not in [
        TestStatus.FAILED,
        TestStatus.ERROR,
    ]:
        return test_info

    # Skip if already tried too many times
    if test_info.fix_attempts >= max_attempts:
        logger.info(
            f"Maximum fix attempts ({max_attempts}) reached for {test_info.test_path}"
        )
        return test_info

    logger.info(
        f"Attempting to fix test: {test_info.test_path} (attempt {test_info.fix_attempts + 1}/{max_attempts})"
    )

    # Analyze the error
    error_analysis = analyze_test_error(test_info.execution_result)

    try:
        # Read the current test content
        with open(test_info.test_path, "r") as f:
            current_content = f.read()

        # Save to fix history
        if test_info.fix_history is None:
            test_info.fix_history = []

        test_info.fix_history.append(current_content)

        # Prepare the prompt
        prompt = f"""
        I need help fixing a failing test. I'll provide:
        1. The test content
        2. The error output
        3. An analysis of the error
        
        Test file: {os.path.basename(test_info.test_path)}
        Source file: {os.path.basename(test_info.source_file)}
        
        Error output:
        {test_info.execution_result}
        
        Error analysis:
        - Error type: {error_analysis.get('error_type')}
        - Error message: {error_analysis.get('error_message')}
        - Has syntax error: {error_analysis.get('has_syntax_error')}
        - Has import error: {error_analysis.get('has_import_error')}
        - Has assertion error: {error_analysis.get('has_assertion_error')}
        - Has exception: {error_analysis.get('has_exception')}
        
        Current test content:
        \`\`\`
        {current_content}
        \`\`\`
        
        Please fix the test based on the error output. Return ONLY the corrected test code without explanations.
        """

        # Call LLM to fix test
        response = await llm_provider.generate(prompt)

        # Extract code from response
        fixed_code = response

        # Try to extract code block if the response contains explanations
        import re

        code_matches = re.findall(r"\`\`\`(?:python|go)?\n(.*?)\`\`\`", fixed_code, re.DOTALL)
        if code_matches:
            # Use the longest code block (most complete)
            fixed_code = max(code_matches, key=len)

        # Increment fix attempts
        test_info.fix_attempts += 1

        # Check if the content actually changed
        if fixed_code.strip() == current_content.strip():
            logger.warning(
                f"Fix attempt produced identical code for {test_info.test_path}"
            )
            test_info.error_message = "Fix attempt produced identical code"
            return test_info

        # Write the fixed code to the test file
        with open(test_info.test_path, "w") as f:
            f.write(fixed_code)

        # Update the test content
        test_info.content = fixed_code

        # Run the fixed test
        updated_test_info = await run_test(test_info, language, env_path)

        # Keep fix history and incremented fix attempts
        updated_test_info.fix_history = test_info.fix_history
        updated_test_info.fix_attempts = test_info.fix_attempts

        # Check if the fix was successful
        if updated_test_info.status == TestStatus.PASSED:
            logger.info(f"Test fixed successfully: {test_info.test_path}")
            updated_test_info.status = TestStatus.FIXED

        return updated_test_info

    except Exception as e:
        logger.error(f"Error fixing test {test_info.test_path}: {str(e)}")
        test_info.error_message = f"Error fixing test: {str(e)}"
        test_info.fix_attempts += 1
        return test_info


async def fix_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to fix failing tests.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    # Get LLM provider
    if not state.llm or not state.llm.provider:
        error_msg = "No LLM provider specified in state"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_fixing",
                "error": error_msg,
                "type": "llm_provider_not_found",
            }
        )
        state.next_phase = "error"
        return state

    llm_provider = get_provider(state.llm.provider)

    # Get environment path
    from .test_execution import setup_test_environment

    success, message, env_path = setup_test_environment(language)

    if not success:
        error_msg = f"Failed to set up test environment: {message}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_fixing",
                "error": error_msg,
                "type": "environment_setup_failed",
            }
        )
        state.next_phase = "error"
        return state

    logger.info(message)

    # Start timing
    start_time = time.time()

    # Get tests to fix
    tests_to_fix = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status in [TestStatus.FAILED, TestStatus.ERROR]
    }

    logger.info(f"Attempting to fix {len(tests_to_fix)} tests")

    # Fix tests in batches
    batch_size = 3  # Smaller batch size for fixing as it requires more resources
    files = list(tests_to_fix.keys())

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {len(batch)} files"
        )

        # Process files in parallel
        tasks = []
        for source_file in batch:
            test_info = tests_to_fix[source_file]
            tasks.append(fix_test(test_info, language, llm_provider, env_path))

        # Run batch
        results = await asyncio.gather(*tasks)

        # Update state with results
        for test_info in results:
            state.tests[test_info.source_file] = test_info

        # Add a small delay between batches
        if i + batch_size < len(files):
            await asyncio.sleep(2)

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state
    passed = len([t for t in state.tests.values() if t.status == TestStatus.PASSED])
    fixed = len([t for t in state.tests.values() if t.status == TestStatus.FIXED])
    failed = len([t for t in state.tests.values() if t.status == TestStatus.FAILED])
    error = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    state.successful_tests = passed + fixed
    state.failed_tests = failed + error
    state.fixed_tests = fixed

    logger.info(f"Test fixing complete in {time_taken:.2f}s")
    logger.info(
        f"Passed: {passed}, Fixed: {fixed}, Failed: {failed}, Error: {error}, Skipped: {skipped}"
    )

    # Set next phase
    state.current_phase = "test_fixing"
    state.next_phase = "complete"

    return state

```

# test_agent/workflow/nodes/test_generation.py

```py
# test_agent/workflow/nodes/test_generation.py

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple

from test_agent.language import get_adapter
from test_agent.llm import get_provider
from test_agent.memory import CacheManager
from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


async def generate_test_content(
    source_file: str,
    test_info: TestInfo,
    language_adapter,
    llm_provider,
    cache_manager: Optional[CacheManager] = None,
    file_analysis: Optional[Dict[str, Any]] = None,
    test_pattern: Optional[Dict[str, Any]] = None,
) -> Tuple[TestInfo, bool]:
    """
    Generate test content for a source file.

    Args:
        source_file: Path to the source file
        test_info: Test information
        language_adapter: Language adapter for the file
        llm_provider: LLM provider
        cache_manager: Optional cache manager
        file_analysis: Optional file analysis (to avoid re-analysis)
        test_pattern: Optional test pattern

    Returns:
        Tuple of (updated test info, success flag)
    """
    try:
        # Set status to running
        test_info.status = TestStatus.RUNNING

        # Check cache first if available
        if cache_manager:
            cached_template = cache_manager.get_template_cache(source_file)
            if cached_template and "content" in cached_template:
                test_info.content = cached_template["content"]
                test_info.status = TestStatus.PENDING
                return test_info, True

        # Get analysis if not provided
        if not file_analysis:
            file_analysis = language_adapter.analyze_source_file(source_file)

        # Generate test template
        template = language_adapter.generate_test_template(
            source_file, file_analysis, test_pattern
        )

        # Enhance template with LLM
        prompt = f"""
        You are an expert test engineer tasked with creating a comprehensive test for a source file.
        
        I'll provide you with:
        1. The source file path
        2. A basic test template
        3. Analysis of the source file
        
        Your task is to enhance the template to create a complete, working test file. Follow these guidelines:
        - Focus on testing the public interface of functions and classes
        - Use good test design practices (arrange-act-assert pattern)
        - Include edge cases and error scenarios
        - Don't modify imports or basic structure - just fill in the test implementations
        - Keep your tests focused, testing one thing at a time
        - Use appropriate test fixtures/mocks where needed
        
        Source file: {source_file}
        
        Source file analysis: {file_analysis}
        
        Basic test template:
        \`\`\`
        {template}
        \`\`\`
        
        Enhance this template into a complete test file. Return ONLY the complete test code without explanations.
        """

        # Call LLM to enhance template
        response = await llm_provider.generate(prompt)

        # Extract code from response
        code = response

        # Try to extract code block if the response contains explanations
        import re

        code_matches = re.findall(r"\`\`\`(?:python|go)?\n(.*?)\`\`\`", code, re.DOTALL)
        if code_matches:
            # Use the longest code block (most complete)
            code = max(code_matches, key=len)

        # Save test content
        test_info.content = code

        # Update cache if available
        if cache_manager:
            cache_manager.set_template_cache(source_file, {"content": code})

        # Update status
        test_info.status = TestStatus.PENDING

        return test_info, True

    except Exception as e:
        logger.error(f"Error generating test for {source_file}: {str(e)}")
        test_info.error_message = str(e)
        test_info.status = TestStatus.ERROR
        return test_info, False


async def generate_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to generate test contents using LLM.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    # Get language adapter
    language = state.project.language
    language_adapter = get_adapter(language)

    if not language_adapter:
        error_msg = f"No adapter found for language: {language}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_generation",
                "error": error_msg,
                "type": "adapter_not_found",
            }
        )
        state.next_phase = "error"
        return state

    # Get LLM provider
    if not state.llm or not state.llm.provider:
        error_msg = "No LLM provider specified in state"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_generation",
                "error": error_msg,
                "type": "llm_provider_not_found",
            }
        )
        state.next_phase = "error"
        return state

    llm_provider = get_provider(state.llm.provider)

    # Initialize cache manager
    cache_manager = CacheManager(state.project.root_directory)

    # Get test pattern
    test_pattern = state.project.patterns

    # Start timing
    start_time = time.time()

    # Get files to generate tests for
    tests_to_generate = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status == TestStatus.PENDING and not test_info.content
    }

    logger.info(f"Generating tests for {len(tests_to_generate)} files")

    # Generate tests in batches to avoid overloading the LLM API
    batch_size = 5
    files = list(tests_to_generate.keys())

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {len(batch)} files"
        )

        # Process files in parallel
        tasks = []
        for source_file in batch:
            test_info = tests_to_generate[source_file]

            # Get file analysis
            file_analysis = None
            for file_info in state.project.source_files:
                if file_info.path == source_file:
                    file_analysis = file_info.analysis
                    break

            tasks.append(
                generate_test_content(
                    source_file,
                    test_info,
                    language_adapter,
                    llm_provider,
                    cache_manager,
                    file_analysis,
                    test_pattern,
                )
            )

        # Run batch
        results = await asyncio.gather(*tasks)

        # Update state with results
        for test_info, success in results:
            state.tests[test_info.source_file] = test_info

        # Add a small delay between batches to avoid rate limiting
        if i + batch_size < len(files):
            await asyncio.sleep(1)

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state
    successful = len(
        [t for t in state.tests.values() if t.content and t.status != TestStatus.ERROR]
    )
    failed = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    logger.info(f"Test generation complete in {time_taken:.2f}s")
    logger.info(f"Generated {successful} tests successfully")
    logger.info(f"Failed to generate {failed} tests")
    logger.info(f"Skipped {skipped} existing tests")

    # Set next phase
    state.current_phase = "test_generation"
    state.next_phase = "test_execution"

    # Write test files to disk
    for test_info in state.tests.values():
        if test_info.content and test_info.status != TestStatus.ERROR:
            try:
                # Create directory if needed
                test_dir = os.path.dirname(test_info.test_path)
                os.makedirs(test_dir, exist_ok=True)

                # Write test file
                with open(test_info.test_path, "w") as f:
                    f.write(test_info.content)

                logger.debug(f"Wrote test file: {test_info.test_path}")

            except Exception as e:
                logger.error(f"Error writing test file {test_info.test_path}: {str(e)}")
                test_info.error_message = f"Error writing test file: {str(e)}"
                test_info.status = TestStatus.ERROR
                state.tests[test_info.source_file] = test_info

    return state

```

# test_agent/workflow/nodes/test_path.py

```py
# test_agent/workflow/nodes/test_path.py

import os
import logging


from test_agent.language import get_adapter
from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


async def generate_test_paths(state: WorkflowState) -> WorkflowState:
    """
    Node to generate test file paths for source files.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    # Get language adapter
    language = state.project.language
    language_adapter = get_adapter(language)

    if not language_adapter:
        error_msg = f"No adapter found for language: {language}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_path_generation",
                "error": error_msg,
                "type": "adapter_not_found",
            }
        )
        state.next_phase = "error"
        return state

    # Get test directory
    test_directory = state.project.test_directory

    if not test_directory:
        test_directory = os.path.join(state.project.root_directory, "tests")
        state.project.test_directory = test_directory
        # Create test directory if it doesn't exist
        os.makedirs(test_directory, exist_ok=True)

    logger.info(f"Generating test paths using test directory: {test_directory}")

    # Get project root
    # project_root = state.project.root_directory

    # Get project test patterns
    test_pattern = state.project.patterns

    # Process files that need test paths
    files_to_process = [
        f for f in state.project.source_files if not f.skip and not f.has_existing_test
    ]

    logger.info(f"Generating test paths for {len(files_to_process)} files")

    # Process each source file
    for file_info in files_to_process:
        source_file = file_info.path

        try:
            # Generate test path
            test_path = language_adapter.generate_test_path(
                source_file, test_directory, test_pattern
            )

            # Create test directory if needed
            test_dir = os.path.dirname(test_path)
            os.makedirs(test_dir, exist_ok=True)

            # Create test info
            test_info = TestInfo(
                source_file=source_file, test_path=test_path, status=TestStatus.PENDING
            )

            # Add to state
            state.tests[source_file] = test_info

        except Exception as e:
            logger.error(f"Error generating test path for {source_file}: {str(e)}")
            state.warnings.append(
                {
                    "phase": "test_path_generation",
                    "file": source_file,
                    "error": str(e),
                    "type": "test_path_generation_failed",
                }
            )

    # Update state with existing tests
    for file_info in [f for f in state.project.source_files if f.has_existing_test]:
        source_file = file_info.path
        existing_test = file_info.existing_test_path

        if existing_test:
            # Create test info for existing test
            test_info = TestInfo(
                source_file=source_file,
                test_path=existing_test,
                status=TestStatus.SKIPPED,
                content=None,  # We don't need to read the content yet
            )

            # Add to state
            state.tests[source_file] = test_info

    # Log summary
    logger.info(f"Generated {len(state.tests)} test paths")
    logger.info(
        f"Skipped {len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])} existing tests"
    )

    # Set next phase
    state.current_phase = "test_path_generation"
    state.next_phase = "test_generation"

    return state

```

# test_agent/workflow/state.py

```py
# test_agent/workflow/state.py

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Status of a test execution"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    FIXED = "fixed"
    SKIPPED = "skipped"


class FileInfo(BaseModel):
    """Information about a source file"""

    path: str = Field(..., description="Absolute path to the file")
    relative_path: Optional[str] = Field(
        None, description="Path relative to project root"
    )
    language: Optional[str] = Field(None, description="Programming language")
    size: Optional[int] = Field(None, description="File size in bytes")
    last_modified: Optional[float] = Field(None, description="Last modified timestamp")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    has_existing_test: bool = Field(
        False, description="Whether the file already has a test"
    )
    existing_test_path: Optional[str] = Field(
        None, description="Path to existing test file"
    )
    skip: bool = Field(False, description="Whether to skip this file for testing")
    skip_reason: Optional[str] = Field(None, description="Reason for skipping")


class TestInfo(BaseModel):
    """Information about a generated test"""

    source_file: str = Field(..., description="Path to the source file")
    test_path: str = Field(
        ..., description="Path where the test file is/will be created"
    )
    content: Optional[str] = Field(None, description="Content of the test file")
    status: TestStatus = Field(
        default=TestStatus.PENDING, description="Status of the test"
    )
    execution_result: Optional[str] = Field(
        None, description="Result of test execution"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if test failed"
    )
    fix_attempts: int = Field(default=0, description="Number of fix attempts made")
    fix_history: List[str] = Field(
        default_factory=list, description="History of fixes attempted"
    )


class TestPattern(BaseModel):
    """Test pattern information for a project"""

    location_pattern: str = Field(
        "tests_directory", description="Where tests are located"
    )
    naming_convention: str = Field("test_prefix", description="How tests are named")
    test_directories: List[str] = Field(
        default_factory=list, description="Test directories found"
    )
    primary_test_dir: Optional[str] = Field(None, description="Primary test directory")
    existing_tests: List[str] = Field(
        default_factory=list, description="Existing test files"
    )
    framework: Optional[str] = Field(
        None, description="Test framework (e.g., pytest, unittest)"
    )


class ProjectInfo(BaseModel):
    """Information about the project being tested"""

    root_directory: str = Field(..., description="Project root directory")
    language: Optional[str] = Field(None, description="Primary programming language")
    test_directory: Optional[str] = Field(
        None, description="Directory to write tests to"
    )
    patterns: Optional[TestPattern] = Field(None, description="Detected test patterns")
    source_files: List[FileInfo] = Field(
        default_factory=list, description="Source files found"
    )
    excluded_directories: List[str] = Field(
        default_factory=list, description="Directories to exclude"
    )
    excluded_files: List[str] = Field(
        default_factory=list, description="Files to exclude"
    )


class CacheInfo(BaseModel):
    """Information about cache status"""

    cache_enabled: bool = Field(True, description="Whether caching is enabled")
    cache_directory: Optional[str] = Field(
        None, description="Directory where cache is stored"
    )
    cached_files: int = Field(0, description="Number of files in cache")
    last_cache_update: Optional[float] = Field(
        None, description="Timestamp of last cache update"
    )


class ConversationTurn(BaseModel):
    """A single turn in the conversation history"""

    role: str = Field(..., description="Role (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: float = Field(..., description="Timestamp when the message was created")


class MemoryInfo(BaseModel):
    """Memory and persistence information"""

    conversation_history: List[ConversationTurn] = Field(
        default_factory=list, description="Conversation history"
    )
    cache_info: CacheInfo = Field(
        default_factory=CacheInfo, description="Cache information"
    )
    decisions: Dict[str, Any] = Field(
        default_factory=dict, description="Key decisions made"
    )


class LLMInfo(BaseModel):
    """Information about the LLM being used"""

    provider: str = Field(..., description="LLM provider name")
    model: Optional[str] = Field(None, description="Specific model being used")
    api_key: Optional[str] = Field(None, description="API key (obfuscated for logs)")
    streaming: bool = Field(True, description="Whether to use streaming mode")
    temperature: float = Field(0.2, description="Temperature setting")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens setting")
    other_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Other provider-specific settings"
    )


class WorkflowState(BaseModel):
    """Complete state of the test generation workflow"""

    project: ProjectInfo = Field(..., description="Project information")
    tests: Dict[str, TestInfo] = Field(
        default_factory=dict, description="Test information keyed by source file path"
    )
    memory: MemoryInfo = Field(
        default_factory=MemoryInfo, description="Memory information"
    )
    llm: Optional[LLMInfo] = Field(None, description="LLM information")

    # Workflow control
    current_phase: str = Field(
        "initialize", description="Current phase in the workflow"
    )
    next_phase: Optional[str] = Field(None, description="Next phase to execute")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered"
    )
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Warnings encountered"
    )
    processed_files: Set[str] = Field(
        default_factory=set, description="Files that have been processed"
    )
    current_file: Optional[str] = Field(
        None, description="Current file being processed"
    )

    # Summary information
    start_time: Optional[float] = Field(None, description="Start time of the workflow")
    end_time: Optional[float] = Field(None, description="End time of the workflow")
    successful_tests: int = Field(0, description="Number of successful tests")
    failed_tests: int = Field(0, description="Number of failed tests")
    skipped_tests: int = Field(0, description="Number of skipped tests")
    fixed_tests: int = Field(0, description="Number of fixed tests")
    total_files: int = Field(0, description="Total number of files to process")
    is_completed: bool = Field(False, description="Whether the workflow has completed")

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True
        str_strip_whitespace = True

```

