# Test Agent

An intelligent, LLM-powered test generation agent that automatically creates, executes, and fixes tests for your software projects.

## Features

🤖 **AI-Powered Test Generation**: Uses advanced language models to generate comprehensive test suites
🔧 **Automatic Test Fixing**: Intelligently fixes failing tests using built-in tools
🌐 **Multi-Language Support**: Currently supports Python and Go projects
🔄 **Multiple LLM Providers**: Works with Claude, OpenAI, DeepSeek, and Gemini
⚡ **Smart Caching**: Efficient caching system to avoid redundant processing
🎯 **Project Structure Detection**: Automatically detects and adapts to your project's testing patterns
🛠️ **Tool Integration**: Built-in tools for installing dependencies, fixing imports, and creating mocks

## Installation

### From Source

```bash
git clone <repository-url>
cd test-agent
pip install -e .
```

### Using pip (when published)(not yet)

```bash
pip install test-agent
```

## Quick Start

1. **Generate tests for a Python project**:

   ```bash
   test-agent /path/to/your/project --provider claude
   ```

2. **Generate tests for a Go project**:

   ```bash
   test-agent /path/to/your/go/project --language go --provider openai
   ```

3. **Interactive setup** (will prompt for provider and API key):
   ```bash
   test-agent /path/to/your/project
   ```

## Usage

### Basic Usage

```bash
test-agent <project_directory> [options]
```

### Command Line Options

#### Core Options

- `--language, -l`: Programming language (auto-detected if not specified)
- `--provider, -p`: LLM provider (claude, openai, deepseek, gemini)
- `--model, -m`: Specific model to use (optional)
- `--test-dir, -t`: Custom test directory (optional)
- `--api-key, -k`: API key for the LLM provider

#### Filtering Options

- `--exclude-dir, -e`: Directory to exclude (can be used multiple times)
- `--exclude-file, -x`: File to exclude (can be used multiple times)
- `--files, -f`: Specific files to process

#### Cache Options

- `--no-cache`: Disable caching
- `--clear-cache`: Clear cache before running
- `--clear-all`: Clear all caches and settings

#### Output Options

- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Minimize output
- `--log-file`: Path to save log file
- `--log-level`: Logging level (debug, info, warning, error)

#### Utility Commands

- `--list-languages`: List supported languages
- `--list-providers`: List supported LLM providers
- `--save-key`: Save API key for a provider

### Examples

```bash
# Generate tests with specific provider
test-agent ./my-project --provider claude --verbose

# Skip certain directories
test-agent ./my-project --exclude-dir venv --exclude-dir .git

# Use custom test directory
test-agent ./my-project --test-dir ./custom-tests

# Clear cache and regenerate
test-agent ./my-project --clear-cache --provider openai

# List available providers
test-agent --list-providers

# Save API key for future use
test-agent --save-key --provider claude
```

## Supported Languages

- **Python**: Full support with pytest and unittest frameworks
- **Go**: Full support with standard Go testing

## Supported LLM Providers

| Provider     | Models                                           | API Key Required    |
| ------------ | ------------------------------------------------ | ------------------- |
| **Claude**   | claude-3-5-sonnet, claude-3-opus, claude-3-haiku | `ANTHROPIC_API_KEY` |
| **OpenAI**   | gpt-4o, gpt-4-turbo, gpt-3.5-turbo               | `OPENAI_API_KEY`    |
| **DeepSeek** | deepseek-chat, deepseek-coder                    | `DEEPSEEK_API_KEY`  |
| **Gemini**   | gemini-1.5-pro, gemini-1.5-flash                 | `GOOGLE_API_KEY`    |

## Configuration

### API Keys

API keys can be provided in several ways:

1. **Environment variables**:

   ```bash
   export ANTHROPIC_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"
   export DEEPSEEK_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   ```

2. **Command line**:

   ```bash
   test-agent ./project --api-key your-key
   ```

3. **Saved configuration**:
   ```bash
   test-agent --save-key --provider claude
   ```

### Configuration File

Settings are stored in `~/.test_agent/config.json`:

```json
{
  "api_keys": {
    "claude": "your-key",
    "openai": "your-key"
  },
  "last_provider": "claude"
}
```

## How It Works

The test agent follows a comprehensive workflow:

1. **Project Analysis**: Detects language, analyzes project structure, and identifies testing patterns
2. **File Discovery**: Finds source files and determines which need tests
3. **Test Generation**: Uses LLMs to generate comprehensive test suites
4. **Test Execution**: Runs generated tests to verify they work
5. **Intelligent Fixing**: Automatically fixes failing tests using built-in tools
6. **Validation**: Ensures all tests pass before completion

### Built-in Tools

The agent includes several tools for fixing tests:

- **Package Installation**: Automatically installs missing Python packages
- **Import Fixing**: Analyzes and fixes import statements
- **Mock Creation**: Creates mocks for unavailable dependencies
- **Test Execution**: Runs tests and captures detailed output

## Project Structure

```
test_agent/
├── cli.py              # Command-line interface
├── main.py             # Main entry point and API
├── config.py           # Configuration management
├── language/           # Language-specific adapters
│   ├── python/         # Python language support
│   └── go/             # Go language support
├── llm/                # LLM provider integrations
│   ├── claude.py       # Claude/Anthropic integration
│   ├── openai.py       # OpenAI integration
│   ├── deepseek.py     # DeepSeek integration
│   └── gemini.py       # Google Gemini integration
├── workflow/           # Workflow orchestration
│   ├── graph.py        # LangGraph workflow definition
│   ├── state.py        # Workflow state management
│   └── nodes/          # Individual workflow nodes
├── tools/              # Built-in tools for test fixing
├── memory/             # Caching and persistence
└── utils/              # Utility functions
```

## Python API

You can also use the test agent programmatically:

```python
from test_agent import generate_tests

# Generate tests for a project
result = generate_tests(
    project_directory="/path/to/project",
    language="python",
    llm_provider="claude",
    api_key="your-api-key",
    verbose=True
)

print(f"Generated {result['tests_generated']} tests")
print(f"Success rate: {result['tests_passed']}/{result['tests_generated']}")
```

### Advanced Usage

```python
from test_agent import TestAgent

# Create agent instance
agent = TestAgent(
    project_directory="/path/to/project",
    llm_provider="claude",
    excluded_dirs=["venv", ".git"],
    cache_enabled=True
)

# Run test generation
result = agent.run_sync()
```

## Performance

- **Caching**: Intelligent caching system avoids re-analyzing unchanged files
- **Parallel Processing**: Concurrent analysis and test execution where possible
- **Smart Batching**: Processes files in batches to optimize LLM API usage
- **Incremental Updates**: Only processes changed files on subsequent runs

## Output

The agent provides comprehensive output including:

- **Summary Statistics**: Number of tests generated, passed, failed, and fixed
- **Individual Test Results**: Status of each generated test file
- **Error Analysis**: Detailed error information for debugging
- **Tool Usage**: Summary of tools used during test fixing
- **Performance Metrics**: Execution time and caching statistics

Example output:

```
=== Test Generation Complete ===
Status: success
Source files analyzed: 25
Tests generated: 23
Tests passed: 20
Tests failed: 2
Tests fixed: 1
Time taken: 45.32 seconds

Generated test files:
✅ tests/test_api_utils.py
✅ tests/test_database.py
✅ tests/test_models.py
❌ tests/test_complex_logic.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

- Python 3.8+
- Required packages are automatically installed via `requirements.txt`
- API key for at least one supported LLM provider
