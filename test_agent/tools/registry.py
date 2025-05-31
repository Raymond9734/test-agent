# test_agent/tools/registry.py

import logging
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema (JSON Schema format)"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class InstallPythonPackageTool(Tool):
    """Tool to install Python packages"""

    def __init__(self, env_path: Optional[str] = None):
        self.env_path = env_path or os.path.join(
            tempfile.gettempdir(), "test_agent_venv"
        )

    @property
    def name(self) -> str:
        return "install_python_package"

    @property
    def description(self) -> str:
        return "Install a Python package using pip in the test environment"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "Name of the Python package to install",
                },
                "version": {
                    "type": "string",
                    "description": "Optional version specification (e.g., '>=1.0.0')",
                },
            },
            "required": ["package_name"],
        }

    async def execute(self, package_name: str, version: str = "", **kwargs) -> str:
        """Install a Python package"""
        try:
            # Format package with version if specified
            package_spec = f"{package_name}{version}" if version else package_name

            # Get pip path based on OS
            if os.name == "nt":  # Windows
                pip_path = os.path.join(self.env_path, "Scripts", "pip")
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(self.env_path, "bin", "pip")

            # Install the package
            result = subprocess.run(
                [pip_path, "install", package_spec],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                logger.info(f"Successfully installed {package_spec}")
                return f"Successfully installed {package_spec}"
            else:
                error_msg = f"Failed to install {package_spec}: {result.stderr}"
                logger.error(error_msg)
                return error_msg

        except Exception as e:
            error_msg = f"Error installing {package_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class FixImportStatementTool(Tool):
    """Tool to suggest import statement fixes"""

    @property
    def name(self) -> str:
        return "fix_import_statement"

    @property
    def description(self) -> str:
        return "Analyze and suggest fixes for failing import statements"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "failing_import": {
                    "type": "string",
                    "description": "The import statement that is failing",
                },
                "error_message": {
                    "type": "string",
                    "description": "The error message from the failed import",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the file with the failing import",
                },
            },
            "required": ["failing_import", "error_message"],
        }

    async def execute(
        self, failing_import: str, error_message: str, file_path: str = "", **kwargs
    ) -> str:
        """Analyze import error and suggest fixes"""
        try:
            suggestions = []

            # Common import error patterns and fixes
            if "No module named" in error_message:
                module_match = (
                    error_message.split("'")[1] if "'" in error_message else ""
                )
                suggestions.append(f"Module '{module_match}' not found. Consider:")
                suggestions.append(
                    f"1. Installing the package: pip install {module_match}"
                )
                suggestions.append(
                    f"2. Using relative import: from .{module_match} import ..."
                )
                suggestions.append("3. Adding the module to PYTHONPATH")

            elif "cannot import name" in error_message:
                suggestions.append("Import name not found. Consider:")
                suggestions.append("1. Checking if the name exists in the module")
                suggestions.append(
                    "2. Using 'import module' instead of 'from module import name'"
                )
                suggestions.append("3. Checking for typos in the import name")

            elif "relative import" in error_message:
                suggestions.append("Relative import issue. Consider:")
                suggestions.append("1. Using absolute imports instead")
                suggestions.append("2. Running as a module with -m flag")
                suggestions.append("3. Adding __init__.py files to make it a package")

            # Add general suggestions
            suggestions.append("\nGeneral fixes:")
            suggestions.append("- Check if the module is installed")
            suggestions.append("- Verify the module name spelling")
            suggestions.append("- Check Python path configuration")

            result = f"Import fix suggestions for '{failing_import}':\n" + "\n".join(
                suggestions
            )
            logger.info(f"Generated import fix suggestions for {failing_import}")
            return result

        except Exception as e:
            error_msg = f"Error analyzing import: {str(e)}"
            logger.error(error_msg)
            return error_msg


class CreateMockDependencyTool(Tool):
    """Tool to create mock dependencies for missing modules"""

    @property
    def name(self) -> str:
        return "create_mock_dependency"

    @property
    def description(self) -> str:
        return "Create a mock implementation for a missing external dependency"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "description": "Name of the module to mock",
                },
                "functions_to_mock": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of functions/classes to mock",
                },
                "mock_type": {
                    "type": "string",
                    "enum": ["unittest_mock", "pytest_mock", "simple_mock"],
                    "description": "Type of mock to create",
                },
            },
            "required": ["module_name"],
        }

    async def execute(
        self,
        module_name: str,
        functions_to_mock: List[str] = None,
        mock_type: str = "unittest_mock",
        **kwargs,
    ) -> str:
        """Create mock dependency code"""
        try:
            functions_to_mock = functions_to_mock or []

            if mock_type == "unittest_mock":
                mock_code = f"""
# Mock for {module_name}
from unittest.mock import Mock, MagicMock

# Create mock module
{module_name.replace('.', '_')}_mock = MagicMock()

"""
                for func in functions_to_mock:
                    mock_code += (
                        f"{module_name.replace('.', '_')}_mock.{func} = Mock()\n"
                    )

            elif mock_type == "pytest_mock":
                mock_code = f"""
# Pytest mock for {module_name}
import pytest

@pytest.fixture
def mock_{module_name.replace('.', '_')}(mocker):
    mock_module = mocker.MagicMock()
"""
                for func in functions_to_mock:
                    mock_code += f"    mock_module.{func} = mocker.Mock()\n"
                mock_code += "    return mock_module\n"

            else:  # simple_mock
                mock_code = f"""
# Simple mock for {module_name}
class Mock{module_name.replace('.', '').title()}:
    def __init__(self):
        pass
"""
                for func in functions_to_mock:
                    mock_code += f"""    
    def {func}(self, *args, **kwargs):
        return None
"""

            logger.info(f"Created mock for {module_name}")
            return f"Mock code for {module_name}:\n\n{mock_code}"

        except Exception as e:
            error_msg = f"Error creating mock for {module_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class RunTestCommandTool(Tool):
    """Tool to run test commands and capture output"""

    @property
    def name(self) -> str:
        return "run_test_command"

    @property
    def description(self) -> str:
        return "Run a test command and capture its output for analysis"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Test command to run as array of strings",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory to run the command in",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: List[str],
        working_directory: str = ".",
        timeout: int = 30,
        **kwargs,
    ) -> str:
        """Run test command and return output"""
        try:
            result = subprocess.run(
                command,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = f"Command: {' '.join(command)}\n"
            output += f"Return code: {result.returncode}\n"
            output += f"STDOUT:\n{result.stdout}\n"
            output += f"STDERR:\n{result.stderr}\n"

            logger.info(f"Executed test command: {' '.join(command)}")
            return output

        except subprocess.TimeoutExpired:
            error_msg = (
                f"Command timed out after {timeout} seconds: {' '.join(command)}"
            )
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error running command {' '.join(command)}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all registered tools"""
        return self._tools.copy()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for LLM"""
        return [tool.to_dict() for tool in self._tools.values()]

    async def execute_tool(self, name: str, **kwargs) -> str:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if not tool:
            return f"Tool '{name}' not found"

        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            error_msg = f"Error executing tool '{name}': {str(e)}"
            logger.error(error_msg)
            return error_msg


# Global tool registry
tool_registry = ToolRegistry()


# Register default tools
def register_default_tools(env_path: Optional[str] = None):
    """Register default tools"""
    tool_registry.register_tool(InstallPythonPackageTool(env_path))
    tool_registry.register_tool(FixImportStatementTool())
    tool_registry.register_tool(CreateMockDependencyTool())
    tool_registry.register_tool(RunTestCommandTool())
