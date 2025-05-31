# test_agent/tools/__init__.py

from .environment_tools import (
    setup_environment,
    install_dependencies,
    cleanup_environment,
)
from .file_tools import find_files, read_file_content, write_file, create_directory
from .test_tools import run_test_command, parse_test_results, check_test_coverage

# Import new registry system
try:
    from .registry import (
        tool_registry,
        register_default_tools,
        Tool,
        InstallPythonPackageTool,
        FixImportStatementTool,
        CreateMockDependencyTool,
        RunTestCommandTool,
    )

    # Registry available
    REGISTRY_AVAILABLE = True

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
        # New registry exports
        "tool_registry",
        "register_default_tools",
        "Tool",
        "InstallPythonPackageTool",
        "FixImportStatementTool",
        "CreateMockDependencyTool",
        "RunTestCommandTool",
        "REGISTRY_AVAILABLE",
    ]

except ImportError:
    # Registry not available yet - this is OK during development
    REGISTRY_AVAILABLE = False

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
        "REGISTRY_AVAILABLE",
    ]
