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
