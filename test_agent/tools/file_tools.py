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
