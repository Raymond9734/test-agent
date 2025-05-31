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
