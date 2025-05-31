# test_agent/utils/security.py


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
