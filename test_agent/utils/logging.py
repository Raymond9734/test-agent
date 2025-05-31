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
