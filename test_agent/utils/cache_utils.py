# test_agent/utils/cache_utils.py - New file for unified cache operations

"""
Unified cache clearing utilities to avoid inconsistencies between
different parts of the codebase.
"""

import os
import shutil
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def clear_project_cache(
    project_dir: Optional[str] = None, force_remove: bool = True
) -> Dict[str, int]:
    """
    Clear cache for a specific project or all projects.

    Args:
        project_dir: Project directory (if None, clears all projects)
        force_remove: Whether to force remove cache directory after clearing

    Returns:
        Dictionary with count of entries cleared by cache type
    """
    from ..memory import CacheManager

    # Normalize project directory path if provided
    if project_dir:
        project_dir = os.path.abspath(project_dir)

    logger.info(f"Clearing cache for project: {project_dir or 'ALL'}")

    # Create cache manager
    cache_manager = CacheManager(project_dir or "")

    # Get current cache status
    stats = cache_manager.get_statistics()
    logger.debug(f"Cache stats before clearing: {stats}")

    # Clear the cache
    result = cache_manager.clear_cache(None)  # None means clear all types

    # Force remove cache directory if requested
    if force_remove and os.path.exists(cache_manager.cache_dir):
        try:
            shutil.rmtree(cache_manager.cache_dir)
            logger.info(f"Removed cache directory: {cache_manager.cache_dir}")

            # Recreate empty cache directory
            os.makedirs(cache_manager.cache_dir, exist_ok=True)
            logger.debug(f"Recreated empty cache directory: {cache_manager.cache_dir}")

        except Exception as e:
            logger.warning(f"Failed to remove cache directory: {e}")

    return result


def clear_all_caches() -> Dict[str, int]:
    """
    Clear all caches and configuration.

    Returns:
        Dictionary with total count of entries cleared
    """
    import tempfile

    # Clear main cache directory
    cache_base_dir = os.path.join(tempfile.gettempdir(), "test_agent_cache")
    total_cleared = {"hashes": 0, "analysis": 0, "template": 0}

    if os.path.exists(cache_base_dir):
        try:
            # Get list of all project cache directories
            project_dirs = [
                d
                for d in os.listdir(cache_base_dir)
                if os.path.isdir(os.path.join(cache_base_dir, d))
            ]

            logger.info(f"Clearing caches for {len(project_dirs)} projects")

            # Clear each project's cache
            for project_hash in project_dirs:
                project_cache_dir = os.path.join(cache_base_dir, project_hash)
                try:
                    # Try to load and count entries before removing
                    from ..memory import CacheManager

                    # Create a temporary cache manager to get counts
                    temp_manager = CacheManager("")
                    temp_manager.cache_dir = project_cache_dir
                    temp_manager.hashes_path = os.path.join(
                        project_cache_dir, "file_hashes.json"
                    )
                    temp_manager.analysis_path = os.path.join(
                        project_cache_dir, "analysis_cache.json"
                    )
                    temp_manager.template_path = os.path.join(
                        project_cache_dir, "template_cache.json"
                    )

                    # Load and count
                    temp_manager._load_cache()
                    total_cleared["hashes"] += len(temp_manager.file_hashes)
                    total_cleared["analysis"] += len(temp_manager.analysis_cache)
                    total_cleared["template"] += len(temp_manager.template_cache)

                except Exception as e:
                    logger.debug(f"Could not count entries in {project_cache_dir}: {e}")

            # Remove entire cache base directory
            shutil.rmtree(cache_base_dir)
            logger.info(f"Removed entire cache directory: {cache_base_dir}")

        except Exception as e:
            logger.error(f"Failed to clear all caches: {e}")

    # Clear configuration
    config_file = Path.home() / ".test_agent" / "config.json"
    if config_file.exists():
        try:
            config_file.unlink()
            logger.info("Cleared configuration file")
        except Exception as e:
            logger.warning(f"Failed to clear configuration: {e}")

    return total_cleared
