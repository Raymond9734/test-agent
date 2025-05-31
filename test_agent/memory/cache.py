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
