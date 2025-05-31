# test_agent/memory/conversation.py

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history and provides persistence.

    This class handles saving and loading conversation turns to/from
    a file, with rotation to prevent files from growing too large.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_turns: int = 100,
        filename: str = "conversation_history.json",
    ):
        """
        Initialize the conversation memory.

        Args:
            storage_dir: Directory to store the conversation history.
                         If None, uses a temp directory.
            max_turns: Maximum number of turns to keep in memory
            filename: Name of the conversation history file
        """
        # Set up storage directory
        if storage_dir is None:
            import tempfile

            storage_dir = os.path.join(tempfile.gettempdir(), "test_agent_memory")

        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.history_path = os.path.join(storage_dir, filename)
        self.max_turns = max_turns
        self.turns = []

        # Load existing history if available
        self._load_history()

    def _load_history(self) -> None:
        """Load conversation history from disk"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self.turns = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.turns)} conversation turns from {self.history_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading conversation history: {e}")
                # Backup the corrupted file
                if os.path.exists(self.history_path):
                    backup_path = f"{self.history_path}.bak.{int(time.time())}"
                    try:
                        os.rename(self.history_path, backup_path)
                        logger.info(
                            f"Backed up corrupted history file to {backup_path}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to backup history file: {e}")

                # Start with empty history
                self.turns = []

    def _save_history(self) -> None:
        """Save conversation history to disk"""
        try:
            # Ensure directory exists
            os.makedirs(self.storage_dir, exist_ok=True)

            with open(self.history_path, "w") as f:
                json.dump(self.turns, f, indent=2)
                logger.debug(
                    f"Saved {len(self.turns)} conversation turns to {self.history_path}"
                )
        except IOError as e:
            logger.error(f"Error saving conversation history: {e}")

    def add_turn(self, role: str, content: str) -> None:
        """
        Add a turn to the conversation history.

        Args:
            role: Role of the speaker ("system", "user", "assistant")
            content: Content of the message
        """
        turn = {"role": role, "content": content, "timestamp": time.time()}

        self.turns.append(turn)

        # Rotate if necessary
        if len(self.turns) > self.max_turns:
            # Keep last max_turns turns
            self.turns = self.turns[-self.max_turns :]

        # Save to disk
        self._save_history()

    def get_history(self, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.

        Args:
            max_turns: Maximum number of recent turns to return.
                       If None, returns all turns up to max_turns.

        Returns:
            List of conversation turns
        """
        if max_turns is None:
            return self.turns

        return self.turns[-max_turns:]

    def clear(self) -> None:
        """Clear the conversation history"""
        self.turns = []

        # Remove the history file if it exists
        if os.path.exists(self.history_path):
            try:
                os.remove(self.history_path)
                logger.info(f"Removed conversation history file: {self.history_path}")
            except Exception as e:
                logger.error(f"Failed to remove history file: {e}")

    def format_for_context(self, max_turns: Optional[int] = None) -> str:
        """
        Format the conversation history for inclusion in an LLM prompt.

        Args:
            max_turns: Maximum number of recent turns to include

        Returns:
            Formatted conversation history
        """
        history = self.get_history(max_turns)

        if not history:
            return ""

        formatted = []
        for turn in history:
            role = turn["role"]
            content = turn["content"]

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")

        return "\n\n".join(formatted)

    def get_turns_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get conversation turns for a specific role.

        Args:
            role: Role to filter by ("system", "user", "assistant")

        Returns:
            List of turns with the specified role
        """
        return [turn for turn in self.turns if turn["role"] == role]


class MemoryManager:
    """
    Central manager for different types of memory in the test agent.
    """

    def __init__(self, project_dir: str):
        """
        Initialize the memory manager.

        Args:
            project_dir: Project directory path
        """
        # Create a hash of the project path to use as part of the storage directory
        self.project_dir = project_dir
        self.project_hash = self._hash_project_path(project_dir)

        # Set up storage directory in temporary location
        import tempfile

        storage_base = os.path.join(tempfile.gettempdir(), "test_agent_memory")
        self.storage_dir = os.path.join(storage_base, self.project_hash)
        os.makedirs(self.storage_dir, exist_ok=True)

        # Initialize different memory types
        self.conversation = ConversationMemory(
            storage_dir=self.storage_dir, filename="conversation_history.json"
        )

        # Store decisions (important choices made during test generation)
        self.decisions = {}
        self.decisions_path = os.path.join(self.storage_dir, "decisions.json")
        self._load_decisions()

    def _hash_project_path(self, project_path: str) -> str:
        """
        Create a hash of the project path to use as a unique identifier

        Args:
            project_path: Path to the project directory

        Returns:
            MD5 hash of the normalized project path
        """
        import hashlib

        # Normalize path to handle different OS path separators
        normalized_path = os.path.normpath(os.path.abspath(project_path))
        return hashlib.md5(normalized_path.encode()).hexdigest()[:8]

    def _load_decisions(self) -> None:
        """Load decisions from disk"""
        if os.path.exists(self.decisions_path):
            try:
                with open(self.decisions_path, "r") as f:
                    self.decisions = json.load(f)
                    logger.debug(
                        f"Loaded {len(self.decisions)} decisions from {self.decisions_path}"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading decisions: {e}")
                self.decisions = {}

    def _save_decisions(self) -> None:
        """Save decisions to disk"""
        try:
            with open(self.decisions_path, "w") as f:
                json.dump(self.decisions, f, indent=2)
                logger.debug(
                    f"Saved {len(self.decisions)} decisions to {self.decisions_path}"
                )
        except IOError as e:
            logger.error(f"Error saving decisions: {e}")

    def record_decision(self, key: str, value: Any) -> None:
        """
        Record an important decision.

        Args:
            key: Decision identifier
            value: Decision value
        """
        self.decisions[key] = {"value": value, "timestamp": time.time()}
        self._save_decisions()

    def get_decision(self, key: str) -> Optional[Any]:
        """
        Get a recorded decision.

        Args:
            key: Decision identifier

        Returns:
            Decision value or None if not found
        """
        if key in self.decisions:
            return self.decisions[key]["value"]
        return None

    def clear_decisions(self) -> None:
        """Clear all recorded decisions"""
        self.decisions = {}

        # Remove the decisions file if it exists
        if os.path.exists(self.decisions_path):
            try:
                os.remove(self.decisions_path)
                logger.info(f"Removed decisions file: {self.decisions_path}")
            except Exception as e:
                logger.error(f"Failed to remove decisions file: {e}")

    def clear_all(self) -> None:
        """Clear all memory"""
        self.clear_decisions()
        self.conversation.clear()

        # Optionally remove the entire storage directory
        try:
            import shutil

            shutil.rmtree(self.storage_dir)
            logger.info(f"Removed memory storage directory: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to remove memory storage directory: {e}")
