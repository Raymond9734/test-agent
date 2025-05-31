# test_agent/workflow/state.py

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Status of a test execution"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    FIXED = "fixed"
    SKIPPED = "skipped"


class FileInfo(BaseModel):
    """Information about a source file"""

    path: str = Field(..., description="Absolute path to the file")
    relative_path: Optional[str] = Field(
        None, description="Path relative to project root"
    )
    language: Optional[str] = Field(None, description="Programming language")
    size: Optional[int] = Field(None, description="File size in bytes")
    last_modified: Optional[float] = Field(None, description="Last modified timestamp")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    has_existing_test: bool = Field(
        False, description="Whether the file already has a test"
    )
    existing_test_path: Optional[str] = Field(
        None, description="Path to existing test file"
    )
    skip: bool = Field(False, description="Whether to skip this file for testing")
    skip_reason: Optional[str] = Field(None, description="Reason for skipping")


class TestInfo(BaseModel):
    """Information about a generated test"""

    source_file: str = Field(..., description="Path to the source file")
    test_path: str = Field(
        ..., description="Path where the test file is/will be created"
    )
    content: Optional[str] = Field(None, description="Content of the test file")
    status: TestStatus = Field(
        default=TestStatus.PENDING, description="Status of the test"
    )
    execution_result: Optional[str] = Field(
        None, description="Result of test execution"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if test failed"
    )
    fix_attempts: int = Field(default=0, description="Number of fix attempts made")
    fix_history: List[str] = Field(
        default_factory=list, description="History of fixes attempted"
    )


class TestPattern(BaseModel):
    """Test pattern information for a project"""

    location_pattern: str = Field(
        "tests_directory", description="Where tests are located"
    )
    naming_convention: str = Field("test_prefix", description="How tests are named")
    test_directories: List[str] = Field(
        default_factory=list, description="Test directories found"
    )
    primary_test_dir: Optional[str] = Field(None, description="Primary test directory")
    existing_tests: List[str] = Field(
        default_factory=list, description="Existing test files"
    )
    framework: Optional[str] = Field(
        None, description="Test framework (e.g., pytest, unittest)"
    )


class ProjectInfo(BaseModel):
    """Information about the project being tested"""

    root_directory: str = Field(..., description="Project root directory")
    language: Optional[str] = Field(None, description="Primary programming language")
    test_directory: Optional[str] = Field(
        None, description="Directory to write tests to"
    )
    patterns: Optional[TestPattern] = Field(None, description="Detected test patterns")
    source_files: List[FileInfo] = Field(
        default_factory=list, description="Source files found"
    )
    excluded_directories: List[str] = Field(
        default_factory=list, description="Directories to exclude"
    )
    excluded_files: List[str] = Field(
        default_factory=list, description="Files to exclude"
    )


class CacheInfo(BaseModel):
    """Information about cache status"""

    cache_enabled: bool = Field(True, description="Whether caching is enabled")
    cache_directory: Optional[str] = Field(
        None, description="Directory where cache is stored"
    )
    cached_files: int = Field(0, description="Number of files in cache")
    last_cache_update: Optional[float] = Field(
        None, description="Timestamp of last cache update"
    )


class ConversationTurn(BaseModel):
    """A single turn in the conversation history"""

    role: str = Field(..., description="Role (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: float = Field(..., description="Timestamp when the message was created")


class MemoryInfo(BaseModel):
    """Memory and persistence information"""

    conversation_history: List[ConversationTurn] = Field(
        default_factory=list, description="Conversation history"
    )
    cache_info: CacheInfo = Field(
        default_factory=CacheInfo, description="Cache information"
    )
    decisions: Dict[str, Any] = Field(
        default_factory=dict, description="Key decisions made"
    )


class LLMInfo(BaseModel):
    """Information about the LLM being used"""

    provider: str = Field(..., description="LLM provider name")
    model: Optional[str] = Field(None, description="Specific model being used")
    api_key: Optional[str] = Field(None, description="API key (obfuscated for logs)")
    streaming: bool = Field(True, description="Whether to use streaming mode")
    temperature: float = Field(0.2, description="Temperature setting")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens setting")
    other_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Other provider-specific settings"
    )


class WorkflowState(BaseModel):
    """Complete state of the test generation workflow"""

    project: ProjectInfo = Field(..., description="Project information")
    tests: Dict[str, TestInfo] = Field(
        default_factory=dict, description="Test information keyed by source file path"
    )
    memory: MemoryInfo = Field(
        default_factory=MemoryInfo, description="Memory information"
    )
    llm: Optional[LLMInfo] = Field(None, description="LLM information")

    # Workflow control
    current_phase: str = Field(
        "initialize", description="Current phase in the workflow"
    )
    next_phase: Optional[str] = Field(None, description="Next phase to execute")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered"
    )
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Warnings encountered"
    )
    processed_files: Set[str] = Field(
        default_factory=set, description="Files that have been processed"
    )
    current_file: Optional[str] = Field(
        None, description="Current file being processed"
    )

    # Summary information
    start_time: Optional[float] = Field(None, description="Start time of the workflow")
    end_time: Optional[float] = Field(None, description="End time of the workflow")
    successful_tests: int = Field(0, description="Number of successful tests")
    failed_tests: int = Field(0, description="Number of failed tests")
    skipped_tests: int = Field(0, description="Number of skipped tests")
    fixed_tests: int = Field(0, description="Number of fixed tests")
    total_files: int = Field(0, description="Total number of files to process")
    is_completed: bool = Field(False, description="Whether the workflow has completed")

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True
        str_strip_whitespace = True
