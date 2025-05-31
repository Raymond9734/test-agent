# test_agent/workflow/nodes/test_path.py

import os
import logging


from test_agent.language import get_adapter
from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


async def generate_test_paths(state: WorkflowState) -> WorkflowState:
    """
    Node to generate test file paths for source files.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    # Get language adapter
    language = state.project.language
    language_adapter = get_adapter(language)

    if not language_adapter:
        error_msg = f"No adapter found for language: {language}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_path_generation",
                "error": error_msg,
                "type": "adapter_not_found",
            }
        )
        state.next_phase = "error"
        return state

    # Get test directory
    test_directory = state.project.test_directory

    if not test_directory:
        test_directory = os.path.join(state.project.root_directory, "tests")
        state.project.test_directory = test_directory
        # Create test directory if it doesn't exist
        os.makedirs(test_directory, exist_ok=True)

    logger.info(f"Generating test paths using test directory: {test_directory}")

    # Get project root
    # project_root = state.project.root_directory

    # Get project test patterns
    test_pattern = state.project.patterns

    # Process files that need test paths
    files_to_process = [
        f for f in state.project.source_files if not f.skip and not f.has_existing_test
    ]

    logger.info(f"Generating test paths for {len(files_to_process)} files")

    # Process each source file
    for file_info in files_to_process:
        source_file = file_info.path

        try:
            # Generate test path
            test_path = language_adapter.generate_test_path(
                source_file, test_directory, test_pattern
            )

            # Create test directory if needed
            test_dir = os.path.dirname(test_path)
            os.makedirs(test_dir, exist_ok=True)

            # Create test info
            test_info = TestInfo(
                source_file=source_file, test_path=test_path, status=TestStatus.PENDING
            )

            # Add to state
            state.tests[source_file] = test_info

        except Exception as e:
            logger.error(f"Error generating test path for {source_file}: {str(e)}")
            state.warnings.append(
                {
                    "phase": "test_path_generation",
                    "file": source_file,
                    "error": str(e),
                    "type": "test_path_generation_failed",
                }
            )

    # Update state with existing tests
    for file_info in [f for f in state.project.source_files if f.has_existing_test]:
        source_file = file_info.path
        existing_test = file_info.existing_test_path

        if existing_test:
            # Create test info for existing test
            test_info = TestInfo(
                source_file=source_file,
                test_path=existing_test,
                status=TestStatus.SKIPPED,
                content=None,  # We don't need to read the content yet
            )

            # Add to state
            state.tests[source_file] = test_info

    # Log summary
    logger.info(f"Generated {len(state.tests)} test paths")
    logger.info(
        f"Skipped {len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])} existing tests"
    )

    # Set next phase
    state.current_phase = "test_path_generation"
    state.next_phase = "test_generation"

    return state
