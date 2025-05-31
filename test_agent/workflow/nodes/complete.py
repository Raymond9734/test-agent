# test_agent/workflow/nodes/complete.py

import os
import logging
import time

from test_agent.workflow import WorkflowState, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


def generate_summary(state: WorkflowState) -> str:
    """
    Generate a summary of the test generation process.

    Args:
        state: Workflow state

    Returns:
        Summary text
    """
    # Calculate counts
    total_files = len(state.project.source_files)
    tests_generated = len([t for t in state.tests.values() if t.content])
    passed_tests = len(
        [t for t in state.tests.values() if t.status == TestStatus.PASSED]
    )
    fixed_tests = len([t for t in state.tests.values() if t.status == TestStatus.FIXED])
    failed_tests = len(
        [t for t in state.tests.values() if t.status == TestStatus.FAILED]
    )
    error_tests = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped_tests = len(
        [t for t in state.tests.values() if t.status == TestStatus.SKIPPED]
    )

    # Calculate timing if available
    time_taken = "unknown"
    if state.start_time and state.end_time:
        time_taken = f"{state.end_time - state.start_time:.2f}s"

    # Generate summary text
    summary = f"""
Test Generation Summary
----------------------

Project: {state.project.root_directory}
Language: {state.project.language}
Test Directory: {state.project.test_directory}

Files & Tests:
- Source files analyzed: {total_files}
- Tests generated: {tests_generated}
- Tests skipped (already existed): {skipped_tests}

Test Results:
- Passed: {passed_tests}
- Fixed: {fixed_tests}
- Failed: {failed_tests}
- Errors: {error_tests}
- Success rate: {(passed_tests + fixed_tests) / max(1, tests_generated - skipped_tests):.1%}

Time taken: {time_taken}
"""

    # Add error summary if there were errors
    if state.errors:
        summary += "\nErrors encountered:\n"
        for i, error in enumerate(state.errors[:5], 1):  # Show first 5 errors
            summary += f"- Error {i}: {error.get('error', 'Unknown error')}\n"

        if len(state.errors) > 5:
            summary += f"- ... and {len(state.errors) - 5} more errors\n"

    # Add warnings summary if there were warnings
    if state.warnings:
        summary += "\nWarnings:\n"
        for i, warning in enumerate(state.warnings[:5], 1):  # Show first 5 warnings
            summary += f"- Warning {i}: {warning.get('error', 'Unknown warning')}\n"

        if len(state.warnings) > 5:
            summary += f"- ... and {len(state.warnings) - 5} more warnings\n"

    # Add list of test files
    summary += "\nGenerated test files:\n"

    # Sort by status for a better overview
    test_files = []

    for test_info in state.tests.values():
        if test_info.status in [TestStatus.PASSED, TestStatus.FIXED]:
            test_files.append((test_info.test_path, "✅"))
        elif test_info.status == TestStatus.FAILED:
            test_files.append((test_info.test_path, "❌"))
        elif test_info.status == TestStatus.ERROR:
            test_files.append((test_info.test_path, "⚠️"))
        elif test_info.status == TestStatus.SKIPPED:
            test_files.append((test_info.test_path, "⏭️"))

    # Sort by path for a consistent output
    test_files.sort()

    # Add files to summary, truncating if too many
    max_files_to_show = 20
    for i, (path, status) in enumerate(test_files[:max_files_to_show], 1):
        rel_path = os.path.relpath(path, state.project.root_directory)
        summary += f"{status} {rel_path}\n"

    if len(test_files) > max_files_to_show:
        summary += f"... and {len(test_files) - max_files_to_show} more test files\n"

    # Add next steps
    summary += f"""
Next Steps:
- Run the tests: cd {state.project.root_directory} && {'python -m pytest' if state.project.language == 'python' else 'go test ./...'}
- Review and improve tests as needed
- Add tests for specific edge cases
"""

    return summary


async def complete_workflow(state: WorkflowState) -> WorkflowState:
    """
    Node to complete the workflow and generate summary.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    logger.info("Completing workflow")

    # Mark as completed
    state.is_completed = True

    # Set end time
    state.end_time = time.time()

    # Generate summary
    summary = generate_summary(state)

    # Log summary
    logger.info("Test generation completed successfully")
    logger.info(f"\n{summary}")

    # Update state
    state.current_phase = "complete"
    state.next_phase = None

    return state
