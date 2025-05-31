# test_agent/workflow/nodes/error.py

import logging
import time

from test_agent.workflow import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


def generate_error_report(state: WorkflowState) -> str:
    """
    Generate a report of errors encountered during the workflow.

    Args:
        state: Workflow state

    Returns:
        Error report text
    """
    # Generate heading
    report = """
Test Generation Error Report
---------------------------

The test generation process encountered errors that prevented completion.
"""

    # Add error details
    if state.errors:
        report += "\nErrors:\n"
        for i, error in enumerate(state.errors, 1):
            phase = error.get("phase", "unknown")
            error_msg = error.get("error", "Unknown error")
            error_type = error.get("type", "unknown")

            report += f"Error {i} (in {phase} phase): {error_msg}\n"
            report += f"  Type: {error_type}\n"

            # Add more details if available
            if "file" in error:
                report += f"  File: {error['file']}\n"
            if "details" in error:
                report += f"  Details: {error['details']}\n"

            report += "\n"
    else:
        report += "\nNo specific errors were recorded.\n"

    # Add warning details
    if state.warnings:
        report += "\nWarnings:\n"
        for i, warning in enumerate(state.warnings, 1):
            phase = warning.get("phase", "unknown")
            error_msg = warning.get("error", "Unknown warning")

            report += f"Warning {i} (in {phase} phase): {error_msg}\n"

            # Add more details if available
            if "file" in warning:
                report += f"  File: {warning['file']}\n"

            report += "\n"

    # Add troubleshooting suggestions
    report += """
Troubleshooting Suggestions:
---------------------------
"""

    # Add specific suggestions based on error types
    language_errors = [e for e in state.errors if e.get("type") == "adapter_not_found"]
    env_errors = [
        e for e in state.errors if e.get("type") == "environment_setup_failed"
    ]
    file_errors = [
        e
        for e in state.errors
        if e.get("type") in ["file_not_found", "permission_denied"]
    ]
    llm_errors = [
        e
        for e in state.errors
        if e.get("type") in ["llm_provider_not_found", "api_key_missing"]
    ]

    if language_errors:
        report += """
- Language Detection Issues:
  * Check if the project contains files with the expected extension
  * Ensure you specified the correct language manually if auto-detection failed
  * The tool currently supports Python and Go primarily
"""

    if env_errors:
        report += """
- Environment Setup Issues:
  * Check if the required tools are installed (Python/Go)
  * Ensure you have sufficient permissions to create virtual environments
  * Try clearing any existing environments and retry
"""

    if file_errors:
        report += """
- File Access Issues:
  * Check if the specified project directory exists and is accessible
  * Ensure you have read/write permissions for the project directory
  * If specific files were mentioned in errors, check their permissions
"""

    if llm_errors:
        report += """
- LLM Provider Issues:
  * Check if you specified a valid LLM provider (claude, openai, deepseek, gemini)
  * Ensure you provided a valid API key for the chosen provider
  * Check if the API key has the necessary permissions and quota
"""

    # Always add general suggestions
    report += """
General Suggestions:
  * Check the logs for more detailed error information
  * Try running with the --verbose flag for additional debugging information
  * Ensure all dependencies are installed: pip install -e .
  * Try using a different LLM provider if available
  * For more help, please report the issue with the full error log
"""

    return report


async def handle_error(state: WorkflowState) -> WorkflowState:
    """
    Node to handle workflow errors.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    logger.error("Workflow encountered errors, generating error report")

    # Set end time
    state.end_time = time.time()

    # Generate error report
    error_report = generate_error_report(state)

    # Log error report
    logger.error(f"\n{error_report}")

    # Update state
    state.current_phase = "error"
    state.next_phase = None
    state.is_completed = True

    return state
