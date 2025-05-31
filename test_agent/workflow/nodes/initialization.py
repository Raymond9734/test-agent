# test_agent/workflow/nodes/initialization.py

import os
import logging
import time

from test_agent.language import get_adapter
from test_agent.workflow import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


def initialize_workflow(state: WorkflowState) -> WorkflowState:
    """
    Node to initialize the workflow.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    logger.info("Initializing workflow")

    # Set start time
    state.start_time = time.time()

    try:
        # Validate project directory
        if not state.project or not state.project.root_directory:
            error_msg = "No project directory specified"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "initialization",
                    "error": error_msg,
                    "type": "missing_project_directory",
                }
            )
            state.next_phase = "error"
            return state

        project_dir = state.project.root_directory

        if not os.path.exists(project_dir):
            error_msg = f"Project directory does not exist: {project_dir}"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "initialization",
                    "error": error_msg,
                    "type": "project_directory_not_found",
                }
            )
            state.next_phase = "error"
            return state

        if not os.path.isdir(project_dir):
            error_msg = f"Project path is not a directory: {project_dir}"
            logger.error(error_msg)
            state.errors.append(
                {
                    "phase": "initialization",
                    "error": error_msg,
                    "type": "project_path_not_directory",
                }
            )
            state.next_phase = "error"
            return state

        # Validate LLM provider if specified
        if state.llm and state.llm.provider:
            try:
                logger.info(f"Using LLM provider: {state.llm.provider}")
            except ValueError as e:
                error_msg = f"Invalid LLM provider: {str(e)}"
                logger.error(error_msg)
                state.errors.append(
                    {
                        "phase": "initialization",
                        "error": error_msg,
                        "type": "invalid_llm_provider",
                    }
                )
                state.next_phase = "error"
                return state

        # Validate language if specified
        if state.project.language:
            language_adapter = get_adapter(state.project.language)
            if not language_adapter:
                error_msg = f"Unsupported language: {state.project.language}"
                logger.error(error_msg)
                state.errors.append(
                    {
                        "phase": "initialization",
                        "error": error_msg,
                        "type": "unsupported_language",
                    }
                )
                state.next_phase = "error"
                return state

            logger.info(f"Using language: {state.project.language}")

        # Set the starting phase
        state.current_phase = "initialization"

        # Determine next phase
        if state.project.language:
            # Skip language detection if already specified
            logger.info("Language already specified, skipping detection")
            state.next_phase = "project_analysis"
        else:
            # Start with language detection
            logger.info("No language specified, will detect language next")
            state.next_phase = "language_detection"

        # DEBUGGING: Print the state at the end of initialization
        logger.info(
            f"State after initialization: current_phase={state.current_phase}, next_phase={state.next_phase}"
        )

        logger.info("Initialization complete")

        return state

    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "initialization", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"
        return state

    except Exception as e:
        error_msg = f"Error during initialization: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "initialization", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"
        return state
