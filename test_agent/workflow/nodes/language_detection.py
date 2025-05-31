# test_agent/workflow/nodes/language_detection.py

import logging

from test_agent.language import detect_language, get_adapter
from test_agent.workflow import WorkflowState

# Configure logging
logger = logging.getLogger(__name__)


async def detect_project_language(state: WorkflowState) -> WorkflowState:
    """
    Node to detect the programming language used in the project.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    project_dir = state.project.root_directory
    logger.info(f"Detecting language for project: {project_dir}")

    # Get language from state if already set
    language = state.project.language

    if language is None:
        try:
            # Auto-detect language
            language = detect_language(project_dir)

            if language:
                logger.info(f"Detected language: {language}")

                # Get language adapter
                language_adapter = get_adapter(language)

                if language_adapter:
                    # Update state with detected language
                    state.project.language = language

                    # Add a decision to the memory
                    if state.memory and hasattr(state.memory, "record_decision"):
                        state.memory.record_decision("detected_language", language)

                    logger.info(f"Language detection successful: {language}")
                else:
                    error_msg = (
                        f"Failed to find adapter for detected language: {language}"
                    )
                    logger.error(error_msg)
                    state.errors.append(
                        {
                            "phase": "language_detection",
                            "error": error_msg,
                            "type": "adapter_not_found",
                        }
                    )
            else:
                error_msg = "Could not detect language automatically"
                logger.error(error_msg)
                state.errors.append(
                    {
                        "phase": "language_detection",
                        "error": error_msg,
                        "type": "detection_failed",
                    }
                )

        except Exception as e:
            error_msg = f"Error during language detection: {str(e)}"
            logger.exception(error_msg)
            state.errors.append(
                {"phase": "language_detection", "error": error_msg, "type": "exception"}
            )

    else:
        logger.info(f"Using language from state: {language}")

    # Determine next phase based on success
    if state.project.language:
        state.current_phase = "language_detection"
        state.next_phase = "project_analysis"
    else:
        state.current_phase = "language_detection"
        state.next_phase = "error"

    return state
