# test_agent/workflow/nodes/file_analysis.py

import logging
import time
import asyncio
from typing import Optional

from test_agent.language import get_adapter
from test_agent.memory import CacheManager
from test_agent.workflow import WorkflowState, FileInfo

# Configure logging
logger = logging.getLogger(__name__)


async def analyze_single_file(
    file_info: FileInfo, language_adapter, cache_manager: Optional[CacheManager] = None
) -> FileInfo:
    """
    Analyze a single source file.

    Args:
        file_info: File information
        language_adapter: Language adapter for the file
        cache_manager: Optional cache manager

    Returns:
        Updated file information with analysis
    """
    file_path = file_info.path

    try:
        # Check cache first if available
        if cache_manager:
            cached_analysis = cache_manager.get_analysis_cache(file_path)
            if cached_analysis:
                file_info.analysis = cached_analysis
                return file_info

        # Analyze file
        analysis = language_adapter.analyze_source_file(file_path)

        # Update file info
        file_info.analysis = analysis

        # Update cache if available
        if cache_manager:
            cache_manager.set_analysis_cache(file_path, analysis)

        return file_info

    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {str(e)}")
        file_info.analysis = {"error": str(e)}
        return file_info


async def analyze_files(state: WorkflowState) -> WorkflowState:
    """
    Node to analyze source files.

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
            {"phase": "file_analysis", "error": error_msg, "type": "adapter_not_found"}
        )
        state.next_phase = "error"
        return state

    # Initialize cache manager
    cache_manager = CacheManager(state.project.root_directory)

    # Start timing
    start_time = time.time()

    # Get files to analyze
    files_to_analyze = [
        f for f in state.project.source_files if not f.skip and not f.analysis
    ]

    logger.info(f"Analyzing {len(files_to_analyze)} source files")

    # Process files in parallel using asyncio
    try:
        # Create analysis tasks
        tasks = []
        for file_info in files_to_analyze:
            tasks.append(
                analyze_single_file(file_info, language_adapter, cache_manager)
            )

        # Run tasks concurrently
        if tasks:
            analyzed_files = await asyncio.gather(*tasks)

            # Update state with analyzed files
            for analyzed_file in analyzed_files:
                # Find and update the file in state.project.source_files
                for i, file_info in enumerate(state.project.source_files):
                    if file_info.path == analyzed_file.path:
                        state.project.source_files[i] = analyzed_file
                        break

        # Calculate time taken
        time_taken = time.time() - start_time

        # Update state
        logger.info(f"File analysis complete in {time_taken:.2f}s")
        logger.info(f"Analyzed {len(tasks)} files")

        # Set next phase
        state.current_phase = "file_analysis"
        state.next_phase = "test_path_generation"

    except Exception as e:
        error_msg = f"Error during file analysis: {str(e)}"
        logger.exception(error_msg)
        state.errors.append(
            {"phase": "file_analysis", "error": error_msg, "type": "exception"}
        )
        state.next_phase = "error"

    return state
