# test_agent/workflow/nodes/test_generation.py

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple

from test_agent.language import get_adapter
from test_agent.llm import get_provider
from test_agent.memory import CacheManager
from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


async def generate_test_content(
    source_file: str,
    test_info: TestInfo,
    language_adapter,
    llm_provider,
    cache_manager: Optional[CacheManager] = None,
    file_analysis: Optional[Dict[str, Any]] = None,
    test_pattern: Optional[Dict[str, Any]] = None,
) -> Tuple[TestInfo, bool]:
    """
    Generate test content for a source file.

    Args:
        source_file: Path to the source file
        test_info: Test information
        language_adapter: Language adapter for the file
        llm_provider: LLM provider
        cache_manager: Optional cache manager
        file_analysis: Optional file analysis (to avoid re-analysis)
        test_pattern: Optional test pattern

    Returns:
        Tuple of (updated test info, success flag)
    """
    try:
        # Set status to running
        test_info.status = TestStatus.RUNNING

        # Check cache first if available
        if cache_manager:
            cached_template = cache_manager.get_template_cache(source_file)
            if cached_template and "content" in cached_template:
                test_info.content = cached_template["content"]
                test_info.status = TestStatus.PENDING
                return test_info, True

        # Get analysis if not provided
        if not file_analysis:
            file_analysis = language_adapter.analyze_source_file(source_file)

        # Generate test template
        template = language_adapter.generate_test_template(
            source_file, file_analysis, test_pattern
        )

        # Enhance template with LLM
        prompt = f"""
        You are an expert test engineer tasked with creating a comprehensive test for a source file.
        
        I'll provide you with:
        1. The source file path
        2. A basic test template
        3. Analysis of the source file
        
        Your task is to enhance the template to create a complete, working test file. Follow these guidelines:
        - Focus on testing the public interface of functions and classes
        - Use good test design practices (arrange-act-assert pattern)
        - Include edge cases and error scenarios
        - Don't modify imports or basic structure - just fill in the test implementations
        - Keep your tests focused, testing one thing at a time
        - Use appropriate test fixtures/mocks where needed
        
        Source file: {source_file}
        
        Source file analysis: {file_analysis}
        
        Basic test template:
        ```
        {template}
        ```
        
        Enhance this template into a complete test file. Return ONLY the complete test code without explanations.
        """

        # Call LLM to enhance template
        response = await llm_provider.generate(prompt)

        # Extract code from response
        code = response

        # Try to extract code block if the response contains explanations
        import re

        code_matches = re.findall(r"```(?:python|go)?\n(.*?)```", code, re.DOTALL)
        if code_matches:
            # Use the longest code block (most complete)
            code = max(code_matches, key=len)

        # Save test content
        test_info.content = code

        # Update cache if available
        if cache_manager:
            cache_manager.set_template_cache(source_file, {"content": code})

        # Update status
        test_info.status = TestStatus.PENDING

        return test_info, True

    except Exception as e:
        logger.error(f"Error generating test for {source_file}: {str(e)}")
        test_info.error_message = str(e)
        test_info.status = TestStatus.ERROR
        return test_info, False


async def generate_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to generate test contents using LLM.

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
                "phase": "test_generation",
                "error": error_msg,
                "type": "adapter_not_found",
            }
        )
        state.next_phase = "error"
        return state

    # Get LLM provider
    if not state.llm or not state.llm.provider:
        error_msg = "No LLM provider specified in state"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_generation",
                "error": error_msg,
                "type": "llm_provider_not_found",
            }
        )
        state.next_phase = "error"
        return state

    llm_provider = get_provider(state.llm.provider)

    # Initialize cache manager
    cache_manager = CacheManager(state.project.root_directory)

    # Get test pattern
    test_pattern = state.project.patterns

    # Start timing
    start_time = time.time()

    # Get files to generate tests for
    tests_to_generate = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status == TestStatus.PENDING and not test_info.content
    }

    logger.info(f"Generating tests for {len(tests_to_generate)} files")

    # Generate tests in batches to avoid overloading the LLM API
    batch_size = 5
    files = list(tests_to_generate.keys())

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {len(batch)} files"
        )

        # Process files in parallel
        tasks = []
        for source_file in batch:
            test_info = tests_to_generate[source_file]

            # Get file analysis
            file_analysis = None
            for file_info in state.project.source_files:
                if file_info.path == source_file:
                    file_analysis = file_info.analysis
                    break

            tasks.append(
                generate_test_content(
                    source_file,
                    test_info,
                    language_adapter,
                    llm_provider,
                    cache_manager,
                    file_analysis,
                    test_pattern,
                )
            )

        # Run batch
        results = await asyncio.gather(*tasks)

        # Update state with results
        for test_info, success in results:
            state.tests[test_info.source_file] = test_info

        # Add a small delay between batches to avoid rate limiting
        if i + batch_size < len(files):
            await asyncio.sleep(1)

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state
    successful = len(
        [t for t in state.tests.values() if t.content and t.status != TestStatus.ERROR]
    )
    failed = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    logger.info(f"Test generation complete in {time_taken:.2f}s")
    logger.info(f"Generated {successful} tests successfully")
    logger.info(f"Failed to generate {failed} tests")
    logger.info(f"Skipped {skipped} existing tests")

    # Set next phase
    state.current_phase = "test_generation"
    state.next_phase = "test_execution"

    # Write test files to disk
    for test_info in state.tests.values():
        if test_info.content and test_info.status != TestStatus.ERROR:
            try:
                # Create directory if needed
                test_dir = os.path.dirname(test_info.test_path)
                os.makedirs(test_dir, exist_ok=True)

                # Write test file
                with open(test_info.test_path, "w") as f:
                    f.write(test_info.content)

                logger.debug(f"Wrote test file: {test_info.test_path}")

            except Exception as e:
                logger.error(f"Error writing test file {test_info.test_path}: {str(e)}")
                test_info.error_message = f"Error writing test file: {str(e)}"
                test_info.status = TestStatus.ERROR
                state.tests[test_info.source_file] = test_info

    return state
