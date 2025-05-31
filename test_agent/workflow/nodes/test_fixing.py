# test_agent/workflow/nodes/test_fixing.py

import os
import logging
import time
import asyncio
import re
from typing import Dict, Any, Optional

from test_agent.llm import get_provider
from test_agent.workflow import WorkflowState, TestInfo, TestStatus
from .test_execution import run_test

# Configure logging
logger = logging.getLogger(__name__)


def analyze_test_error(execution_result: str) -> Dict[str, Any]:
    """
    Analyze a test execution result to determine the type of error.

    Args:
        execution_result: Test execution output

    Returns:
        Dictionary with error analysis
    """
    result = {
        "has_syntax_error": False,
        "has_import_error": False,
        "has_assertion_error": False,
        "has_exception": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "error_line": None,
        "missing_dependencies": [],
    }

    # Skip if no execution result
    if not execution_result:
        return result

    # Check for common error types
    if "SyntaxError" in execution_result:
        result["has_syntax_error"] = True
        result["error_type"] = "SyntaxError"

        # Extract error details
        syntax_match = re.search(r"SyntaxError: (.*?)(?:\n|$)", execution_result)
        if syntax_match:
            result["error_message"] = syntax_match.group(1)

        # Extract line number
        line_match = re.search(r"line (\d+)", execution_result)
        if line_match:
            result["error_line"] = line_match.group(1)

    elif "ImportError" in execution_result or "ModuleNotFoundError" in execution_result:
        result["has_import_error"] = True
        result["error_type"] = (
            "ImportError"
            if "ImportError" in execution_result
            else "ModuleNotFoundError"
        )

        # Extract error details
        import_match = re.search(
            r"(ImportError|ModuleNotFoundError): (.*?)(?:\n|$)", execution_result
        )
        if import_match:
            result["error_message"] = import_match.group(2)

        # Extract missing module
        if "No module named" in execution_result:
            module_match = re.search(r"No module named '(.*?)'", execution_result)
            if module_match:
                module_name = module_match.group(1)
                result["missing_dependencies"].append(module_name)

    elif "AssertionError" in execution_result:
        result["has_assertion_error"] = True
        result["error_type"] = "AssertionError"

        # Extract error details
        assertion_match = re.search(r"AssertionError: (.*?)(?:\n|$)", execution_result)
        if assertion_match:
            result["error_message"] = assertion_match.group(1)

    elif "Exception" in execution_result or "Error" in execution_result:
        result["has_exception"] = True

        # Try to extract error type
        error_match = re.search(
            r"([A-Za-z]+Error|Exception): (.*?)(?:\n|$)", execution_result
        )
        if error_match:
            result["error_type"] = error_match.group(1)
            result["error_message"] = error_match.group(2)

    return result


async def fix_test(
    test_info: TestInfo,
    language: str,
    llm_provider,
    env_path: Optional[str] = None,
    max_attempts: int = 3,
) -> TestInfo:
    """
    Attempt to fix a failing test.

    Args:
        test_info: Test information
        language: Language name
        llm_provider: LLM provider
        env_path: Optional path to the test environment
        max_attempts: Maximum number of fix attempts

    Returns:
        Updated test information
    """
    # Skip if no execution result or not a failed/error status
    if not test_info.execution_result or test_info.status not in [
        TestStatus.FAILED,
        TestStatus.ERROR,
    ]:
        return test_info

    # Skip if already tried too many times
    if test_info.fix_attempts >= max_attempts:
        logger.info(
            f"Maximum fix attempts ({max_attempts}) reached for {test_info.test_path}"
        )
        return test_info

    logger.info(
        f"Attempting to fix test: {test_info.test_path} (attempt {test_info.fix_attempts + 1}/{max_attempts})"
    )

    # Analyze the error
    error_analysis = analyze_test_error(test_info.execution_result)

    try:
        # Read the current test content
        with open(test_info.test_path, "r") as f:
            current_content = f.read()

        # Save to fix history
        if test_info.fix_history is None:
            test_info.fix_history = []

        test_info.fix_history.append(current_content)

        # Prepare the prompt
        prompt = f"""
        I need help fixing a failing test. I'll provide:
        1. The test content
        2. The error output
        3. An analysis of the error
        
        Test file: {os.path.basename(test_info.test_path)}
        Source file: {os.path.basename(test_info.source_file)}
        
        Error output:
        {test_info.execution_result}
        
        Error analysis:
        - Error type: {error_analysis.get('error_type')}
        - Error message: {error_analysis.get('error_message')}
        - Has syntax error: {error_analysis.get('has_syntax_error')}
        - Has import error: {error_analysis.get('has_import_error')}
        - Has assertion error: {error_analysis.get('has_assertion_error')}
        - Has exception: {error_analysis.get('has_exception')}
        
        Current test content:
        ```
        {current_content}
        ```
        
        Please fix the test based on the error output. Return ONLY the corrected test code without explanations.
        """

        # Call LLM to fix test
        response = await llm_provider.generate(prompt)

        # Extract code from response
        fixed_code = response

        # Try to extract code block if the response contains explanations
        import re

        code_matches = re.findall(r"```(?:python|go)?\n(.*?)```", fixed_code, re.DOTALL)
        if code_matches:
            # Use the longest code block (most complete)
            fixed_code = max(code_matches, key=len)

        # Increment fix attempts
        test_info.fix_attempts += 1

        # Check if the content actually changed
        if fixed_code.strip() == current_content.strip():
            logger.warning(
                f"Fix attempt produced identical code for {test_info.test_path}"
            )
            test_info.error_message = "Fix attempt produced identical code"
            return test_info

        # Write the fixed code to the test file
        with open(test_info.test_path, "w") as f:
            f.write(fixed_code)

        # Update the test content
        test_info.content = fixed_code

        # Run the fixed test
        updated_test_info = await run_test(test_info, language, env_path)

        # Keep fix history and incremented fix attempts
        updated_test_info.fix_history = test_info.fix_history
        updated_test_info.fix_attempts = test_info.fix_attempts

        # Check if the fix was successful
        if updated_test_info.status == TestStatus.PASSED:
            logger.info(f"Test fixed successfully: {test_info.test_path}")
            updated_test_info.status = TestStatus.FIXED

        return updated_test_info

    except Exception as e:
        logger.error(f"Error fixing test {test_info.test_path}: {str(e)}")
        test_info.error_message = f"Error fixing test: {str(e)}"
        test_info.fix_attempts += 1
        return test_info


async def fix_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to fix failing tests.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    # Get LLM provider
    if not state.llm or not state.llm.provider:
        error_msg = "No LLM provider specified in state"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_fixing",
                "error": error_msg,
                "type": "llm_provider_not_found",
            }
        )
        state.next_phase = "error"
        return state

    llm_provider = get_provider(state.llm.provider)

    # Get environment path
    from .test_execution import setup_test_environment

    success, message, env_path = setup_test_environment(language)

    if not success:
        error_msg = f"Failed to set up test environment: {message}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_fixing",
                "error": error_msg,
                "type": "environment_setup_failed",
            }
        )
        state.next_phase = "error"
        return state

    logger.info(message)

    # Start timing
    start_time = time.time()

    # Get tests to fix
    tests_to_fix = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status in [TestStatus.FAILED, TestStatus.ERROR]
    }

    logger.info(f"Attempting to fix {len(tests_to_fix)} tests")

    # Fix tests in batches
    batch_size = 3  # Smaller batch size for fixing as it requires more resources
    files = list(tests_to_fix.keys())

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {len(batch)} files"
        )

        # Process files in parallel
        tasks = []
        for source_file in batch:
            test_info = tests_to_fix[source_file]
            tasks.append(fix_test(test_info, language, llm_provider, env_path))

        # Run batch
        results = await asyncio.gather(*tasks)

        # Update state with results
        for test_info in results:
            state.tests[test_info.source_file] = test_info

        # Add a small delay between batches
        if i + batch_size < len(files):
            await asyncio.sleep(2)

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state
    passed = len([t for t in state.tests.values() if t.status == TestStatus.PASSED])
    fixed = len([t for t in state.tests.values() if t.status == TestStatus.FIXED])
    failed = len([t for t in state.tests.values() if t.status == TestStatus.FAILED])
    error = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    state.successful_tests = passed + fixed
    state.failed_tests = failed + error
    state.fixed_tests = fixed

    logger.info(f"Test fixing complete in {time_taken:.2f}s")
    logger.info(
        f"Passed: {passed}, Fixed: {fixed}, Failed: {failed}, Error: {error}, Skipped: {skipped}"
    )

    # Set next phase
    state.current_phase = "test_fixing"
    state.next_phase = "complete"

    return state
