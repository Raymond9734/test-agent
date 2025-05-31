# test_agent/workflow/nodes/test_execution.py

import os
import logging
import time
import asyncio
import subprocess
import tempfile
from typing import Optional, Tuple

from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


def setup_test_environment(language: str) -> Tuple[bool, str, Optional[str]]:
    """
    Set up the test environment for a specific language.

    Args:
        language: Language name

    Returns:
        Tuple of (success flag, message, environment path or None)
    """
    try:
        if language.lower() == "python":
            # Create a virtual environment for Python
            venv_dir = os.path.join(tempfile.gettempdir(), "test_agent_venv")

            if not os.path.exists(venv_dir):
                logger.info(f"Creating Python virtual environment at {venv_dir}")

                # Use Python's venv module
                import venv

                venv.create(venv_dir, with_pip=True, clear=True)

                # Install pytest
                if os.name == "nt":  # Windows
                    pip_path = os.path.join(venv_dir, "Scripts", "pip")
                else:  # Unix/Linux/Mac
                    pip_path = os.path.join(venv_dir, "bin", "pip")

                subprocess.run(
                    [pip_path, "install", "pytest"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                logger.info("Installed pytest in virtual environment")

            return True, f"Test environment set up at {venv_dir}", venv_dir

        elif language.lower() == "go":
            # Go doesn't need a special environment, it uses go test command
            return True, "Using system Go installation for tests", None

        else:
            return False, f"No test environment setup implemented for {language}", None

    except Exception as e:
        error_msg = f"Error setting up test environment: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, None


async def run_test(
    test_info: TestInfo, language: str, env_path: Optional[str] = None
) -> TestInfo:
    """
    Run a single test.

    Args:
        test_info: Test information
        language: Language name
        env_path: Optional path to the test environment

    Returns:
        Updated test information
    """
    if test_info.status == TestStatus.SKIPPED:
        # Skip existing tests
        return test_info

    if not os.path.exists(test_info.test_path):
        test_info.status = TestStatus.ERROR
        test_info.error_message = f"Test file not found: {test_info.test_path}"
        return test_info

    try:
        # Update status
        test_info.status = TestStatus.RUNNING

        # Get command based on language
        if language.lower() == "python":
            if os.name == "nt":  # Windows
                python_path = os.path.join(env_path, "Scripts", "python")
            else:  # Unix/Linux/Mac
                python_path = os.path.join(env_path, "bin", "python")

            # Set up environment variables
            env = os.environ.copy()

            # Add project directory to PYTHONPATH to allow imports
            project_dir = os.path.dirname(test_info.source_file)
            if "PYTHONPATH" in env:
                if os.name == "nt":  # Windows
                    env["PYTHONPATH"] = f"{project_dir};{env['PYTHONPATH']}"
                else:  # Unix/Linux/Mac
                    env["PYTHONPATH"] = f"{project_dir}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = project_dir

            # Create command
            command = [python_path, "-m", "pytest", "-v", test_info.test_path]

        elif language.lower() == "go":
            # Get directory containing the test file
            # test_dir = os.path.dirname(test_info.test_path)

            # Create command
            command = ["go", "test", "-v", test_info.test_path]

            # Set up environment variables
            env = os.environ.copy()

        else:
            test_info.status = TestStatus.ERROR
            test_info.error_message = f"No test execution implemented for {language}"
            return test_info

        # Run the test
        logger.info(f"Running test: {test_info.test_path}")

        # Execute the command
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(test_info.test_path),
        )

        # Get output
        stdout, stderr = await process.communicate()

        # Store execution result
        test_info.execution_result = (
            f"STDOUT:\n{stdout.decode()}\n\nSTDERR:\n{stderr.decode()}"
        )

        # Update status based on exit code
        if process.returncode == 0:
            test_info.status = TestStatus.PASSED
            logger.info(f"Test passed: {test_info.test_path}")
        else:
            # Check if it's a test failure (assertion failed) or an error
            if "AssertionError" in test_info.execution_result:
                test_info.status = TestStatus.FAILED
                logger.info(f"Test failed (assertions): {test_info.test_path}")
            else:
                test_info.status = TestStatus.ERROR
                test_info.error_message = (
                    f"Test execution error (code {process.returncode})"
                )
                logger.info(f"Test error: {test_info.test_path}")

        return test_info

    except Exception as e:
        test_info.status = TestStatus.ERROR
        test_info.error_message = f"Error running test: {str(e)}"
        logger.error(f"Error running test {test_info.test_path}: {str(e)}")
        return test_info


async def execute_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to execute generated tests.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    logger.info(f"Executing tests for language: {language}")

    # Set up test environment
    success, message, env_path = setup_test_environment(language)

    if not success:
        error_msg = f"Failed to set up test environment: {message}"
        logger.error(error_msg)
        state.errors.append(
            {
                "phase": "test_execution",
                "error": error_msg,
                "type": "environment_setup_failed",
            }
        )
        state.next_phase = "error"
        return state

    logger.info(message)

    # Start timing
    start_time = time.time()

    # Get tests to execute (skip already executed tests)
    tests_to_execute = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status in [TestStatus.PENDING]
    }

    # Skip tests that don't have content
    tests_to_execute = {
        source_file: test_info
        for source_file, test_info in tests_to_execute.items()
        if test_info.content
    }

    logger.info(f"Executing {len(tests_to_execute)} tests")

    # Run tests in batches to avoid resource issues
    batch_size = 5
    files = list(tests_to_execute.keys())

    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}: {len(batch)} files"
        )

        # Process files in parallel
        tasks = []
        for source_file in batch:
            test_info = tests_to_execute[source_file]
            tasks.append(run_test(test_info, language, env_path))

        # Run batch
        results = await asyncio.gather(*tasks)

        # Update state with results
        for test_info in results:
            state.tests[test_info.source_file] = test_info

        # Add a small delay between batches
        if i + batch_size < len(files):
            await asyncio.sleep(1)

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state
    passed = len([t for t in state.tests.values() if t.status == TestStatus.PASSED])
    failed = len([t for t in state.tests.values() if t.status == TestStatus.FAILED])
    error = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    state.successful_tests = passed
    state.failed_tests = failed + error

    logger.info(f"Test execution complete in {time_taken:.2f}s")
    logger.info(
        f"Passed: {passed}, Failed: {failed}, Error: {error}, Skipped: {skipped}"
    )

    # Set next phase
    state.current_phase = "test_execution"

    # Determine next phase based on results
    if failed > 0 or error > 0:
        # Need to fix tests
        state.next_phase = "test_fixing"
    else:
        # All tests passed or skipped, we're done
        state.next_phase = "complete"

    return state
