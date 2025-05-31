# test_agent/workflow/nodes/test_execution.py - Fixed with robust environment setup

import os
import logging
import time
import asyncio
import subprocess
import tempfile
import sys
from typing import Optional, Tuple

from test_agent.workflow import WorkflowState, TestInfo, TestStatus

# Configure logging
logger = logging.getLogger(__name__)


def setup_test_environment(
    language: str, force_recreate: bool = False
) -> Tuple[bool, str, Optional[str]]:
    """
    Set up the test environment for a specific language with robust error handling.

    Args:
        language: Language name
        force_recreate: Whether to force recreation of existing environment

    Returns:
        Tuple of (success flag, message, environment path or None)
    """
    try:
        if language.lower() == "python":
            # Create a virtual environment for Python
            venv_dir = os.path.join(tempfile.gettempdir(), "test_agent_venv")

            # Remove existing environment if force_recreate or if it's broken
            if force_recreate and os.path.exists(venv_dir):
                logger.info(f"Removing existing virtual environment at {venv_dir}")
                import shutil

                shutil.rmtree(venv_dir, ignore_errors=True)

            # Check if environment exists and is functional
            if os.path.exists(venv_dir):
                # Check if pip exists
                pip_path = os.path.join(
                    venv_dir, "Scripts" if os.name == "nt" else "bin", "pip"
                )
                python_path = os.path.join(
                    venv_dir, "Scripts" if os.name == "nt" else "bin", "python"
                )

                if os.path.exists(pip_path) and os.path.exists(python_path):
                    logger.info(f"Using existing virtual environment at {venv_dir}")
                    return (
                        True,
                        f"Using existing test environment at {venv_dir}",
                        venv_dir,
                    )
                else:
                    logger.warning("Existing environment is broken, recreating...")
                    import shutil

                    shutil.rmtree(venv_dir, ignore_errors=True)

            logger.info(f"Creating new Python virtual environment at {venv_dir}")

            # Create virtual environment using subprocess for better error handling
            try:
                # Use current Python interpreter to create venv
                subprocess.run(
                    [sys.executable, "-m", "venv", venv_dir, "--clear"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )

                logger.info("Virtual environment created successfully")

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to create virtual environment: {e.stderr.decode() if e.stderr else str(e)}"
                )
                return False, f"Failed to create virtual environment: {str(e)}", None
            except subprocess.TimeoutExpired:
                logger.error("Virtual environment creation timed out")
                return False, "Virtual environment creation timed out", None

            # Verify paths exist
            if os.name == "nt":  # Windows
                pip_path = os.path.join(venv_dir, "Scripts", "pip")
                python_path = os.path.join(venv_dir, "Scripts", "python")
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(venv_dir, "bin", "pip")
                python_path = os.path.join(venv_dir, "bin", "python")

            if not os.path.exists(pip_path):
                logger.error(f"pip not found at expected path: {pip_path}")
                return False, f"pip not found at {pip_path}", None

            if not os.path.exists(python_path):
                logger.error(f"Python not found at expected path: {python_path}")
                return False, f"Python not found at {python_path}", None

            # Install pytest with better error handling
            logger.info("Installing pytest in virtual environment")
            try:
                result = subprocess.run(
                    [pip_path, "install", "-U", "pytest"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    text=True,
                )
                logger.info("pytest installed successfully")

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to install pytest: {e.stderr if e.stderr else str(e)}"
                )
                return False, f"Failed to install pytest: {str(e)}", None
            except subprocess.TimeoutExpired:
                logger.error("pytest installation timed out")
                return False, "pytest installation timed out", None

            # Verify pytest installation
            try:
                subprocess.run(
                    [python_path, "-m", "pytest", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                logger.info("pytest installation verified")
            except subprocess.CalledProcessError as e:
                logger.warning(f"pytest verification failed: {str(e)}")
            except subprocess.TimeoutExpired:
                logger.warning("pytest verification timed out")

            return True, f"Python test environment created at {venv_dir}", venv_dir

        elif language.lower() == "go":
            # Go doesn't need a special environment, it uses go test command
            try:
                subprocess.run(
                    ["go", "version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                return True, "Using system Go installation for tests", None
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False, "Go is not installed or not in PATH", None

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
    Run a single test with enhanced error capture and robust environment handling.

    Args:
        test_info: Test information
        language: Language name
        env_path: Optional path to the test environment

    Returns:
        Updated test information
    """
    # DEBUG: Log test execution start
    logger.debug("=" * 80)
    logger.debug(f"RUNNING TEST: {test_info.test_path}")
    logger.debug(f"Source file: {test_info.source_file}")
    logger.debug(f"Current status: {test_info.status}")
    logger.debug("=" * 80)

    if test_info.status == TestStatus.SKIPPED:
        # Skip existing tests
        logger.debug("Test is marked as SKIPPED, returning as-is")
        return test_info

    if not os.path.exists(test_info.test_path):
        test_info.status = TestStatus.ERROR
        test_info.error_message = f"Test file not found: {test_info.test_path}"
        logger.debug(f"Test file not found: {test_info.test_path}")
        return test_info

    try:
        # Update status
        test_info.status = TestStatus.RUNNING

        # Get command based on language
        if language.lower() == "python":
            # Ensure we have a valid environment
            if not env_path or not os.path.exists(env_path):
                logger.warning(
                    f"Environment path invalid: {env_path}, setting up new environment"
                )
                success, message, new_env_path = setup_test_environment(
                    language, force_recreate=True
                )
                if not success:
                    test_info.status = TestStatus.ERROR
                    test_info.error_message = (
                        f"Failed to setup test environment: {message}"
                    )
                    return test_info
                env_path = new_env_path

            if os.name == "nt":  # Windows
                python_path = os.path.join(env_path, "Scripts", "python")
            else:  # Unix/Linux/Mac
                python_path = os.path.join(env_path, "bin", "python")

            # Verify python path exists
            if not os.path.exists(python_path):
                logger.error(f"Python interpreter not found at {python_path}")
                test_info.status = TestStatus.ERROR
                test_info.error_message = (
                    f"Python interpreter not found at {python_path}"
                )
                return test_info

            # Set up environment variables for better Python path resolution
            env = os.environ.copy()

            # Get the project root directory (go up from test file to find project root)
            test_dir = os.path.dirname(test_info.test_path)
            project_root = os.path.dirname(
                test_dir
            )  # Assume tests are in project/tests/

            # Also try to find the actual project root by looking for common indicators
            current_dir = test_dir
            while current_dir and current_dir != os.path.dirname(current_dir):
                # Look for setup.py, pyproject.toml, or .git
                if any(
                    os.path.exists(os.path.join(current_dir, f))
                    for f in ["setup.py", "pyproject.toml", ".git", "requirements.txt"]
                ):
                    project_root = current_dir
                    break
                current_dir = os.path.dirname(current_dir)

            logger.debug(f"Detected project root: {project_root}")

            # Add multiple paths to PYTHONPATH for better import resolution
            paths_to_add = [
                project_root,  # Project root
                os.path.dirname(
                    test_info.source_file
                ),  # Directory containing source file
                test_dir,  # Test directory
            ]

            # Add parent directories of source file to handle nested packages
            source_dir = os.path.dirname(test_info.source_file)
            while (
                source_dir
                and source_dir != project_root
                and source_dir != os.path.dirname(source_dir)
            ):
                paths_to_add.append(source_dir)
                source_dir = os.path.dirname(source_dir)

            # Remove duplicates and join paths
            unique_paths = []
            for path in paths_to_add:
                if path not in unique_paths:
                    unique_paths.append(path)

            python_path_value = os.pathsep.join(unique_paths)

            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = (
                    f"{python_path_value}{os.pathsep}{env['PYTHONPATH']}"
                )
            else:
                env["PYTHONPATH"] = python_path_value

            logger.debug(f"Set PYTHONPATH to: {env['PYTHONPATH']}")

            # Create command with more verbose output
            command = [
                python_path,
                "-m",
                "pytest",
                "-v",  # verbose
                "-s",  # don't capture output
                "--tb=long",  # long traceback format
                "--no-header",  # no header
                test_info.test_path,
            ]

            # Set working directory to project root for better module resolution
            work_dir = project_root

        elif language.lower() == "go":
            # Get directory containing the test file
            work_dir = os.path.dirname(test_info.test_path)

            # Create command
            command = ["go", "test", "-v", test_info.test_path]

            # Set up environment variables
            env = os.environ.copy()

        else:
            test_info.status = TestStatus.ERROR
            test_info.error_message = f"No test execution implemented for {language}"
            return test_info

        # DEBUG: Log the command and environment
        logger.debug(f"Command: {' '.join(command)}")
        logger.debug(f"Working directory: {work_dir}")
        logger.debug("Environment variables added:")
        if language.lower() == "python":
            logger.debug(f"  PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")

        # Run the test
        logger.info(f"Running test: {test_info.test_path}")

        # Execute the command with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=work_dir,
            )

            # Get output with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30.0
                )
            except asyncio.TimeoutError:
                process.kill()
                test_info.status = TestStatus.ERROR
                test_info.error_message = "Test execution timed out after 30 seconds"
                test_info.execution_result = "Test execution timed out"
                return test_info

        except Exception as e:
            test_info.status = TestStatus.ERROR
            test_info.error_message = f"Failed to start test process: {str(e)}"
            test_info.execution_result = f"Process execution error: {str(e)}"
            logger.error(f"Failed to start test process: {str(e)}")
            return test_info

        # Combine stdout and stderr for full context
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

        # Create comprehensive execution result
        execution_result_parts = []

        if stdout_text.strip():
            execution_result_parts.append(f"STDOUT:\n{stdout_text}")

        if stderr_text.strip():
            execution_result_parts.append(f"STDERR:\n{stderr_text}")

        if not execution_result_parts:
            execution_result_parts.append("No output captured")

        # Add process information
        execution_result_parts.append("\nPROCESS INFO:")
        execution_result_parts.append(f"Return code: {process.returncode}")
        execution_result_parts.append(f"Command: {' '.join(command)}")
        execution_result_parts.append(f"Working directory: {work_dir}")

        execution_result = "\n\n".join(execution_result_parts)

        # Store execution result
        test_info.execution_result = execution_result

        # DEBUG: Log execution results
        logger.debug("TEST EXECUTION COMPLETED")
        logger.debug(f"Return code: {process.returncode}")
        logger.debug(f"STDOUT length: {len(stdout_text)}")
        logger.debug(f"STDERR length: {len(stderr_text)}")
        logger.debug("STDOUT content:")
        logger.debug(stdout_text[:1000] if stdout_text else "No STDOUT")
        logger.debug("STDERR content:")
        logger.debug(stderr_text[:1000] if stderr_text else "No STDERR")
        logger.debug("=" * 80)

        # Update status based on exit code and output analysis
        if process.returncode == 0:
            test_info.status = TestStatus.PASSED
            logger.info(f"Test passed: {test_info.test_path}")
        else:
            # Analyze the type of failure
            combined_output = stdout_text + stderr_text

            if any(
                keyword in combined_output.lower()
                for keyword in ["importerror", "modulenotfounderror", "no module named"]
            ):
                test_info.status = TestStatus.ERROR
                test_info.error_message = "Import error detected"
                logger.info(f"Test failed with import error: {test_info.test_path}")
            elif any(
                keyword in combined_output.lower()
                for keyword in ["syntaxerror", "indentationerror"]
            ):
                test_info.status = TestStatus.ERROR
                test_info.error_message = "Syntax error detected"
                logger.info(f"Test failed with syntax error: {test_info.test_path}")
            elif any(
                keyword in combined_output.lower()
                for keyword in ["assertionerror", "assert", "failed"]
            ):
                test_info.status = TestStatus.FAILED
                test_info.error_message = "Test assertion failed"
                logger.info(f"Test failed (assertions): {test_info.test_path}")
            else:
                test_info.status = TestStatus.ERROR
                test_info.error_message = (
                    f"Test execution error (code {process.returncode})"
                )
                logger.info(f"Test error (unknown): {test_info.test_path}")

        return test_info

    except Exception as e:
        test_info.status = TestStatus.ERROR
        test_info.error_message = f"Error running test: {str(e)}"
        test_info.execution_result = f"Exception during test execution: {str(e)}"
        logger.error(f"Error running test {test_info.test_path}: {str(e)}")
        logger.exception("Full exception details:")
        return test_info


async def execute_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to execute generated tests with robust environment management.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    logger.info(f"Executing tests for language: {language}")

    # Set up test environment with retry logic
    max_retries = 2
    for attempt in range(max_retries):
        success, message, env_path = setup_test_environment(
            language, force_recreate=(attempt > 0)
        )

        if success:
            logger.info(message)
            state.environment_path = env_path  # Store in state for later use
            break
        else:
            logger.warning(f"Environment setup attempt {attempt + 1} failed: {message}")
            if attempt == max_retries - 1:
                error_msg = f"Failed to set up test environment after {max_retries} attempts: {message}"
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

    # DEBUG: Log info about tests to execute
    logger.debug("TESTS TO EXECUTE:")
    for source_file, test_info in tests_to_execute.items():
        logger.debug(f"  {source_file}:")
        logger.debug(f"    Test path: {test_info.test_path}")
        logger.debug(f"    Status: {test_info.status}")
        logger.debug(f"    Has content: {bool(test_info.content)}")

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
