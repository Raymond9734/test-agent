# test_agent/tools/test_tools.py

import os
import re
import logging
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


def run_test_command(
    command: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """
    Run a test command and return the results.

    Args:
        command: Command to run as a list of strings
        cwd: Working directory
        env: Environment variables
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        Dictionary with test command results
    """
    logger.info(f"Running test command: {' '.join(command)}")

    result = {
        "command": command,
        "success": False,
        "return_code": None,
        "stdout": None,
        "stderr": None,
        "error": None,
        "timed_out": False,
    }

    try:
        # Set up environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Run the command
        if capture_output:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=process_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
        else:
            process = subprocess.run(
                command,
                cwd=cwd,
                env=process_env,
                timeout=timeout,
            )

        result["return_code"] = process.returncode
        result["success"] = process.returncode == 0

    except subprocess.TimeoutExpired as e:
        result["error"] = f"Command timed out after {timeout} seconds"
        result["timed_out"] = True
        logger.warning(f"Test command timed out: {' '.join(command)}")
        logger.error(f"Error: {e}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error running test command: {str(e)}")

    return result


def parse_test_results(test_output: str, format: str = "auto") -> Dict[str, Any]:
    """
    Parse test results from the output of a test command.

    Args:
        test_output: Output from test command
        format: Format of the test output ("auto", "pytest", "junit", "go")

    Returns:
        Dictionary with parsed test results
    """
    # Determine format if auto
    if format == "auto":
        if "pytest" in test_output.lower():
            format = "pytest"
        elif "<testsuites>" in test_output or "<testsuite>" in test_output:
            format = "junit"
        elif "=== RUN" in test_output:
            format = "go"
        else:
            # Default to pytest
            format = "pytest"

    if format == "pytest":
        return _parse_pytest_output(test_output)
    elif format == "junit":
        return _parse_junit_output(test_output)
    elif format == "go":
        return _parse_go_output(test_output)
    else:
        logger.warning(f"Unknown test output format: {format}")
        return {
            "success": False,
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "tests": [],
            "error": f"Unknown test output format: {format}",
        }


def _parse_pytest_output(output: str) -> Dict[str, Any]:
    """
    Parse pytest output.

    Args:
        output: pytest output

    Returns:
        Dictionary with parsed results
    """
    result = {
        "success": False,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests": [],
    }

    # Extract test results
    # Example: "2 passed, 1 failed, 3 skipped"
    summary_match = re.search(
        r"(\d+) passed(?:, )?(\d+)? failed(?:, )?(\d+)? skipped", output
    )
    if summary_match:
        passed = int(summary_match.group(1) or 0)
        failed = int(summary_match.group(2) or 0)
        skipped = int(summary_match.group(3) or 0)

        result["tests_passed"] = passed
        result["tests_failed"] = failed
        result["tests_skipped"] = skipped
        result["tests_total"] = passed + failed + skipped
        result["success"] = failed == 0

    # Extract individual test results
    test_pattern = r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\s+(.+?)(?:\[|$)"
    test_matches = re.finditer(test_pattern, output)

    for match in test_matches:
        status = match.group(1)
        test_name = match.group(2).strip()

        test_result = {
            "name": test_name,
            "status": status.lower(),
            "message": None,
        }

        # Try to extract error message for failed tests
        if status in ["FAILED", "ERROR"]:
            # Look for error message after the test name
            error_match = re.search(
                rf"{re.escape(test_name)}.*?\n(.*?)(?:\n\n|\n=====|$)",
                output,
                re.DOTALL,
            )
            if error_match:
                test_result["message"] = error_match.group(1).strip()

        result["tests"].append(test_result)

    return result


def _parse_junit_output(output: str) -> Dict[str, Any]:
    """
    Parse JUnit XML output.

    Args:
        output: JUnit XML output

    Returns:
        Dictionary with parsed results
    """
    result = {
        "success": False,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests": [],
    }

    try:
        # Parse XML
        root = ET.fromstring(output)

        # Find all test cases
        test_cases = root.findall(".//testcase")

        for test_case in test_cases:
            test_name = test_case.get("name", "")
            class_name = test_case.get("classname", "")
            full_name = f"{class_name}.{test_name}" if class_name else test_name

            # Check status
            failure = test_case.find("failure")
            error = test_case.find("error")
            skipped = test_case.find("skipped")

            if failure is not None:
                status = "failed"
                message = failure.get("message", "") or failure.text
                result["tests_failed"] += 1
            elif error is not None:
                status = "error"
                message = error.get("message", "") or error.text
                result["tests_failed"] += 1
            elif skipped is not None:
                status = "skipped"
                message = skipped.get("message", "") or skipped.text
                result["tests_skipped"] += 1
            else:
                status = "passed"
                message = None
                result["tests_passed"] += 1

            result["tests"].append(
                {
                    "name": full_name,
                    "status": status,
                    "message": message,
                }
            )

        result["tests_total"] = len(test_cases)
        result["success"] = result["tests_failed"] == 0

    except Exception as e:
        logger.error(f"Error parsing JUnit XML: {str(e)}")
        result["error"] = f"Error parsing JUnit XML: {str(e)}"

    return result


def _parse_go_output(output: str) -> Dict[str, Any]:
    """
    Parse Go test output.

    Args:
        output: Go test output

    Returns:
        Dictionary with parsed results
    """
    result = {
        "success": False,
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests": [],
    }

    # Extract test results
    # Example: "ok      package/path    0.015s"
    # Example: "FAIL    package/path    0.015s"

    # Track tests
    current_test = None
    tests = {}

    # Parse line by line
    for line in output.splitlines():
        # Check for test start
        start_match = re.match(r"=== RUN\s+(\S+)", line)
        if start_match:
            test_name = start_match.group(1)
            current_test = test_name
            tests[test_name] = {
                "name": test_name,
                "status": "unknown",
                "message": [],
            }
            continue

        # Check for test pass
        pass_match = re.match(r"--- PASS:\s+(\S+)", line)
        if pass_match:
            test_name = pass_match.group(1)
            if test_name in tests:
                tests[test_name]["status"] = "passed"
            result["tests_passed"] += 1
            continue

        # Check for test fail
        fail_match = re.match(r"--- FAIL:\s+(\S+)", line)
        if fail_match:
            test_name = fail_match.group(1)
            if test_name in tests:
                tests[test_name]["status"] = "failed"
            result["tests_failed"] += 1
            continue

        # Check for test skip
        skip_match = re.match(r"--- SKIP:\s+(\S+)", line)
        if skip_match:
            test_name = skip_match.group(1)
            if test_name in tests:
                tests[test_name]["status"] = "skipped"
            result["tests_skipped"] += 1
            continue

        # Collect output for current test
        if current_test and current_test in tests:
            tests[current_test]["message"].append(line)

    # Process tests
    for test_name, test_info in tests.items():
        if test_info["status"] == "unknown":
            # Assume it passed if we didn't see a result
            test_info["status"] = "passed"
            result["tests_passed"] += 1

        # Join message lines
        test_info["message"] = "\n".join(test_info["message"]).strip()

        # Add to results
        result["tests"].append(
            {
                "name": test_info["name"],
                "status": test_info["status"],
                "message": test_info["message"] if test_info["message"] else None,
            }
        )

    # Set totals
    result["tests_total"] = len(tests)
    result["success"] = result["tests_failed"] == 0 and result["tests_total"] > 0

    # Check for overall success/failure
    if "FAIL" in output:
        result["success"] = False

    return result


def check_test_coverage(
    project_dir: str,
    test_dir: Optional[str] = None,
    language: str = "python",
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check test coverage for a project.

    Args:
        project_dir: Project directory
        test_dir: Test directory (if None, uses default for language)
        language: Programming language
        exclude_patterns: Patterns to exclude from coverage

    Returns:
        Dictionary with coverage results
    """
    result = {
        "success": False,
        "total_coverage": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "covered_branches": 0,
        "total_branches": 0,
        "files": [],
        "error": None,
    }

    try:
        if language.lower() == "python":
            return _check_python_coverage(project_dir, test_dir, exclude_patterns)
        elif language.lower() == "go":
            return _check_go_coverage(project_dir, test_dir, exclude_patterns)
        else:
            result["error"] = f"Coverage checking not implemented for {language}"
            return result
    except Exception as e:
        logger.error(f"Error checking coverage: {str(e)}")
        result["error"] = f"Error checking coverage: {str(e)}"
        return result


def _check_python_coverage(
    project_dir: str,
    test_dir: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check test coverage for a Python project using coverage.py.

    Args:
        project_dir: Project directory
        test_dir: Test directory
        exclude_patterns: Patterns to exclude

    Returns:
        Dictionary with coverage results
    """
    result = {
        "success": False,
        "total_coverage": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "covered_branches": 0,
        "total_branches": 0,
        "files": [],
        "error": None,
    }

    try:
        # Check if coverage is installed
        try:
            import coverage
        except ImportError:
            result["error"] = "coverage.py is not installed"
            return result

        # Set up coverage
        cov = coverage.Coverage(
            source=[project_dir],
            omit=exclude_patterns,
        )

        # Determine test command
        if test_dir is None:
            test_dir = os.path.join(project_dir, "tests")

        if not os.path.exists(test_dir):
            result["error"] = f"Test directory not found: {test_dir}"
            return result

        # Run tests with coverage
        cov.start()

        # Import and run pytest
        import pytest

        pytest.main(["-xvs", test_dir])

        cov.stop()
        cov.save()

        # Get coverage data
        coverage_data = cov.get_data()
        total_lines = 0
        covered_lines = 0

        # Get file coverage
        for file_path in coverage_data.measured_files():
            rel_path = os.path.relpath(file_path, project_dir)

            # Skip files in excluded patterns
            if exclude_patterns and any(
                re.match(pattern, rel_path) for pattern in exclude_patterns
            ):
                continue

            # Get line coverage for file
            file_lines = coverage_data.lines(file_path)
            file_missing = coverage_data.missing_lines(file_path)

            file_total_lines = len(file_lines)
            file_covered_lines = file_total_lines - len(file_missing)

            file_coverage = 0.0
            if file_total_lines > 0:
                file_coverage = (file_covered_lines / file_total_lines) * 100.0

            result["files"].append(
                {
                    "path": rel_path,
                    "coverage": file_coverage,
                    "covered_lines": file_covered_lines,
                    "total_lines": file_total_lines,
                }
            )

            total_lines += file_total_lines
            covered_lines += file_covered_lines

        # Calculate overall coverage
        if total_lines > 0:
            result["line_coverage"] = (covered_lines / total_lines) * 100.0
            result["total_coverage"] = result["line_coverage"]

        result["covered_lines"] = covered_lines
        result["total_lines"] = total_lines
        result["success"] = True

    except Exception as e:
        logger.error(f"Error checking Python coverage: {str(e)}")
        result["error"] = f"Error checking Python coverage: {str(e)}"

    return result


def _check_go_coverage(
    project_dir: str,
    test_dir: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check test coverage for a Go project.

    Args:
        project_dir: Project directory
        test_dir: Test directory
        exclude_patterns: Patterns to exclude

    Returns:
        Dictionary with coverage results
    """
    result = {
        "success": False,
        "total_coverage": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "covered_lines": 0,
        "total_lines": 0,
        "covered_branches": 0,
        "total_branches": 0,
        "files": [],
        "error": None,
    }

    try:
        # Determine test command
        if test_dir is None:
            # Use whole project
            test_dir = "./..."

        # Create coverage output file
        coverage_file = os.path.join(project_dir, "coverage.out")

        # Run go test with coverage
        cmd = ["go", "test", "-coverprofile", coverage_file, test_dir]

        process = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            result["error"] = f"Error running go test: {process.stderr}"
            return result

        # Check if coverage file was created
        if not os.path.exists(coverage_file):
            result["error"] = "Coverage file not created"
            return result

        # Parse coverage file
        with open(coverage_file, "r") as f:
            coverage_data = f.readlines()

        # First line is mode, skip it
        coverage_data = coverage_data[1:]

        # Process each line
        total_statements = 0
        covered_statements = 0

        for line in coverage_data:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            file_path = parts[0]
            coverage_info = parts[1]
            coverage_percent = float(parts[2].rstrip("%"))

            # Extract covered/total from coverage_info (e.g. "10/20")
            coverage_parts = coverage_info.split("/")
            if len(coverage_parts) == 2:
                file_covered = int(coverage_parts[0])
                file_total = int(coverage_parts[1])

                total_statements += file_total
                covered_statements += file_covered

                # Get relative path
                rel_path = os.path.relpath(file_path, project_dir)

                # Skip excluded patterns
                if exclude_patterns and any(
                    re.match(pattern, rel_path) for pattern in exclude_patterns
                ):
                    continue

                result["files"].append(
                    {
                        "path": rel_path,
                        "coverage": coverage_percent,
                        "covered_lines": file_covered,
                        "total_lines": file_total,
                    }
                )

        # Calculate overall coverage
        if total_statements > 0:
            result["line_coverage"] = (covered_statements / total_statements) * 100.0
            result["total_coverage"] = result["line_coverage"]

        result["covered_lines"] = covered_statements
        result["total_lines"] = total_statements
        result["success"] = True

        # Clean up coverage file
        if os.path.exists(coverage_file):
            os.remove(coverage_file)

    except Exception as e:
        logger.error(f"Error checking Go coverage: {str(e)}")
        result["error"] = f"Error checking Go coverage: {str(e)}"

    return result
