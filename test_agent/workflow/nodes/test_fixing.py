# test_agent/workflow/nodes/test_fixing.py - Enhanced version with better error context

"""
Enhanced test fixing that ensures ALL stderr is passed to AI agent for comprehensive error analysis.

Key improvements:
1. Always includes full stderr context for ALL error types
2. Better error categorization and context preservation
3. Handles Gemini's lack of system prompt support
4. More comprehensive error details extraction
"""

import os
import logging
import time
import asyncio
import re
import json
from typing import Dict, Any, Optional, List

from test_agent.llm import get_provider
from test_agent.workflow.state import WorkflowState, TestInfo, TestStatus, ToolCall
from .test_execution import run_test

# Try to import tool registry, but handle gracefully if not available
try:
    from test_agent.tools.registry import tool_registry

    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    tool_registry = None

# Configure logging
logger = logging.getLogger(__name__)


def extract_comprehensive_error_details(execution_result: str) -> Dict[str, str]:
    """
    Extract ALL error details from execution result for comprehensive AI analysis.

    Args:
        execution_result: Full execution result

    Returns:
        Dictionary with all error details extracted
    """
    if not execution_result:
        return {"stderr": "", "stdout": "", "full_output": "", "process_info": ""}

    # Extract stderr section
    stderr_match = re.search(
        r"STDERR:\n(.*?)(?:\n\nPROCESS INFO:|$)", execution_result, re.DOTALL
    )
    stderr = stderr_match.group(1).strip() if stderr_match else ""

    # Extract stdout section
    stdout_match = re.search(
        r"STDOUT:\n(.*?)(?:\n\nSTDERR:|$)", execution_result, re.DOTALL
    )
    stdout = stdout_match.group(1).strip() if stdout_match else ""

    # Extract process info section
    process_info_match = re.search(
        r"PROCESS INFO:\n(.*?)$", execution_result, re.DOTALL
    )
    process_info = process_info_match.group(1).strip() if process_info_match else ""

    return {
        "stderr": stderr,
        "stdout": stdout,
        "full_output": execution_result,
        "process_info": process_info,
    }


def analyze_test_error_comprehensive(execution_result: str) -> Dict[str, Any]:
    """
    Comprehensive error analysis that preserves ALL error details for AI.

    Args:
        execution_result: Test execution output

    Returns:
        Dictionary with comprehensive error analysis including ALL raw details
    """
    result = {
        "has_syntax_error": False,
        "has_import_error": False,
        "has_assertion_error": False,
        "has_exception": False,
        "has_clear_error": False,
        "test_failed_no_clear_error": False,
        "error_type": None,
        "error_message": None,
        "error_location": None,
        "error_line": None,
        "missing_dependencies": [],
        "failing_imports": [],
        "suggested_actions": [],
        # Enhanced fields for comprehensive error details
        "error_details": {},
        "comprehensive_context": "",
        "should_use_tools_first": False,
        "needs_stdout_analysis": False,
    }

    if not execution_result:
        return result

    # Extract comprehensive error details
    error_details = extract_comprehensive_error_details(execution_result)
    result["error_details"] = error_details

    # Create comprehensive context that includes EVERYTHING
    context_parts = []

    if error_details["stderr"]:
        context_parts.append(f"=== STDERR OUTPUT ===\n{error_details['stderr']}")

    if error_details["stdout"]:
        context_parts.append(f"=== STDOUT OUTPUT ===\n{error_details['stdout']}")

    if error_details["process_info"]:
        context_parts.append(f"=== PROCESS INFO ===\n{error_details['process_info']}")

    # Always include full output as fallback
    context_parts.append(f"=== FULL EXECUTION OUTPUT ===\n{execution_result}")

    result["comprehensive_context"] = "\n\n".join(context_parts)

    lines = execution_result.split("\n")

    # Check for import errors and extract missing modules
    if "ImportError" in execution_result or "ModuleNotFoundError" in execution_result:
        result["has_import_error"] = True
        result["has_clear_error"] = True
        result["error_type"] = (
            "ImportError"
            if "ImportError" in execution_result
            else "ModuleNotFoundError"
        )
        result["should_use_tools_first"] = True  # Use tools for import errors

        # Extract missing modules
        module_patterns = [
            r"No module named '([^']+)'",
            r"cannot import name '([^']+)'",
            r"ImportError: (.+)",
            r"ModuleNotFoundError: (.+)",
        ]

        for pattern in module_patterns:
            matches = re.findall(pattern, execution_result)
            for match in matches:
                if "No module named" in match:
                    module_match = re.search(r"'([^']+)'", match)
                    if module_match:
                        module_name = module_match.group(1)
                        result["missing_dependencies"].append(module_name)
                        result["suggested_actions"].append(
                            f"install_package:{module_name}"
                        )
                else:
                    result["missing_dependencies"].append(match.strip())

        # Extract failing import statements
        for line in lines:
            if line.strip().startswith(("from ", "import ")) and any(
                keyword in line for keyword in ["ImportError", "ModuleNotFoundError"]
            ):
                result["failing_imports"].append(line.strip())

    # Check for syntax errors
    elif "SyntaxError" in execution_result:
        result["has_syntax_error"] = True
        result["has_clear_error"] = True
        result["error_type"] = "SyntaxError"
        result["suggested_actions"].append("fix_syntax")

    # Check for assertion errors
    elif "AssertionError" in execution_result or "assert" in execution_result.lower():
        result["has_assertion_error"] = True
        result["has_clear_error"] = True
        result["error_type"] = "AssertionError"
        result["suggested_actions"].append("fix_assertion")

        # Try to extract specific assertion failure details
        assertion_patterns = [
            r"assert (.+)",
            r"AssertionError: (.+)",
            r"E\s+assert (.+)",  # pytest format
            r"FAILED.*assert (.+)",
        ]

        for pattern in assertion_patterns:
            matches = re.findall(pattern, execution_result, re.IGNORECASE)
            if matches:
                result["error_message"] = f"Assertion failed: {matches[0]}"
                break

    # Check for other exceptions
    elif any(
        keyword in execution_result for keyword in ["Error:", "Exception:", "Traceback"]
    ):
        result["has_exception"] = True
        result["has_clear_error"] = True
        if not result["error_type"]:
            # Try to extract specific exception type
            exception_match = re.search(r"(\w+Error): (.+)", execution_result)
            if exception_match:
                result["error_type"] = exception_match.group(1)
                result["error_message"] = exception_match.group(2)
            else:
                result["error_type"] = "Exception"

        result["suggested_actions"].append("fix_exception")

    # Check if test failed but no clear error was detected
    if not result["has_clear_error"]:
        # Look for test failure indicators in stdout/stderr
        failure_indicators = [
            "FAILED",
            "FAIL",
            "failed",
            "failure",
            "test failed",
            "0 passed",
            "1 failed",
            "2 failed",
            "3 failed",
            "4 failed",
            "5 failed",
            "Exit code: 1",
            "Return code: 1",
            "❌",
            "✗",
        ]

        # Check if any failure indicators are present
        execution_lower = execution_result.lower()
        if any(indicator in execution_lower for indicator in failure_indicators):
            result["test_failed_no_clear_error"] = True
            result["needs_stdout_analysis"] = True
            result["error_type"] = "TestFailedNoErrorDetected"
            result["error_message"] = (
                "Test failed but no clear error type detected - needs stdout analysis"
            )
            result["suggested_actions"].append("analyze_stdout_for_failure")

            logger.info(
                "Test failed but no clear error detected - will analyze stdout/stderr comprehensively"
            )

    return result


async def record_tool_call_from_llm_response(
    tool_call_data: Dict[str, Any],
    state: WorkflowState,
    test_info: TestInfo,
) -> None:
    """
    Record a tool call that was executed by the LLM provider.

    Args:
        tool_call_data: Tool call data from LLM response
        state: Workflow state
        test_info: Test information
    """
    try:
        # Extract tool information
        tool_name = tool_call_data.get("name") or tool_call_data.get(
            "function", {}
        ).get("name")
        tool_args = (
            tool_call_data.get("input")
            or tool_call_data.get("arguments")
            or tool_call_data.get("function", {}).get("arguments", {})
        )
        tool_result = tool_call_data.get("result", "Tool executed by LLM provider")

        if tool_name:
            # Create tool call record
            tool_call = ToolCall(
                tool_name=tool_name,
                tool_input=tool_args if isinstance(tool_args, dict) else {},
                tool_output=tool_result,
                timestamp=time.time(),
                success=True,  # Assume success if LLM provider executed it
            )

            # Record in state and test info
            state.record_tool_call(tool_call)
            test_info.tool_calls.append(tool_call)

            logger.info(f"Recorded tool call from LLM: {tool_name}")

    except Exception as e:
        logger.warning(f"Error recording tool call from LLM response: {str(e)}")


async def fix_test_with_comprehensive_approach(
    test_info: TestInfo,
    language: str,
    llm_provider,
    state: WorkflowState,
    max_attempts: int = 2,
) -> TestInfo:
    """
    Enhanced test fixing with comprehensive error context and tool-first approach.

    Args:
        test_info: Test information
        language: Language name
        llm_provider: LLM provider with tools bound
        state: Workflow state
        max_attempts: Maximum number of fix attempts

    Returns:
        Updated test information
    """
    if not test_info.execution_result or test_info.status not in [
        TestStatus.FAILED,
        TestStatus.ERROR,
    ]:
        return test_info

    if test_info.fix_attempts >= max_attempts:
        logger.info(
            f"Maximum fix attempts ({max_attempts}) reached for {test_info.test_path}"
        )
        return test_info

    logger.info(
        f"Starting comprehensive fix for test: {test_info.test_path} (attempt {test_info.fix_attempts + 1}/{max_attempts})"
    )

    try:
        # Read current test content
        with open(test_info.test_path, "r") as f:
            current_content = f.read()

        # Save to fix history
        if test_info.fix_history is None:
            test_info.fix_history = []
        test_info.fix_history.append(current_content)

        current_execution_result = test_info.execution_result
        tools_used_this_attempt = []

        # Iterative fixing loop
        for iteration in max_attempts:  # Max 3 iterations per attempt
            logger.info(
                f"Fix iteration {iteration + 1}/{max_attempts} for {test_info.test_path}"
            )

            # Comprehensive error analysis
            error_analysis = analyze_test_error_comprehensive(current_execution_result)
            logger.info(
                f"Error analysis: {error_analysis.get('error_type')} - Should use tools first: {error_analysis.get('should_use_tools_first')}"
            )

            # Create unified prompt that works for all LLM providers
            # For Gemini, we'll combine system and user prompts
            if error_analysis.get("should_use_tools_first"):
                # For import errors - tool-focused approach
                base_system_content = """You are an expert test fixing agent with access to tools for resolving import and dependency issues.

Available tools:
- install_python_package: Install missing Python packages
- fix_import_statement: Analyze and suggest import fixes  
- create_mock_dependency: Create mocks for unavailable dependencies

Your goal is to fix import/module errors by using tools to install missing dependencies or create appropriate mocks.

Focus on using tools first, then provide code fixes if needed."""

            elif error_analysis.get("test_failed_no_clear_error") or error_analysis.get(
                "needs_stdout_analysis"
            ):
                # For tests that failed but no clear error detected - comprehensive analysis approach
                base_system_content = """You are an expert test fixing agent specializing in analyzing test failures where no clear error is apparent.

The test failed but there are no obvious ImportError, SyntaxError, AssertionError, or Exception messages. This means you need to carefully analyze the STDOUT and STDERR output to understand why the test failed.

Common causes of "silent" test failures:
1. Test logic issues (wrong assertions, incorrect expected values)
2. Missing test data or setup
3. Tests expecting different behavior than what the code provides
4. Configuration or environment issues
5. Silent failures in the test framework
6. Tests that expect specific output but get different output
7. Missing or incorrect mocks/fixtures
8. Race conditions or timing issues

Your approach:
1. Carefully examine the STDOUT output for test execution results
2. Look for any test framework output (pytest, unittest) that indicates what failed
3. Analyze any assertion failures that might be buried in the output
4. Check for missing expected values or incorrect actual values
5. Look for test setup/teardown issues
6. Examine the test code logic for potential issues

Focus on providing specific code fixes based on what you find in the comprehensive output analysis."""

            else:
                # For assertion/syntax/other errors - analysis-focused approach
                base_system_content = """You are an expert test fixing agent. I will provide you with comprehensive error information including ALL raw output.

Your goal is to analyze the specific error details and fix the test code accordingly.

For assertion errors: Look at the specific assertion that failed and understand why it failed.
For syntax errors: Identify the exact syntax issue and fix it.
For other exceptions: Analyze the stack trace and exception details to understand the root cause.

Focus on providing accurate code fixes based on the specific error details provided."""

            # Create detailed user prompt with COMPREHENSIVE error context
            if error_analysis.get("test_failed_no_clear_error") or error_analysis.get(
                "needs_stdout_analysis"
            ):
                # Special handling for tests that failed without clear errors
                user_prompt = f"""I need to fix a failing test that has NO CLEAR ERROR MESSAGES. The test failed but there are no obvious ImportError, SyntaxError, AssertionError, or Exception traces.

Test file: {os.path.basename(test_info.test_path)}
Source file: {os.path.basename(test_info.source_file)}
Fix iteration: {iteration + 1}/3
Fix attempt: {test_info.fix_attempts + 1}/{max_attempts}

⚠️  CRITICAL: This test failed WITHOUT clear error indicators. You MUST analyze the STDOUT and STDERR output comprehensively to understand WHY it failed.

COMPREHENSIVE TEST OUTPUT (ANALYZE CAREFULLY):
{error_analysis.get('comprehensive_context')}

ANALYSIS SUMMARY:
- Error type detected: {error_analysis.get('error_type')}
- No clear ImportError, SyntaxError, or Exception detected
- Test appears to have failed silently or with subtle issues
- May require STDOUT analysis to identify the failure reason

CURRENT TEST CODE:
```python
{current_content}
```

Tools already used this attempt: {tools_used_this_attempt}

🔍 IMPORTANT ANALYSIS TASKS:
1. Examine the STDOUT output for test framework messages (pytest/unittest output)
2. Look for assertion failures that might be buried in the output
3. Check for expected vs actual value mismatches
4. Identify any test setup or logic issues
5. Look for missing imports, fixtures, or test data
6. Analyze test execution flow and identify where it might be failing

Based on your comprehensive analysis of the output above, provide the corrected test code that addresses the specific failure you identify. Focus on understanding WHY the test failed by analyzing the execution output."""

            else:
                # Standard handling for tests with clear errors
                user_prompt = f"""I need to fix a failing test. Here are the comprehensive details:

Test file: {os.path.basename(test_info.test_path)}
Source file: {os.path.basename(test_info.source_file)}
Fix iteration: {iteration + 1}/3
Fix attempt: {test_info.fix_attempts + 1}/{max_attempts}

COMPREHENSIVE ERROR CONTEXT:
{error_analysis.get('comprehensive_context')}

ERROR ANALYSIS:
- Error type: {error_analysis.get('error_type')}
- Has import error: {error_analysis.get('has_import_error')}
- Has syntax error: {error_analysis.get('has_syntax_error')}
- Has assertion error: {error_analysis.get('has_assertion_error')}
- Has exception: {error_analysis.get('has_exception')}
- Missing dependencies: {error_analysis.get('missing_dependencies')}
- Suggested actions: {error_analysis.get('suggested_actions')}

CURRENT TEST CODE:
```python
{current_content}
```

Tools already used this attempt: {tools_used_this_attempt}

Please analyze the comprehensive error context above and {'use appropriate tools to fix the import/dependency issues' if error_analysis.get('should_use_tools_first') else 'provide the corrected test code based on the detailed error analysis'}.

{'Focus on installing missing packages or creating mocks for import errors.' if error_analysis.get('should_use_tools_first') else 'Focus on understanding the specific failure from the stderr/stdout output and fixing the test logic, assertions, or setup accordingly.'}
"""

            # Check if this is Gemini provider (which doesn't support system prompts)
            is_gemini = (
                hasattr(llm_provider, "provider_name")
                and llm_provider.provider_name == "gemini"
            )

            if is_gemini:
                # For Gemini, combine system and user prompts
                combined_prompt = f"{base_system_content}\n\n{user_prompt}"
                system_prompt = None
            else:
                # For other providers, use separate system prompt
                combined_prompt = user_prompt
                system_prompt = base_system_content

            # Use the LLM with tools to analyze and fix
            if hasattr(llm_provider, "generate_with_tools") and error_analysis.get(
                "should_use_tools_first"
            ):
                # For import errors, use tools
                tools = None
                if hasattr(llm_provider, "get_bound_tools"):
                    tools = llm_provider.get_bound_tools()

                response = await llm_provider.generate_with_tools(
                    prompt=combined_prompt,
                    tools=tools,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
            else:
                # For assertion/exception errors, focus on code generation
                if error_analysis.get("should_use_tools_first"):
                    enhanced_prompt = llm_provider.create_tool_calling_prompt(
                        combined_prompt,
                        "Focus on using tools to fix import errors and missing dependencies.",
                    )
                    response = await llm_provider.generate(
                        enhanced_prompt,
                        system_prompt=system_prompt,
                        temperature=0.1,
                        use_tools=True,
                    )
                else:
                    # Direct code fixing for assertion/exception errors
                    response = await llm_provider.generate(
                        combined_prompt,
                        system_prompt=system_prompt,
                        temperature=0.1,
                    )

            # Process the response and record tool calls
            tools_used_in_iteration = []

            if isinstance(response, dict):
                # Handle structured response with potential tool calls
                if "tool_calls" in response and response["tool_calls"]:
                    logger.info(
                        f"LLM executed {len(response['tool_calls'])} tool calls"
                    )

                    # Record tool calls
                    for tool_call_data in response["tool_calls"]:
                        await record_tool_call_from_llm_response(
                            tool_call_data, state, test_info
                        )

                        tool_name = tool_call_data.get("name") or tool_call_data.get(
                            "function", {}
                        ).get("name")
                        if tool_name:
                            tools_used_in_iteration.append(tool_name)
                            tools_used_this_attempt.append(tool_name)

                        # Track package installations
                        if tool_name == "install_python_package":
                            tool_args = (
                                tool_call_data.get("input")
                                or tool_call_data.get("arguments")
                                or tool_call_data.get("function", {}).get(
                                    "arguments", {}
                                )
                            )
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    pass

                            if isinstance(tool_args, dict):
                                package_name = tool_args.get("package_name", "")
                                if (
                                    package_name
                                    and package_name
                                    not in test_info.dependencies_installed
                                ):
                                    test_info.dependencies_installed.append(
                                        package_name
                                    )

            # Process response for code fixes
            content = ""
            if isinstance(response, dict):
                content = (
                    response.get("content", "")
                    or response.get("message", "")
                    or str(response)
                )
            else:
                content = str(response)

            # If tools were used, re-run the test to see if it's fixed
            if tools_used_in_iteration:
                logger.info(
                    f"Tools used: {tools_used_in_iteration}. Re-running test..."
                )

                # Wait a moment for installations to take effect
                await asyncio.sleep(2)

                # Re-run the test with current content
                temp_test_info = TestInfo(
                    source_file=test_info.source_file,
                    test_path=test_info.test_path,
                    content=current_content,
                    status=TestStatus.PENDING,
                )

                updated_test_info = await run_test(
                    temp_test_info, language, state.environment_path
                )

                if updated_test_info.status == TestStatus.PASSED:
                    logger.info(f"Test FIXED by tools in iteration {iteration + 1}!")
                    test_info.status = TestStatus.FIXED
                    test_info.execution_result = updated_test_info.execution_result
                    test_info.fix_attempts += 1
                    return test_info
                else:
                    logger.info(
                        f"Test still failing after iteration {iteration + 1}. Trying next iteration..."
                    )
                    current_execution_result = updated_test_info.execution_result
                    test_info.execution_result = current_execution_result

            # Check for code fixes in the response
            code_matches = re.findall(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
            if code_matches:
                fixed_code = max(code_matches, key=len).strip()

                # Check if content actually changed
                if fixed_code.strip() != current_content.strip():
                    logger.info(f"LLM provided code fixes in iteration {iteration + 1}")

                    # Write fixed code
                    with open(test_info.test_path, "w") as f:
                        f.write(fixed_code)

                    current_content = fixed_code
                    test_info.content = fixed_code

                    # Re-run the test with new code
                    temp_test_info = TestInfo(
                        source_file=test_info.source_file,
                        test_path=test_info.test_path,
                        content=fixed_code,
                        status=TestStatus.PENDING,
                    )

                    updated_test_info = await run_test(
                        temp_test_info, language, state.environment_path
                    )

                    if updated_test_info.status == TestStatus.PASSED:
                        logger.info(
                            f"Test FIXED by code changes in iteration {iteration + 1}!"
                        )
                        test_info.status = TestStatus.FIXED
                        test_info.execution_result = updated_test_info.execution_result
                        test_info.fix_attempts += 1
                        return test_info
                    else:
                        logger.info(
                            f"Test still failing after code changes in iteration {iteration + 1}"
                        )
                        current_execution_result = updated_test_info.execution_result
                        test_info.execution_result = current_execution_result
                else:
                    logger.info(
                        f"No significant changes suggested by LLM in iteration {iteration + 1}"
                    )
                    break
            else:
                logger.info(
                    f"No code fixes provided by LLM in iteration {iteration + 1}"
                )
                if not tools_used_in_iteration:
                    break  # No tools and no code fixes, exit loop

        # If we get here, the test is still failing after all iterations
        logger.info("Test still failing after all iterations. Moving to next attempt.")
        test_info.fix_attempts += 1
        test_info.error_message = f"Fix attempt {test_info.fix_attempts} failed after {iteration + 1} iterations"

        return test_info

    except Exception as e:
        logger.error(f"Error in comprehensive fix for {test_info.test_path}: {str(e)}")
        test_info.error_message = f"Error in comprehensive fix: {str(e)}"
        test_info.fix_attempts += 1
        return test_info


async def fix_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to fix failing tests using comprehensive error analysis.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    logger.info(f"Starting comprehensive test fixing phase for language: {language}")

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

    # Initialize tools if not already done
    if not state.tools_registry_initialized:
        try:
            if TOOLS_AVAILABLE:
                from test_agent.tools.registry import register_default_tools

                register_default_tools(state.environment_path)
                state.tools_registry_initialized = True
                state.memory.tool_usage.tools_available = list(
                    tool_registry.get_all_tools().keys()
                )
                logger.info(
                    f"Initialized tools registry with {len(tool_registry.get_all_tools())} tools"
                )
            else:
                logger.warning(
                    "Tool registry not available - test fixing will be limited"
                )
                state.warnings.append(
                    {
                        "phase": "test_fixing",
                        "error": "Tool registry not available - limited test fixing capabilities",
                        "type": "tools_unavailable",
                    }
                )
        except Exception as e:
            logger.error(f"Failed to initialize tools: {str(e)}")
            state.warnings.append(
                {
                    "phase": "test_fixing",
                    "error": f"Failed to initialize tools: {str(e)}",
                    "type": "tool_init_failed",
                }
            )

    # Get LLM provider and bind tools if available
    llm_provider = get_provider(state.llm.provider)

    if TOOLS_AVAILABLE and tool_registry:
        llm_provider.bind_tools(tool_registry)
        logger.info("Bound tools to LLM provider")
    else:
        logger.warning("Tools not available - LLM will operate without tools")

    # Set up environment if needed
    if not state.environment_path:
        from .test_execution import setup_test_environment

        success, message, env_path = setup_test_environment(language)
        if success:
            state.environment_path = env_path
            logger.info(f"Set up test environment: {message}")
        else:
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

    start_time = time.time()

    # Get tests to fix
    tests_to_fix = {
        source_file: test_info
        for source_file, test_info in state.tests.items()
        if test_info.status in [TestStatus.FAILED, TestStatus.ERROR]
    }

    logger.info(f"Found {len(tests_to_fix)} tests that need fixing")

    if not tests_to_fix:
        logger.info("No tests need fixing, proceeding to completion")
        state.current_phase = "test_fixing"
        state.next_phase = "complete"
        return state

    # Fix tests sequentially for better debugging and tool usage tracking
    for source_file, test_info in tests_to_fix.items():
        logger.info(f"Fixing test for {os.path.basename(source_file)}")

        try:
            fixed_test_info = await fix_test_with_comprehensive_approach(
                test_info, language, llm_provider, state
            )
            state.tests[source_file] = fixed_test_info

            # Add delay between tests to avoid rate limiting
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Error fixing test for {source_file}: {str(e)}")
            test_info.error_message = f"Fix processing error: {str(e)}"
            test_info.status = TestStatus.ERROR
            state.tests[source_file] = test_info

    # Calculate time taken
    time_taken = time.time() - start_time

    # Update state with final counts
    passed = len([t for t in state.tests.values() if t.status == TestStatus.PASSED])
    fixed = len([t for t in state.tests.values() if t.status == TestStatus.FIXED])
    failed = len([t for t in state.tests.values() if t.status == TestStatus.FAILED])
    error = len([t for t in state.tests.values() if t.status == TestStatus.ERROR])
    skipped = len([t for t in state.tests.values() if t.status == TestStatus.SKIPPED])

    state.successful_tests = passed + fixed
    state.failed_tests = failed + error
    state.fixed_tests = fixed

    # Log summary with tool usage
    tool_summary = state.get_tool_usage_summary()
    logger.info(f"Comprehensive test fixing complete in {time_taken:.2f}s")
    logger.info(
        f"Test results - Passed: {passed}, Fixed: {fixed}, Failed: {failed}, Error: {error}, Skipped: {skipped}"
    )
    logger.info(
        f"Tool usage - Total calls: {tool_summary['total_calls']}, Success rate: {tool_summary['success_rate']:.1f}%"
    )
    logger.info(f"Most used tool: {tool_summary['most_used_tool']}")

    # Set next phase
    state.current_phase = "test_fixing"
    state.next_phase = "complete"

    return state
