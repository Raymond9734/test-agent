# test_agent/workflow/nodes/test_fixing.py - Fixed iterative test fixing

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
        "failing_imports": [],
        "suggested_actions": [],
    }

    if not execution_result:
        return result

    lines = execution_result.split("\n")

    # Check for import errors and extract missing modules
    if "ImportError" in execution_result or "ModuleNotFoundError" in execution_result:
        result["has_import_error"] = True
        result["error_type"] = (
            "ImportError"
            if "ImportError" in execution_result
            else "ModuleNotFoundError"
        )

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
    if "SyntaxError" in execution_result:
        result["has_syntax_error"] = True
        result["error_type"] = "SyntaxError"
        result["suggested_actions"].append("fix_syntax")

    # Check for assertion errors
    elif "AssertionError" in execution_result:
        result["has_assertion_error"] = True
        result["error_type"] = "AssertionError"
        result["suggested_actions"].append("fix_assertion")

    # Check for other exceptions
    elif any(
        keyword in execution_result for keyword in ["Error:", "Exception:", "Traceback"]
    ):
        result["has_exception"] = True
        if not result["error_type"]:
            result["error_type"] = "Exception"
        result["suggested_actions"].append("fix_exception")

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


async def fix_test_with_iterative_approach(
    test_info: TestInfo,
    language: str,
    llm_provider,
    state: WorkflowState,
    max_attempts: int = 5,
) -> TestInfo:
    """
    Attempt to fix a failing test using an iterative approach with tools and LLM.

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
        f"Starting iterative fix for test: {test_info.test_path} (attempt {test_info.fix_attempts + 1}/{max_attempts})"
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
        for iteration in range(3):  # Max 3 iterations per attempt
            logger.info(f"Fix iteration {iteration + 1}/3 for {test_info.test_path}")

            # Analyze the current error
            error_analysis = analyze_test_error(current_execution_result)
            logger.info(
                f"Error analysis: {error_analysis.get('error_type')} - {error_analysis.get('missing_dependencies')}"
            )

            # Create system prompt that encourages tool usage
            system_prompt = """
You are an expert test fixing agent with access to tools that can help resolve test failures.

Available tools:
- install_python_package: Install missing Python packages
- fix_import_statement: Analyze and suggest import fixes  
- create_mock_dependency: Create mocks for unavailable dependencies
- run_test_command: Run test commands to verify fixes

Your goal is to fix the failing test by using tools to resolve underlying issues.

IMPORTANT: 
- For ImportError/ModuleNotFoundError: Use install_python_package to install missing modules
- After using tools, I will re-run the test to see if it's fixed
- Only provide code fixes if tools alone don't solve the problem
"""

            # Create detailed prompt with error analysis
            user_prompt = f"""
I need to fix a failing test. Here are the details:

Test file: {os.path.basename(test_info.test_path)}
Source file: {os.path.basename(test_info.source_file)}
Fix iteration: {iteration + 1}/3

Error Analysis:
- Error type: {error_analysis.get('error_type')}
- Has import error: {error_analysis.get('has_import_error')}
- Has syntax error: {error_analysis.get('has_syntax_error')}
- Missing dependencies: {error_analysis.get('missing_dependencies')}
- Suggested actions: {error_analysis.get('suggested_actions')}

Test execution output:
\`\`\`
{current_execution_result}
\`\`\`

Tools already used this attempt: {tools_used_this_attempt}

Please analyze the error and use appropriate tools to fix it. Focus on:
1. Installing missing packages for ImportError/ModuleNotFoundError
2. Creating mocks for external dependencies that can't be installed
3. Fixing import statements

I will re-run the test after you use tools to see if the problem is resolved.
"""

            # Use the LLM with tools to analyze and fix
            if hasattr(llm_provider, "generate_with_tools"):
                # Get available tools
                tools = None
                if hasattr(llm_provider, "get_bound_tools"):
                    tools = llm_provider.get_bound_tools()

                # Use tool-enabled generation
                response = await llm_provider.generate_with_tools(
                    prompt=user_prompt,
                    tools=tools,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
            else:
                # Fallback to regular generation with tool-enabled prompt
                enhanced_prompt = llm_provider.create_tool_calling_prompt(
                    user_prompt,
                    "Focus on using tools to fix import errors and missing dependencies.",
                )
                response = await llm_provider.generate(
                    enhanced_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    use_tools=True,
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
            else:
                # No tools used - check if LLM provided code fixes
                content = ""
                if isinstance(response, dict):
                    content = (
                        response.get("content", "")
                        or response.get("message", "")
                        or str(response)
                    )
                else:
                    content = str(response)

                # Extract code block from content
                code_matches = re.findall(
                    r"\`\`\`(?:python)?\n(.*?)\`\`\`", content, re.DOTALL
                )
                if code_matches:
                    fixed_code = max(code_matches, key=len).strip()

                    # Check if content actually changed
                    if fixed_code.strip() != current_content.strip():
                        logger.info(
                            f"LLM provided code fixes in iteration {iteration + 1}"
                        )

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
                            test_info.execution_result = (
                                updated_test_info.execution_result
                            )
                            test_info.fix_attempts += 1
                            return test_info
                        else:
                            logger.info(
                                f"Test still failing after code changes in iteration {iteration + 1}"
                            )
                            current_execution_result = (
                                updated_test_info.execution_result
                            )
                            test_info.execution_result = current_execution_result
                    else:
                        logger.info(
                            f"No changes suggested by LLM in iteration {iteration + 1}"
                        )
                        break
                else:
                    logger.info(
                        f"No code fixes provided by LLM in iteration {iteration + 1}"
                    )
                    if not tools_used_in_iteration:
                        break  # No tools and no code fixes, exit loop

        # If we get here, the test is still failing after all iterations
        logger.info(f"Test still failing after all iterations. Moving to next attempt.")
        test_info.fix_attempts += 1
        test_info.error_message = f"Fix attempt {test_info.fix_attempts} failed after {iteration + 1} iterations"

        return test_info

    except Exception as e:
        logger.error(f"Error in iterative fix for {test_info.test_path}: {str(e)}")
        test_info.error_message = f"Error in iterative fix: {str(e)}"
        test_info.fix_attempts += 1
        return test_info


async def fix_tests(state: WorkflowState) -> WorkflowState:
    """
    Node to fix failing tests using iterative LLM with tools approach.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    language = state.project.language

    logger.info(f"Starting iterative test fixing phase for language: {language}")

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
            fixed_test_info = await fix_test_with_iterative_approach(
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
    logger.info(f"Iterative test fixing complete in {time_taken:.2f}s")
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
