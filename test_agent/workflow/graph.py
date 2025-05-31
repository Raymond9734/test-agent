# test_agent/workflow/graph.py - Enhanced with tool integration

import logging
from typing import Callable, Awaitable

# Import conditional to handle both LangGraph and LangChain
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    from langchain.graphs import StateGraph, END

from .state import WorkflowState
from .nodes import (
    initialize_workflow,
    detect_project_language,
    analyze_project,
    analyze_files,
    generate_test_paths,
    generate_tests,
    execute_tests,
    fix_tests,
    complete_workflow,
    handle_error,
)

# Configure logging
logger = logging.getLogger(__name__)

# Type for node function
NodeFn = Callable[[WorkflowState], Awaitable[WorkflowState]]


def initialize_tools_in_state(state: WorkflowState) -> WorkflowState:
    """
    Initialize tools in the workflow state if not already done.

    Args:
        state: Workflow state

    Returns:
        Updated workflow state
    """
    if not state.tools_registry_initialized:
        try:
            # Check if registry is available
            from test_agent.tools import REGISTRY_AVAILABLE

            if REGISTRY_AVAILABLE:
                from test_agent.tools.registry import (
                    register_default_tools,
                    tool_registry,
                )

                # Register default tools
                register_default_tools()

                # Update state
                state.tools_registry_initialized = True
                state.memory.tool_usage.tools_available = list(
                    tool_registry.get_all_tools().keys()
                )

                logger.info(
                    f"Initialized tools registry with {len(tool_registry.get_all_tools())} tools"
                )
                logger.info(
                    f"Available tools: {', '.join(state.memory.tool_usage.tools_available)}"
                )
            else:
                logger.warning("Tool registry not available - running without tools")
                state.warnings.append(
                    {
                        "phase": "tool_initialization",
                        "error": "Tool registry not available",
                        "type": "tool_registry_unavailable",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to initialize tools: {str(e)}")
            state.warnings.append(
                {
                    "phase": "tool_initialization",
                    "error": f"Failed to initialize tools: {str(e)}",
                    "type": "tool_init_failed",
                }
            )

    return state


def setup_llm_with_tools(state: WorkflowState) -> WorkflowState:
    """
    Set up LLM provider with tools if configured.

    Args:
        state: Workflow state

    Returns:
        Updated workflow state
    """
    if state.llm and state.llm.provider and state.tools_registry_initialized:
        try:
            from test_agent.llm import get_provider
            from test_agent.tools.registry import tool_registry

            # Get LLM provider
            llm_provider = get_provider(state.llm.provider)

            # Bind tools to the provider
            llm_provider.bind_tools(tool_registry)

            # Mark tools as enabled
            state.llm.tools_enabled = True

            logger.info(
                f"Bound {len(tool_registry.get_all_tools())} tools to {state.llm.provider} provider"
            )

        except Exception as e:
            logger.error(f"Failed to bind tools to LLM: {str(e)}")
            state.warnings.append(
                {
                    "phase": "llm_tool_binding",
                    "error": f"Failed to bind tools to LLM: {str(e)}",
                    "type": "llm_tool_binding_failed",
                }
            )

    return state


async def enhanced_initialization(state: WorkflowState) -> WorkflowState:
    """
    Enhanced initialization that sets up tools and LLM integration.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    # First run standard initialization
    state = initialize_workflow(state)

    # If initialization failed, don't proceed with tool setup
    if state.next_phase == "error":
        return state

    # Initialize tools
    state = initialize_tools_in_state(state)

    # Set up LLM with tools
    state = setup_llm_with_tools(state)

    return state


async def enhanced_test_generation(state: WorkflowState) -> WorkflowState:
    """
    Enhanced test generation that ensures tools are available.

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state
    """
    # Ensure tools are initialized before test generation
    if not state.tools_registry_initialized:
        state = initialize_tools_in_state(state)
        state = setup_llm_with_tools(state)

    # Run standard test generation
    return await generate_tests(state)


class TestGenerationGraph:
    """
    Enhanced graph definition for the test generation workflow with tool support.
    """

    def __init__(self):
        """Initialize the workflow graph with tool support."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the workflow graph with all nodes and edges, including tool setup.

        Returns:
            StateGraph: The constructed workflow graph
        """
        # Create a new StateGraph with WorkflowState
        try:
            graph = StateGraph(WorkflowState)
        except TypeError:
            graph = StateGraph()

        # Add all nodes (using enhanced versions where needed)
        graph.add_node("initialization", enhanced_initialization)
        graph.add_node("language_detection", detect_project_language)
        graph.add_node("project_analysis", analyze_project)
        graph.add_node("file_analysis", analyze_files)
        graph.add_node("test_path_generation", generate_test_paths)
        graph.add_node("test_generation", enhanced_test_generation)
        graph.add_node("test_execution", execute_tests)
        graph.add_node("test_fixing", fix_tests)  # This now has full tool support
        graph.add_node("complete", complete_workflow)
        graph.add_node("error", handle_error)

        # Function to determine next state based on next_phase
        def route_by_next_phase(state):
            next_phase = state.next_phase
            logger.info(f"Routing from {state.current_phase} to {next_phase or 'END'}")

            # Log tool status when transitioning
            if (
                hasattr(state, "tools_registry_initialized")
                and state.tools_registry_initialized
            ):
                tool_summary = state.get_tool_usage_summary()
                if tool_summary["total_calls"] > 0:
                    logger.info(
                        f"Tool usage so far: {tool_summary['total_calls']} calls, {tool_summary['success_rate']:.1f}% success rate"
                    )

            if next_phase is None:
                return "end"
            return next_phase

        # Add conditional edges for each processing node
        graph.add_conditional_edges(
            "initialization",
            route_by_next_phase,
            {
                "language_detection": "language_detection",
                "project_analysis": "project_analysis",
                "error": "error",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "language_detection",
            route_by_next_phase,
            {"project_analysis": "project_analysis", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "project_analysis",
            route_by_next_phase,
            {"file_analysis": "file_analysis", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "file_analysis",
            route_by_next_phase,
            {
                "test_path_generation": "test_path_generation",
                "error": "error",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "test_path_generation",
            route_by_next_phase,
            {"test_generation": "test_generation", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "test_generation",
            route_by_next_phase,
            {"test_execution": "test_execution", "error": "error", "end": END},
        )

        graph.add_conditional_edges(
            "test_execution",
            route_by_next_phase,
            {
                "test_fixing": "test_fixing",
                "complete": "complete",
                "error": "error",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "test_fixing",
            route_by_next_phase,
            {"complete": "complete", "error": "error", "end": END},
        )

        # Complete and error should end the workflow
        graph.add_conditional_edges("complete", route_by_next_phase, {"end": END})
        graph.add_conditional_edges("error", route_by_next_phase, {"end": END})

        # Set the entry point
        graph.set_entry_point("initialization")

        # Compile the graph
        try:
            graph = graph.compile()
        except AttributeError:
            pass

        return graph

    async def run(self, state: WorkflowState) -> WorkflowState:
        """
        Run the workflow with the given initial state.

        Args:
            state: Initial workflow state

        Returns:
            The final workflow state
        """
        logger.info("Starting enhanced test generation workflow with tool support")

        try:
            # LangGraph async invoke
            try:
                result = await self.graph.ainvoke(state)
            except AttributeError:
                result = self.graph.invoke(state)

            # Log final tool usage summary
            if hasattr(result, "get_tool_usage_summary"):
                tool_summary = result.get_tool_usage_summary()
                if tool_summary["total_calls"] > 0:
                    logger.info("=== FINAL TOOL USAGE SUMMARY ===")
                    logger.info(f"Total tool calls: {tool_summary['total_calls']}")
                    logger.info(f"Successful calls: {tool_summary['successful_calls']}")
                    logger.info(f"Failed calls: {tool_summary['failed_calls']}")
                    logger.info(f"Success rate: {tool_summary['success_rate']:.1f}%")
                    logger.info(f"Most used tool: {tool_summary['most_used_tool']}")
                    logger.info("Tools used:")
                    for tool_name, count in tool_summary["tools_used"].items():
                        logger.info(f"  - {tool_name}: {count} times")
                    logger.info("=== END TOOL USAGE SUMMARY ===")

            return result

        except Exception as e:
            logger.exception(f"Error running enhanced workflow: {str(e)}")

            # Create error state
            state.errors.append(
                {
                    "phase": "workflow",
                    "error": f"Enhanced workflow execution error: {str(e)}",
                    "type": "exception",
                }
            )
            state.is_completed = True
            state.next_phase = None

            return state
