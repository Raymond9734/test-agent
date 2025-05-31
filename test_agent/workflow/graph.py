# test_agent/workflow/graph.py

import logging
from typing import Callable, Awaitable

# Import conditional to handle both LangGraph and LangChain
try:
    # LangGraph imports
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for LangChain
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


class TestGenerationGraph:
    """
    Graph definition for the test generation workflow.
    """

    def __init__(self):
        """Initialize the workflow graph."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the workflow graph with all nodes and edges.

        Returns:
            StateGraph: The constructed workflow graph
        """
        # Create a new StateGraph with WorkflowState
        try:
            # LangGraph with type
            graph = StateGraph(WorkflowState)
        except TypeError:
            # Fallback for older versions, manually handle type validation
            graph = StateGraph()

        # Add all nodes
        graph.add_node("initialization", initialize_workflow)
        graph.add_node("language_detection", detect_project_language)
        graph.add_node("project_analysis", analyze_project)
        graph.add_node("file_analysis", analyze_files)
        graph.add_node("test_path_generation", generate_test_paths)
        graph.add_node("test_generation", generate_tests)
        graph.add_node("test_execution", execute_tests)
        graph.add_node("test_fixing", fix_tests)
        graph.add_node("complete", complete_workflow)
        graph.add_node("error", handle_error)

        # Function to determine next state based on next_phase
        def route_by_next_phase(state):
            next_phase = state.next_phase
            logger.info(f"Routing from {state.current_phase} to {next_phase or 'END'}")

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
            # LangGraph uses compile, LangChain may not
            graph = graph.compile()
        except AttributeError:
            # If compile is not available, just continue
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
        logger.info("Starting test generation workflow")

        try:
            # LangGraph/LangChain interface might differ slightly
            try:
                # LangGraph async invoke
                result = await self.graph.ainvoke(state)
            except AttributeError:
                # Fallback to sync invoke
                result = self.graph.invoke(state)

            return result
        except Exception as e:
            logger.exception(f"Error running workflow: {str(e)}")

            # Create error state
            state.errors.append(
                {
                    "phase": "workflow",
                    "error": f"Workflow execution error: {str(e)}",
                    "type": "exception",
                }
            )
            state.is_completed = True
            state.next_phase = None

            return state
