# test_agent/llm/base.py - Enhanced version with tool binding

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
import logging

# Define a logger for the module
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers with tool support.
    """

    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._tool_registry = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the name of the provider."""
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Returns a list of available models for this provider."""
        pass

    @property
    def default_model(self) -> str:
        """Returns the default model for this provider."""
        return self.available_models[0] if self.available_models else ""

    @abstractmethod
    def validate_api_key(self, api_key: str) -> bool:
        """Validates that the API key is correctly formatted."""
        pass

    @abstractmethod
    def get_llm(self, **kwargs) -> Any:
        """Returns a configured LLM instance for use with LangGraph."""
        pass

    def bind_tools(self, tool_registry):
        """
        Bind tools from a tool registry to this LLM provider.

        Args:
            tool_registry: ToolRegistry instance containing available tools
        """
        self._tool_registry = tool_registry
        self._tools = {
            name: tool.to_dict() for name, tool in tool_registry.get_all_tools().items()
        }
        logger.info(f"Bound {len(self._tools)} tools to {self.provider_name} provider")

    def get_bound_tools(self) -> List[Dict[str, Any]]:
        """Get list of bound tools formatted for this provider."""
        if not self._tools:
            return []

        return [self.format_tool_for_provider(tool) for tool in self._tools.values()]

    async def execute_bound_tool(self, tool_name: str, tool_input: Any) -> str:
        """
        Execute a bound tool.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Result of tool execution
        """
        if not self._tool_registry:
            return f"No tool registry bound to provider"

        # Normalize tool input
        normalized_input = self.normalize_tool_input(tool_name, tool_input)

        # Execute tool
        if isinstance(normalized_input, dict):
            result = await self._tool_registry.execute_tool(
                tool_name, **normalized_input
            )
        else:
            result = await self._tool_registry.execute_tool(
                tool_name, input=normalized_input
            )

        return self.normalize_tool_output(result)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        use_tools: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Generate text from the LLM with optional tool usage.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            use_tools: Whether to enable tool usage
            **kwargs: Additional provider-specific parameters
        """
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities.

        Args:
            prompt: The prompt to send to the LLM
            tools: List of tools (if None, uses bound tools)
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters
        """
        pass

    def create_tool_enabled_llm(self, **kwargs):
        """
        Create an LLM instance with tools bound for LangGraph usage.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            LLM instance with tools bound
        """
        llm = self.get_llm(**kwargs)

        # If we have bound tools, attach them to the LLM
        if hasattr(llm, "bind_tools") and self._tools:
            try:
                formatted_tools = self.get_bound_tools()
                if formatted_tools:
                    llm = llm.bind_tools(formatted_tools)
                    logger.info(f"Bound {len(formatted_tools)} tools to LLM instance")
            except Exception as e:
                logger.warning(f"Failed to bind tools to LLM: {str(e)}")

        return llm

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for the specific provider.
        Default implementation (OpenAI-like format).
        """
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize tool input from the LLM to a consistent format.
        """
        return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize tool output to ensure it's a string.
        """
        if isinstance(output_data, str):
            return output_data
        return str(output_data)

    def create_tool_calling_prompt(self, base_prompt: str, context: str = "") -> str:
        """
        Create a prompt that encourages tool usage when appropriate.

        Args:
            base_prompt: The base prompt
            context: Additional context about when to use tools

        Returns:
            Enhanced prompt that encourages tool usage
        """
        if not self._tools:
            return base_prompt

        tool_descriptions = []
        for tool_name, tool_info in self._tools.items():
            tool_descriptions.append(
                f"- {tool_name}: {tool_info.get('description', '')}"
            )

        enhanced_prompt = f"""
{base_prompt}

You have access to the following tools that can help solve problems:
{chr(10).join(tool_descriptions)}

{context}

When you encounter issues that these tools can solve (like missing packages, import errors, etc.), 
use the appropriate tools to fix the problems. After using tools, continue with your task.

Remember to:
1. Use tools when they can help solve specific problems
2. Install missing packages when you see import errors
3. Fix import statements when modules can't be found
4. Create mocks for external dependencies that can't be installed
5. Run test commands to verify fixes

Please proceed with the task and use tools as needed.
"""
        return enhanced_prompt
