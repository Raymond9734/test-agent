# test_agent/llm/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
import logging

# Define a logger for the module
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement,
    allowing the test agent to work with different LLM backends.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Returns the name of the provider.

        Returns:
            str: Provider name (e.g., 'openai', 'claude', 'deepseek', 'gemini')
        """
        pass

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Returns a list of available models for this provider.

        Returns:
            List[str]: List of model names
        """
        pass

    @property
    def default_model(self) -> str:
        """
        Returns the default model for this provider.

        Returns:
            str: Default model name
        """
        return self.available_models[0] if self.available_models else ""

    @abstractmethod
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validates that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_llm(self, **kwargs) -> Any:
        """
        Returns a configured LLM instance for use with LangGraph.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: LLM instance
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities.

        Args:
            prompt: The prompt to send to the LLM
            tools: List of tools available to the LLM
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Response containing text and/or tool calls
        """
        pass

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for the specific provider.

        This method allows customizing the tool format for each provider.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for this provider
        """
        # Default implementation (OpenAI-like format)
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

        Different LLMs may format tool calls differently, this method
        ensures they're normalized to a consistent format for the agent.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in provider-specific format

        Returns:
            Any: Normalized tool input
        """
        # Default implementation (passthrough)
        return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize tool output to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # Default implementation (convert to string)
        if isinstance(output_data, str):
            return output_data
        return str(output_data)
