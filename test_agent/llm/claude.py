# test_agent/llm/claude.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """
    Provider for Anthropic's Claude models.
    """

    def __init__(self):
        """Initialize the Claude provider."""
        self._api_key = None
        self._models = {
            "claude-3-5-sonnet-20240620": {
                "context_window": 200000,
                "description": "Claude 3.5 Sonnet - Anthropic's latest mid-range model",
            },
            "claude-3-opus-20240229": {
                "context_window": 200000,
                "description": "Claude 3 Opus - Anthropic's most powerful model",
            },
            "claude-3-sonnet-20240229": {
                "context_window": 200000,
                "description": "Claude 3 Sonnet - Balanced performance and speed",
            },
            "claude-3-haiku-20240307": {
                "context_window": 200000,
                "description": "Claude 3 Haiku - Anthropic's fastest model",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "claude"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available Claude models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default Claude model."""
        return "claude-3-5-sonnet-20240620"  # Using the latest model as default

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # Claude API keys typically start with 'sk-ant-'
        return bool(
            api_key and isinstance(api_key, str) and api_key.startswith("sk-ant-")
        )

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured Claude LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: Claude LLM instance
        """
        try:
            # Try to import Anthropic libraries
            from langchain_anthropic import ChatAnthropic

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY or provide api_key parameter."
                )

            self._api_key = api_key

            # Configure callbacks for streaming if enabled
            callbacks = []
            if streaming:
                try:
                    from langchain.callbacks.streaming_stdout import (
                        StreamingStdOutCallbackHandler,
                    )

                    callbacks.append(StreamingStdOutCallbackHandler())
                except ImportError:
                    logger.warning(
                        "StreamingStdOutCallbackHandler not available, disabling streaming"
                    )

            return ChatAnthropic(
                model=model,
                temperature=temperature,
                anthropic_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )
        except ImportError as e:
            logger.error(f"Error importing Anthropic libraries: {e}")
            raise ImportError(
                "Anthropic integration not available. Please install langchain-anthropic package."
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """
        Generate text using Claude's API directly.

        Args:
            prompt: The prompt to send to Claude
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Claude-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key")
            or self._api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("Anthropic API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        if system_prompt:
            data["system"] = system_prompt

        if max_tokens:
            data["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Claude API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["content"][0]["text"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using Claude.

        Args:
            prompt: The prompt to send to Claude
            tools: List of tools available to Claude
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional Claude-specific parameters

        Returns:
            Dict[str, Any]: Claude response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key")
            or self._api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("Anthropic API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Format tools for Claude's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "tools": formatted_tools,
        }

        if system_prompt:
            data["system"] = system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Claude API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for Claude's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for Claude
        """
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize Claude's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in Claude's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # Handle the simple list format [tool_name, args]
            if isinstance(input_data, list) and len(input_data) >= 2:
                return input_data[1]

            # Handle Claude's advanced tool_use object format
            if isinstance(input_data, dict):
                # Check for special tool_use format
                if input_data.get("type") == "tool_use":
                    # Extract from partial_json if available
                    if "partial_json" in input_data:
                        try:
                            # Parse the partial_json string
                            partial_data = json.loads(input_data["partial_json"])

                            # Handle the __arg1 pattern
                            if "__arg1" in partial_data:
                                # Parse the nested JSON string
                                return json.loads(partial_data["__arg1"])

                            return partial_data
                        except json.JSONDecodeError:
                            # If not valid JSON, return it as is
                            return input_data["partial_json"]

                    # Try input field if partial_json isn't available
                    if "input" in input_data and input_data["input"]:
                        return input_data["input"]

                # Check if we have "responded" field with list of tool_use objects
                if "responded" in input_data and isinstance(
                    input_data["responded"], list
                ):
                    for item in input_data["responded"]:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            # Recursively process this tool_use object
                            return self.normalize_tool_input(tool_name, item)

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing Claude input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize Claude's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        try:
            # If it's already a string, return it
            if isinstance(output_data, str):
                return output_data

            # If it's a list with one element that's a string, return that
            if (
                isinstance(output_data, list)
                and len(output_data) == 1
                and isinstance(output_data[0], str)
            ):
                return output_data[0]

            # If it's a list with objects that have 'text' keys, extract and join them
            if isinstance(output_data, list):
                texts = []
                for item in output_data:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                if texts:
                    return "".join(texts)

            # Last resort - convert to JSON string
            return json.dumps(output_data)

        except Exception as e:
            logger.error(f"Error normalizing Claude output: {str(e)}")
            # Return a string representation as a fallback
            return str(output_data)
