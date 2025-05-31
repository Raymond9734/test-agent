# test_agent/llm/openai.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI's models.
    """

    def __init__(self):
        """Initialize the OpenAI provider."""
        self._api_key = None
        self._models = {
            "gpt-4o": {
                "context_window": 128000,
                "description": "GPT-4o - Most powerful general model from OpenAI",
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "description": "GPT-4 Turbo - Balanced performance and speed",
            },
            "gpt-4": {
                "context_window": 8192,
                "description": "GPT-4 - Capable large language model from OpenAI",
            },
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "description": "GPT-3.5 Turbo - Fast, economical model",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "openai"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available OpenAI models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default OpenAI model."""
        return "gpt-4o"  # Using the latest model as default

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # OpenAI API keys typically start with 'sk-'
        return bool(api_key and isinstance(api_key, str) and api_key.startswith("sk-"))

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured OpenAI LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: OpenAI LLM instance
        """
        try:
            # Try to import OpenAI libraries
            from langchain_openai import ChatOpenAI

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY or provide api_key parameter."
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

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )
        except ImportError as e:
            logger.error(f"Error importing OpenAI libraries: {e}")
            raise ImportError(
                "OpenAI integration not available. Please install langchain-openai package."
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
        Generate text using OpenAI's API directly.

        Args:
            prompt: The prompt to send to OpenAI
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"OpenAI API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using OpenAI.

        Args:
            prompt: The prompt to send to OpenAI
            tools: List of tools available to OpenAI
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Dict[str, Any]: OpenAI response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        model = kwargs.get("model", self.default_model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Format tools for OpenAI's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": formatted_tools,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"OpenAI API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for OpenAI's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for OpenAI
        """
        # OpenAI format is our standard format
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
        Normalize OpenAI's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in OpenAI's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # If input_data is already a dictionary, return it
            if isinstance(input_data, dict):
                return input_data

            # If it's a string, try to parse it as JSON
            if isinstance(input_data, str):
                try:
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    pass

            # Handle nested structures
            if isinstance(input_data, dict):
                if "arguments" in input_data:
                    try:
                        # Parse arguments if it's a string
                        if isinstance(input_data["arguments"], str):
                            return json.loads(input_data["arguments"])
                        return input_data["arguments"]
                    except json.JSONDecodeError:
                        pass

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing OpenAI input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize OpenAI's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # OpenAI expects tool outputs to be strings
        if isinstance(output_data, str):
            return output_data

        # Convert non-string outputs to JSON
        try:
            return json.dumps(output_data)
        except Exception as e:
            logger.error(f"Error normalizing OpenAI output: {str(e)}")
            return str(output_data)
