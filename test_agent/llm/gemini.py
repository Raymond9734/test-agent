# test_agent/llm/gemini.py - Fixed version

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Provider for Google's Gemini models.
    """

    def __init__(self):
        """Initialize the Gemini provider."""
        self._api_key = None
        self._models = {
            "gemini-1.0-pro-latest": {
                "context_window": 32768,
                "description": "Gemini 1.0 Pro - Google's general purpose model",
            },
            "gemini-1.5-pro-latest": {
                "context_window": 1000000,
                "description": "Gemini 1.5 Pro - Google's advanced model with large context window",
            },
            "gemini-1.5-flash-latest": {
                "context_window": 1000000,
                "description": "Gemini 1.5 Flash - Faster, more economical model",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "gemini"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available Gemini models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default Gemini model."""
        return "gemini-2.0-flash"  # Using the latest model as default

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # Basic validation for Gemini API keys
        return bool(api_key and isinstance(api_key, str) and len(api_key) > 20)

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured Gemini LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: Gemini LLM instance
        """
        try:
            # Try to import Google libraries
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "Google Gemini integration not available. Please install langchain-google-genai package."
                )

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY")

            if not api_key:
                raise ValueError(
                    "Google API key is required. Set GOOGLE_API_KEY or provide api_key parameter."
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

            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )
        except ImportError as e:
            logger.error(f"Error importing Google libraries: {e}")
            raise ImportError(
                "Google Gemini integration not available. Please install langchain-google-genai package."
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
        Generate text using Gemini's API directly.

        Args:
            prompt: The prompt to send to Gemini
            system_prompt: Optional system prompt (will be prepended to user prompt for Gemini)
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Gemini-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Google API key is required.")

        model = kwargs.get("model", self.default_model)

        # Gemini API URL with API key
        api_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # Combine system prompt with user prompt since Gemini doesn't support system role
        combined_prompt = prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{prompt}"

        # Format content according to Gemini API (no system role support)
        content = [{"role": "user", "parts": [{"text": combined_prompt}]}]

        data = {"contents": content, "generationConfig": {"temperature": temperature}}

        if max_tokens:
            data["generationConfig"]["maxOutputTokens"] = max_tokens

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Gemini API error ({response.status}): {error_text}"
                    )

                if stream:
                    # Return a streaming response
                    return response
                else:
                    # Return the completed response text
                    result = await response.json()
                    return result["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using Gemini.

        Args:
            prompt: The prompt to send to Gemini
            tools: List of tools available to Gemini
            system_prompt: Optional system prompt (will be prepended to user prompt)
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional Gemini-specific parameters

        Returns:
            Dict[str, Any]: Gemini response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Google API key is required.")

        model = kwargs.get("model", self.default_model)

        # Gemini API URL with API key
        api_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # Combine system prompt with user prompt since Gemini doesn't support system role
        combined_prompt = prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{prompt}"

        # Format content according to Gemini API (no system role support)
        content = [{"role": "user", "parts": [{"text": combined_prompt}]}]

        # Format tools for Gemini's API
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(self.format_tool_for_provider(tool))

        data = {
            "contents": content,
            "generationConfig": {"temperature": temperature},
            "tools": formatted_tools,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"Gemini API error ({response.status}): {error_text}"
                    )

                result = await response.json()
                return result

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for Gemini's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for Gemini
        """
        # Gemini uses a format similar to OpenAI with some differences
        return {
            "functionDeclarations": [
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
            ]
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize Gemini's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in Gemini's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # Handle Gemini's function calling format
            if isinstance(input_data, dict):
                # Check for functionCall format
                if "functionCall" in input_data:
                    function_call = input_data["functionCall"]
                    if "args" in function_call:
                        return function_call["args"]

                # Check for our standard format
                if "action_input" in input_data:
                    return input_data["action_input"]

                # If none of the above, return as is
                return input_data

            # For string input, try to parse JSON
            if isinstance(input_data, str):
                try:
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    return input_data

            return input_data

        except Exception as e:
            logger.error(f"Error normalizing Gemini input: {str(e)}")
            return input_data

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize Gemini's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # For simple strings
        if isinstance(output_data, str):
            return output_data

        # For dictionaries and other objects, convert to JSON
        return json.dumps(output_data)

    def create_tool_calling_prompt(self, base_prompt: str, context: str = "") -> str:
        """Create a prompt that encourages Gemini to use tools appropriately."""
        if not self._tools:
            return base_prompt

        tool_descriptions = []
        for tool_name, tool_info in self._tools.items():
            tool_descriptions.append(
                f"- {tool_name}: {tool_info.get('description', '')}"
            )

        enhanced_prompt = f"""
You are an expert test fixing agent with access to the following tools:
{chr(10).join(tool_descriptions)}

{context}

IMPORTANT: When you encounter issues that these tools can solve, USE THEM! For example:
- If you see ImportError or ModuleNotFoundError, use install_python_package to install missing packages
- If you need help with import statements, use fix_import_statement to get suggestions  
- If external dependencies can't be installed, use create_mock_dependency to create mocks
- Use run_test_command to verify that fixes work

Don't just suggest what to do - actually use the tools to fix problems. After using tools successfully, then provide the corrected code or solution.

{base_prompt}

Please proceed with the task and actively use the available tools when they can help.
"""
        return enhanced_prompt
