# test_agent/llm/deepseek.py

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .base import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)


class DeepSeekProvider(LLMProvider):
    """
    Provider for DeepSeek models.
    """

    def __init__(self):
        """Initialize the DeepSeek provider."""
        super().__init__()
        self._api_key = None
        self._models = {
            "deepseek-chat": {
                "context_window": 32768,
                "description": "DeepSeek Chat - General purpose model",
            },
            "deepseek-coder": {
                "context_window": 32768,
                "description": "DeepSeek Coder - Optimized for code generation",
            },
        }

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "deepseek"

    @property
    def available_models(self) -> List[str]:
        """Return a list of available DeepSeek models."""
        return list(self._models.keys())

    @property
    def default_model(self) -> str:
        """Return the default DeepSeek model."""
        return "deepseek-coder"  # Using coder model as default for test generation

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate that the API key is correctly formatted.

        Args:
            api_key: API key to validate

        Returns:
            bool: True if the API key is valid, False otherwise
        """
        # Basic validation - DeepSeek API keys follow a specific format
        return bool(api_key and isinstance(api_key, str) and len(api_key) > 20)

    def get_llm(self, **kwargs) -> Any:
        """
        Return a configured DeepSeek LLM instance.

        Args:
            **kwargs: Configuration options for the LLM

        Returns:
            Any: DeepSeek LLM instance
        """
        try:
            # Import DeepSeek conditionally to avoid import errors if not installed
            try:
                from langchain_deepseek import ChatDeepSeek
            except ImportError:
                raise ImportError(
                    "DeepSeek integration not available. Please install langchain-deepseek package."
                )

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")

            if not api_key:
                raise ValueError(
                    "DeepSeek API key is required. Set DEEPSEEK_API_KEY or provide api_key parameter."
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

            return ChatDeepSeek(
                model=model,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                callbacks=callbacks if streaming else None,
            )
        except Exception as e:
            logger.error(f"Error initializing DeepSeek LLM: {e}")
            raise

    def get_bound_tools(self) -> List[Dict[str, Any]]:
        """Get list of bound tools formatted for this provider."""
        if not self._tools:
            return []

        formatted_tools = []
        for tool_name, tool_data in self._tools.items():
            try:
                # Validate tool has required fields
                if not tool_data.get("name", "").strip():
                    logger.warning(f"Skipping tool with empty name: {tool_name}")
                    continue

                formatted_tool = self.format_tool_for_provider(tool_data)
                formatted_tools.append(formatted_tool)
            except Exception as e:
                logger.warning(f"Failed to format tool {tool_name}: {str(e)}")
                continue

        logger.debug(
            f"Formatted {len(formatted_tools)} valid tools for {self.provider_name}"
        )
        return formatted_tools

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
        Generate text using DeepSeek's API directly.

        Args:
            prompt: The prompt to send to DeepSeek
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            use_tools: Whether to enable tool usage
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            Union[str, Any]: Generated text or stream object
        """
        if use_tools and self._tools:
            # Use tool-enabled generation
            tools = self.get_bound_tools()
            return await self.generate_with_tools(
                prompt=prompt,
                tools=tools,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
        else:
            # Regular generation
            return await self._generate_without_tools(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )

    async def _generate_without_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """Generate text using DeepSeek's API without tools."""
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise ValueError("DeepSeek API key is required.")

        # Set environment variable for consistency
        os.environ["DEEPSEEK_API_KEY"] = api_key
        self._api_key = api_key

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

        # DeepSeek API URL may vary - using a placeholder
        api_url = "https://api.deepseek.com/v1/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"DeepSeek API error ({response.status}): {error_text}"
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
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities using DeepSeek.

        Args:
            prompt: The prompt to send to DeepSeek
            tools: List of tools available to DeepSeek (if None, uses bound tools)
            system_prompt: Optional system prompt
            temperature: Temperature setting (0.0 to 1.0)
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            Dict[str, Any]: DeepSeek response containing text and/or tool calls
        """
        api_key = (
            kwargs.get("api_key") or self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise ValueError("DeepSeek API key is required.")

        os.environ["DEEPSEEK_API_KEY"] = api_key
        self._api_key = api_key

        model = kwargs.get("model", self.default_model)

        # Use bound tools if none provided
        if tools is None:
            tools = self.get_bound_tools()

        # Debug logging
        logger.debug(
            f"DeepSeek generate_with_tools called with {len(tools) if tools else 0} tools"
        )
        if tools:
            for i, tool in enumerate(tools):
                tool_name = tool.get("function", {}).get("name", "UNKNOWN")
                logger.debug(f"Tool {i}: {tool_name}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Tools from get_bound_tools() are already formatted,
        # but tools passed directly might need formatting
        formatted_tools = []
        if tools:
            for tool in tools:
                # Check if tool is already formatted (has 'function' key)
                if "function" in tool:
                    # Already formatted
                    tool_name = tool.get("function", {}).get("name", "").strip()
                    if tool_name:
                        formatted_tools.append(tool)
                    else:
                        logger.warning(f"Skipping tool with empty name: {tool}")
                else:
                    # Need to format
                    try:
                        formatted_tool = self.format_tool_for_provider(tool)
                        if formatted_tool.get("function", {}).get("name", "").strip():
                            formatted_tools.append(formatted_tool)
                        else:
                            logger.warning(f"Skipping tool with invalid format: {tool}")
                    except Exception as e:
                        logger.warning(f"Failed to format tool: {e}")
                        continue

        logger.debug(f"Sending {len(formatted_tools)} valid tools to DeepSeek API")

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Only add tools if we have valid ones
        if formatted_tools:
            data["tools"] = formatted_tools
        else:
            logger.info("No valid tools to send to DeepSeek API")

        # DeepSeek API URL may vary - using a placeholder
        api_url = "https://api.deepseek.com/v1/chat/completions"

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(
                        f"DeepSeek API error ({response.status}): {error_text}"
                    )

                result = await response.json()

                # Process the response to extract tool calls and content
                processed_result = self._process_deepseek_response(result)

                # If there are tool calls, execute them
                if "tool_calls" in processed_result and processed_result["tool_calls"]:
                    logger.info(
                        f"DeepSeek requested {len(processed_result['tool_calls'])} tool calls"
                    )

                    # Execute tool calls if we have a tool registry
                    if self._tool_registry:
                        for tool_call in processed_result["tool_calls"]:
                            tool_name = tool_call.get("name")
                            tool_input = self.normalize_tool_input(
                                tool_name, tool_call.get("input")
                            )

                            if tool_name:
                                try:
                                    tool_result = await self.execute_bound_tool(
                                        tool_name, tool_input
                                    )
                                    tool_call["result"] = tool_result
                                    logger.info(
                                        f"Executed tool {tool_name}: {tool_result[:100]}..."
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error executing tool {tool_name}: {str(e)}"
                                    )
                                    tool_call["result"] = f"Error: {str(e)}"

                return processed_result

    def _process_deepseek_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process DeepSeek's response to extract tool calls and content."""
        processed = {"content": "", "tool_calls": []}

        # Extract content from DeepSeek's response format
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            message = choice.get("message", {})

            # Extract text content
            if "content" in message and message["content"]:
                processed["content"] = message["content"]

            # Extract tool calls if present
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if tool_call.get("type") == "function":
                        function = tool_call.get("function", {})
                        processed_tool_call = {
                            "id": tool_call.get("id"),
                            "name": function.get("name"),
                            "input": function.get("arguments", {}),
                        }
                        processed["tool_calls"].append(processed_tool_call)

        return processed

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool definition for DeepSeek's API format.

        Args:
            tool: Tool definition in standard format

        Returns:
            Dict[str, Any]: Tool definition formatted for DeepSeek
        """
        # DeepSeek uses a format similar to OpenAI
        name = tool.get("name", "").strip()
        description = tool.get("description", "").strip()

        # Validate required fields
        if not name:
            raise ValueError(f"Tool missing required 'name' field: {tool}")

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description
                or f"Tool: {name}",  # Provide default description
                "parameters": tool.get("parameters", {}),
            },
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """
        Normalize DeepSeek's tool input format.

        Args:
            tool_name: Name of the tool being called
            input_data: Tool input in DeepSeek's format

        Returns:
            Any: Normalized tool input
        """
        try:
            # Handle different input formats
            if isinstance(input_data, dict):
                # If there's a direct arguments field, use it
                if "arguments" in input_data:
                    args = input_data["arguments"]
                    if isinstance(args, str):
                        return json.loads(args)
                    return args

                # If it has action_input, it's probably our standard format
                if "action_input" in input_data:
                    return input_data["action_input"]

                # For direct parameter dictionaries, return as is
                return input_data

            # Handle string input that might be JSON
            if isinstance(input_data, str):
                try:
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    return {"input": input_data}

            # Fallback - wrap in dict
            return {"input": input_data} if input_data is not None else {}

        except Exception as e:
            logger.error(f"Error normalizing DeepSeek input: {str(e)}")
            return {"input": str(input_data)} if input_data is not None else {}

    def normalize_tool_output(self, output_data: Any) -> str:
        """
        Normalize DeepSeek's tool output format to ensure it's a string.

        Args:
            output_data: Tool output data

        Returns:
            str: Normalized string output
        """
        # For simple strings
        if isinstance(output_data, str):
            return output_data

        # For dictionaries and other objects, convert to JSON
        try:
            return json.dumps(output_data, indent=2)
        except Exception as e:
            logger.error(f"Error normalizing DeepSeek tool output: {str(e)}")
            return str(output_data)

    def create_tool_calling_prompt(self, base_prompt: str, context: str = "") -> str:
        """Create a prompt that encourages DeepSeek to use tools appropriately."""
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

IMPORTANT: When you encounter issues that these tools can solve, USE THEM! For example:
- If you see ImportError or ModuleNotFoundError, use install_python_package to install missing packages
- If you need help with import statements, use fix_import_statement to get suggestions  
- If external dependencies can't be installed, use create_mock_dependency to create mocks
- Use run_test_command to verify that fixes work

Don't just suggest what to do - actually use the tools to fix problems. After using tools successfully, then provide the corrected code or solution.

Please proceed with the task and actively use the available tools when they can help.
"""
        return enhanced_prompt
