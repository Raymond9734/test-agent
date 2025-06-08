# test_agent/llm/claude.py - Enhanced with tool support

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
    Provider for Anthropic's Claude models with enhanced tool support.
    """

    def __init__(self):
        """Initialize the Claude provider."""
        super().__init__()
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
        return "claude-3-5-sonnet-20240620"

    def validate_api_key(self, api_key: str) -> bool:
        """Validate that the API key is correctly formatted."""
        return bool(
            api_key and isinstance(api_key, str) and api_key.startswith("sk-ant-")
        )

    def get_llm(self, **kwargs) -> Any:
        """Return a configured Claude LLM instance."""
        try:
            from langchain_anthropic import ChatAnthropic

            model = kwargs.get("model", self.default_model)
            temperature = kwargs.get("temperature", 0.2)
            streaming = kwargs.get("streaming", True)
            api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                raise ValueError("Anthropic API key is required.")

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

            llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                anthropic_api_key=api_key,
                streaming=streaming,
                callbacks=callbacks if streaming else None,
            )

            return llm

        except ImportError as e:
            logger.error(f"Error importing Anthropic libraries: {e}")
            raise ImportError(
                "Anthropic integration not available. Please install langchain-anthropic package."
            )

    def create_tool_enabled_llm(self, **kwargs):
        """Create an LLM instance with tools bound for LangGraph usage."""
        llm = self.get_llm(**kwargs)

        # If we have bound tools, attach them to the LLM
        if hasattr(llm, "bind_tools") and self._tools:
            try:
                formatted_tools = self.get_bound_tools()
                if formatted_tools:
                    llm = llm.bind_tools(formatted_tools)
                    logger.info(
                        f"Bound {len(formatted_tools)} tools to Claude LLM instance"
                    )
            except Exception as e:
                logger.warning(f"Failed to bind tools to Claude LLM: {str(e)}")

        return llm

    async def _generate_without_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """Generate text using Claude's API without tools."""
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

        # FIXED: Always provide max_tokens with a reasonable default
        effective_max_tokens = max_tokens or 4000

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "max_tokens": effective_max_tokens,  # Always include max_tokens
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

                if stream:
                    return response
                else:
                    result = await response.json()
                    return result["content"][0]["text"]

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
        """Generate text using Claude's API with optional tool usage."""

        # FIXED: Ensure max_tokens has a default value
        effective_max_tokens = max_tokens or 4000

        if use_tools and self._tools:
            # Use tool-enabled generation
            tools = self.get_bound_tools()
            return await self.generate_with_tools(
                prompt=prompt,
                tools=tools,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=effective_max_tokens,  # Pass the effective max_tokens
                **kwargs,
            )
        else:
            # Regular generation
            return await self._generate_without_tools(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=effective_max_tokens,  # Pass the effective max_tokens
                stream=stream,
                **kwargs,
            )

    async def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text with tool calling capabilities using Claude."""
        api_key = (
            kwargs.get("api_key")
            or self._api_key
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise ValueError("Anthropic API key is required.")

        model = kwargs.get("model", self.default_model)

        # Use bound tools if none provided
        if tools is None:
            tools = self.get_bound_tools()

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,  # Ensure we have enough tokens for tool calls
        }

        if system_prompt:
            data["system"] = system_prompt

        if tools:
            data["tools"] = tools

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

                # Process the response to extract tool calls and content
                processed_result = self._process_claude_response(result)

                # If there are tool calls, execute them
                if "tool_calls" in processed_result and processed_result["tool_calls"]:
                    logger.info(
                        f"Claude requested {len(processed_result['tool_calls'])} tool calls"
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

    def _process_claude_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process Claude's response to extract tool calls and content."""
        processed = {"content": "", "tool_calls": []}

        # Extract content from Claude's response format
        if "content" in response:
            content_blocks = response["content"]

            for block in content_blocks:
                if block.get("type") == "text":
                    processed["content"] += block.get("text", "")
                elif block.get("type") == "tool_use":
                    # Claude's tool use format
                    tool_call = {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                    }
                    processed["tool_calls"].append(tool_call)

        return processed

    def format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Format a tool definition for Claude's API format."""
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
        }

    def normalize_tool_input(self, tool_name: str, input_data: Any) -> Any:
        """Normalize Claude's tool input format."""
        try:
            # Handle different input formats from Claude
            if isinstance(input_data, dict):
                # Direct dictionary input - most common case
                return input_data

            elif isinstance(input_data, list) and len(input_data) >= 2:
                # Handle list format [tool_name, args]
                return (
                    input_data[1]
                    if isinstance(input_data[1], dict)
                    else {"input": input_data[1]}
                )

            elif isinstance(input_data, str):
                # Try to parse JSON string
                try:
                    return json.loads(input_data)
                except json.JSONDecodeError:
                    return {"input": input_data}

            # Fallback - wrap in dict
            return {"input": input_data} if input_data is not None else {}

        except Exception as e:
            logger.error(
                f"Error normalizing Claude tool input for {tool_name}: {str(e)}"
            )
            return {"input": str(input_data)} if input_data is not None else {}

    def normalize_tool_output(self, output_data: Any) -> str:
        """Normalize Claude's tool output format to ensure it's a string."""
        if isinstance(output_data, str):
            return output_data

        try:
            return json.dumps(output_data, indent=2)
        except Exception as e:
            logger.error(f"Error normalizing Claude tool output: {str(e)}")
            return str(output_data)

    def create_tool_calling_prompt(self, base_prompt: str, context: str = "") -> str:
        """Create a prompt that encourages Claude to use tools appropriately."""
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
