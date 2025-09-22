# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, TypeVar, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Input, Output, gather_with_concurrency
from langchain_core.tools import Tool

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.utils import async_wrap
from nemoguardrails.rails.llm.options import GenerationOptions

logger = logging.getLogger(__name__)


class RunnableRails(Runnable[Input, Output]):
    """A runnable that wraps a rails configuration.

    This class implements the LangChain Runnable protocol to provide a way
    to add guardrails to LangChain components. It can wrap LLM models or
    entire chains and add input/output rails and dialog rails.

    Args:
        config: The rails configuration to use.
        llm: Optional LLM to use with the rails.
        tools: Optional list of tools to register with the rails.
        passthrough: Whether to pass through the original prompt or let
            rails modify it. Defaults to True.
        runnable: Optional runnable to wrap with the rails.
        input_key: The key to use for the input when dealing with dict input.
        output_key: The key to use for the output when dealing with dict output.
        verbose: Whether to print verbose logs.
        input_blocked_message: Message to return when input is blocked by rails.
        output_blocked_message: Message to return when output is blocked by rails.
    """

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[Tool]] = None,
        passthrough: bool = True,
        runnable: Optional[Runnable] = None,
        input_key: str = "input",
        output_key: str = "output",
        verbose: bool = False,
        input_blocked_message: str = "I cannot process this request.",
        output_blocked_message: str = "I cannot provide this response.",
    ) -> None:
        self.llm = llm
        self.passthrough = passthrough
        self.passthrough_runnable = runnable
        self.passthrough_user_input_key = input_key
        self.passthrough_bot_output_key = output_key
        self.verbose = verbose
        self.config: Optional[RunnableConfig] = None
        self.input_blocked_message = input_blocked_message
        self.output_blocked_message = output_blocked_message
        self.kwargs: Dict[str, Any] = {}

        # We override the config passthrough.
        config.passthrough = passthrough

        try:
            self.rails = LLMRails(config=config, llm=llm, verbose=verbose)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize LLMRails with configuration: {str(e)}\n\n"
                "Common causes:\n"
                "- Invalid configuration files\n"
                "- Missing required configuration sections\n"
                "- Unsupported model configuration\n\n"
                "Check your config.yml file and ensure all required fields are present."
            ) from e

        if tools:
            # When tools are used, we disable the passthrough mode.
            self.passthrough = False

            for tool in tools:
                self.rails.register_action(tool, tool.name)

        # If we have a passthrough Runnable, we need to register a passthrough fn
        # that will call it
        if self.passthrough_runnable:
            self._init_passthrough_fn()

    def _init_passthrough_fn(self):
        """Initialize the passthrough function for the LLM rails instance."""

        async def passthrough_fn(context: dict, events: List[dict]):
            # First, we fetch the input from the context
            _input = context.get("passthrough_input")
            if hasattr(self.passthrough_runnable, "ainvoke"):
                _output = await self.passthrough_runnable.ainvoke(
                    _input, self.config, **self.kwargs
                )
            else:
                async_wrapped_invoke = async_wrap(self.passthrough_runnable.invoke)
                _output = await async_wrapped_invoke(_input, self.config, **self.kwargs)

            # If the output is a string, we consider it to be the output text
            if isinstance(_output, str):
                text = _output
            elif isinstance(_output, BaseMessage):
                text = _output.content
            else:
                text = _output.get(self.passthrough_bot_output_key)

            return text, _output

        self.rails.llm_generation_actions.passthrough_fn = passthrough_fn

    def __or__(
        self, other: Union[BaseLanguageModel, Runnable[Any, Any]]
    ) -> Union["RunnableRails", Runnable[Any, Any]]:
        """Chain this runnable with another, returning a new runnable.

        This method handles two different cases:
        1. If other is a BaseLanguageModel, set it as the LLM for this RunnableRails
        2. If other is a Runnable, either:
           a. Set it as the passthrough_runnable if this RunnableRails has no passthrough_runnable yet
           b. Otherwise, delegate to the standard Runnable.__or__ to create a proper chain

        This ensures associativity in complex chains.
        """
        if isinstance(other, BaseLanguageModel):
            # Case 1: Set the LLM for this RunnableRails
            self.llm = other
            self.rails.update_llm(other)
            return self

        elif isinstance(other, Runnable):
            # Case 2: Check if this is a RunnableBinding that wraps a BaseLanguageModel
            # This happens when you call llm.bind_tools([...]) - the result is a RunnableBinding
            # that wraps the original LLM but is no longer a BaseLanguageModel instance
            if (
                hasattr(other, "bound")
                and hasattr(other.bound, "__class__")
                and isinstance(other.bound, BaseLanguageModel)
            ):
                # This is an LLM with tools bound to it - treat it as an LLM, not passthrough
                self.llm = other
                self.rails.update_llm(other)
                return self

            if self.passthrough_runnable is None:
                # Case 3a: Set as passthrough_runnable if none exists yet
                self.passthrough_runnable = other
                self.passthrough = True
                self._init_passthrough_fn()
                return self
            else:
                # Case 3b: Delegate to standard Runnable.__or__ for proper chaining
                # This ensures correct behavior in complex chains
                from langchain_core.runnables.base import RunnableSequence

                return RunnableSequence(first=self, last=other)

    @property
    def InputType(self) -> Any:
        return Any

    @property
    def OutputType(self) -> Any:
        """The type of the output of this runnable as a type annotation."""
        return Any

    def get_name(self, suffix: str = "") -> str:
        """Get the name of this runnable."""
        name = "RunnableRails"
        if suffix:
            name += suffix
        return name

    def _extract_text_from_input(self, _input) -> str:
        """Extract text content from various input types for passthrough mode."""
        if isinstance(_input, str):
            return _input
        elif isinstance(_input, BaseMessage):
            return _input.content
        elif isinstance(_input, dict) and self.passthrough_user_input_key in _input:
            return _input.get(self.passthrough_user_input_key)
        else:
            return str(_input)

    def _create_passthrough_messages(self, _input) -> List[Dict[str, Any]]:
        """Create messages for passthrough mode."""
        text_input = self._extract_text_from_input(_input)
        return [
            {
                "role": "context",
                "content": {
                    "passthrough_input": _input,
                    **(_input if isinstance(_input, dict) else {}),
                },
            },
            {
                "role": "user",
                "content": text_input,
            },
        ]

    def _message_to_dict(self, msg: BaseMessage) -> Dict[str, Any]:
        """Convert a BaseMessage to dictionary format."""
        if isinstance(msg, AIMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        else:  # Handle other message types
            role = getattr(msg, "type", "user")
            return {"role": role, "content": msg.content}

    def _transform_chat_prompt_value(
        self, _input: ChatPromptValue
    ) -> List[Dict[str, Any]]:
        """Transform ChatPromptValue to messages list."""
        return [self._message_to_dict(msg) for msg in _input.messages]

    def _transform_base_message_list(
        self, _input: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Transform list of BaseMessage objects to messages list."""
        return [self._message_to_dict(msg) for msg in _input]

    def _extract_user_input_from_dict(self, _input: dict):
        """Extract user input from dictionary, checking configured key first."""
        if self.passthrough_user_input_key in _input:
            return _input[self.passthrough_user_input_key]
        elif "input" in _input:
            return _input["input"]
        else:
            available_keys = list(_input.keys())
            raise ValueError(
                "Expected '{}' or 'input' key in input dictionary. Available keys: {}".format(
                    self.passthrough_user_input_key, available_keys
                )
            )

    def _transform_dict_message_list(self, user_input: list) -> List[Dict[str, Any]]:
        """Transform list from dictionary input to messages."""
        if all(isinstance(msg, BaseMessage) for msg in user_input):
            # Handle BaseMessage objects in the list
            return [self._message_to_dict(msg) for msg in user_input]
        elif all(isinstance(msg, dict) for msg in user_input):
            # Handle dict-style messages
            for msg in user_input:
                if "role" not in msg or "content" not in msg:
                    raise ValueError(
                        "Message missing 'role' or 'content': {}".format(msg)
                    )
            return [
                {"role": msg["role"], "content": msg["content"]} for msg in user_input
            ]
        else:
            raise ValueError("Cannot handle list input with mixed types")

    def _transform_dict_user_input(self, user_input) -> List[Dict[str, Any]]:
        """Transform user input value from dictionary."""
        if isinstance(user_input, str):
            return [{"role": "user", "content": user_input}]
        elif isinstance(user_input, BaseMessage):
            return [self._message_to_dict(user_input)]
        elif isinstance(user_input, list):
            return self._transform_dict_message_list(user_input)
        else:
            raise ValueError(
                "Cannot handle input of type {}".format(type(user_input).__name__)
            )

    def _transform_dict_input(self, _input: dict) -> List[Dict[str, Any]]:
        """Transform dictionary input to messages list."""
        user_input = self._extract_user_input_from_dict(_input)
        messages = self._transform_dict_user_input(user_input)

        if "context" in _input:
            if not isinstance(_input["context"], dict):
                raise ValueError(
                    "The input `context` key for `RunnableRails` must be a dict."
                )
            messages = [{"role": "context", "content": _input["context"]}] + messages

        return messages

    def _transform_input_to_rails_format(self, _input) -> List[Dict[str, Any]]:
        """Transform input to the format expected by the rails.

        Args:
            _input: The input to transform.

        Returns:
            A list of messages in the format expected by the rails.

        Raises:
            ValueError: If the input format cannot be handled.
        """
        if self.passthrough and self.passthrough_runnable:
            return self._create_passthrough_messages(_input)

        try:
            if isinstance(_input, ChatPromptValue):
                return self._transform_chat_prompt_value(_input)
            elif isinstance(_input, StringPromptValue):
                return [{"role": "user", "content": _input.text}]
            elif isinstance(_input, BaseMessage):
                return [self._message_to_dict(_input)]
            elif isinstance(_input, list) and all(
                isinstance(msg, BaseMessage) for msg in _input
            ):
                return self._transform_base_message_list(_input)
            elif isinstance(_input, dict):
                return self._transform_dict_input(_input)
            elif isinstance(_input, str):
                return [{"role": "user", "content": _input}]
            else:
                input_type = type(_input).__name__
                raise ValueError(
                    "Unsupported input type '{}'. Supported formats: str, dict with 'input' key, "
                    "BaseMessage, List[BaseMessage], ChatPromptValue, StringPromptValue".format(
                        input_type
                    )
                )
        except Exception as e:
            # Re-raise known ValueError exceptions
            if isinstance(e, ValueError):
                raise
            # Wrap other exceptions with helpful context
            raise ValueError(
                "Input transformation error: {}. Input type: {}".format(
                    str(e), type(_input).__name__
                )
            ) from e

    def _extract_content_from_result(self, result: Any) -> str:
        """Extract text content from result, handling both dict and direct formats."""
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        return str(result)

    def _get_bot_message(self, result: Any, context: Dict[str, Any]) -> str:
        """Extract the bot message from context or result."""
        return context.get(
            "bot_message", result.get("content") if isinstance(result, dict) else result
        )

    def _format_passthrough_output(self, result: Any, context: Dict[str, Any]) -> Any:
        """Format output for passthrough mode."""
        passthrough_output = context.get("passthrough_output")
        bot_message = self._get_bot_message(result, context)

        # If a rail was triggered (input or dialog), the passthrough_output
        # will not be set. In this case, we only set the output key to the
        # message that was received from the guardrail configuration.
        if passthrough_output is None:
            content = self._extract_content_from_result(result)
            passthrough_output = {self.passthrough_bot_output_key: content}

        # We make sure that, if the output rails altered the bot message, we
        # replace it in the passthrough_output
        if isinstance(passthrough_output, str):
            passthrough_output = bot_message
        elif isinstance(passthrough_output, dict):
            passthrough_output[self.passthrough_bot_output_key] = bot_message

        return passthrough_output

    def _format_chat_prompt_output(
        self,
        result: Any,
        tool_calls: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> AIMessage:
        """Format output for ChatPromptValue input."""
        content = self._extract_content_from_result(result)

        if metadata and isinstance(metadata, dict):
            metadata_copy = metadata.copy()
            metadata_copy.pop("content", None)
            if tool_calls:
                metadata_copy["tool_calls"] = tool_calls
            return AIMessage(content=content, **metadata_copy)
        elif tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=content)

    def _format_string_prompt_output(self, result: Any) -> str:
        """Format output for StringPromptValue input."""
        return self._extract_content_from_result(result)

    def _format_message_output(
        self,
        result: Any,
        tool_calls: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> AIMessage:
        """Format output for BaseMessage input types."""
        content = self._extract_content_from_result(result)

        if metadata and isinstance(metadata, dict):
            metadata_copy = metadata.copy()
            metadata_copy.pop("content", None)
            if tool_calls:
                metadata_copy["tool_calls"] = tool_calls
            return AIMessage(content=content, **metadata_copy)
        elif tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=content)

    def _format_dict_output_for_string_input(
        self, result: Any, output_key: str
    ) -> Dict[str, Any]:
        """Format dict output when the user input was a string."""
        content = self._extract_content_from_result(result)
        return {output_key: content}

    def _format_dict_output_for_dict_message_list(
        self, result: Any, output_key: str
    ) -> Dict[str, Any]:
        """Format dict output when user input was a list of dict messages."""
        content = self._extract_content_from_result(result)
        return {
            output_key: {
                "role": "assistant",
                "content": content,
            }
        }

    def _format_dict_output_for_base_message_list(
        self,
        result: Any,
        output_key: str,
        tool_calls: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Format dict output when user input was a list of BaseMessage objects."""
        content = self._extract_content_from_result(result)

        if metadata and isinstance(metadata, dict):
            metadata_copy = metadata.copy()
            metadata_copy.pop("content", None)
            if tool_calls:
                metadata_copy["tool_calls"] = tool_calls
            return {output_key: AIMessage(content=content, **metadata_copy)}
        elif tool_calls:
            return {output_key: AIMessage(content=content, tool_calls=tool_calls)}
        return {output_key: AIMessage(content=content)}

    def _format_dict_output_for_base_message(
        self,
        result: Any,
        output_key: str,
        tool_calls: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Format dict output when user input was a BaseMessage."""
        content = self._extract_content_from_result(result)

        if metadata:
            metadata_copy = metadata.copy()
            if tool_calls:
                metadata_copy["tool_calls"] = tool_calls
            return {output_key: AIMessage(content=content, **metadata_copy)}
        elif tool_calls:
            return {output_key: AIMessage(content=content, tool_calls=tool_calls)}
        return {output_key: AIMessage(content=content)}

    def _format_dict_output(
        self,
        input_dict: dict,
        result: Any,
        tool_calls: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Format output for dictionary input."""
        output_key = self.passthrough_bot_output_key

        # Get the correct output based on input type
        if self.passthrough_user_input_key in input_dict or "input" in input_dict:
            user_input = input_dict.get(
                self.passthrough_user_input_key, input_dict.get("input")
            )
            if isinstance(user_input, str):
                return self._format_dict_output_for_string_input(result, output_key)
            elif isinstance(user_input, list):
                if all(isinstance(msg, dict) and "role" in msg for msg in user_input):
                    return self._format_dict_output_for_dict_message_list(
                        result, output_key
                    )
                elif all(isinstance(msg, BaseMessage) for msg in user_input):
                    return self._format_dict_output_for_base_message_list(
                        result, output_key, tool_calls, metadata
                    )
                else:
                    return {output_key: result}
            elif isinstance(user_input, BaseMessage):
                return self._format_dict_output_for_base_message(
                    result, output_key, tool_calls, metadata
                )

        # Generic fallback for dictionaries
        content = self._extract_content_from_result(result)
        return {output_key: content}

    def _format_output(
        self,
        input: Any,
        result: Any,
        context: Dict[str, Any],
        tool_calls: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Any:
        """Format the output based on the input type and rails result.

        Args:
            input: The original input.
            result: The result from the rails.
            context: The context returned by the rails.

        Returns:
            The formatted output.

        Raises:
            ValueError: If the input type cannot be handled.
        """
        # Standardize result format if it's a list
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        if self.passthrough and self.passthrough_runnable:
            return self._format_passthrough_output(result, context)

        if isinstance(input, ChatPromptValue):
            return self._format_chat_prompt_output(result, tool_calls, metadata)
        elif isinstance(input, StringPromptValue):
            return self._format_string_prompt_output(result)
        elif isinstance(input, (HumanMessage, AIMessage, BaseMessage)):
            return self._format_message_output(result, tool_calls, metadata)
        elif isinstance(input, list) and all(
            isinstance(msg, BaseMessage) for msg in input
        ):
            return self._format_message_output(result, tool_calls, metadata)
        elif isinstance(input, dict):
            return self._format_dict_output(input, result, tool_calls, metadata)
        elif isinstance(input, str):
            return self._format_string_prompt_output(result)
        else:
            raise ValueError(f"Unexpected input type: {type(input)}")

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this runnable synchronously."""
        self.config = config
        self.kwargs = kwargs

        try:
            return self._full_rails_invoke(input, config, **kwargs)
        except Exception as e:
            # Provide helpful error messages based on the error type
            error_msg = str(e)

            if "RunnableBinding" in error_msg and "model_kwargs" in error_msg:
                raise ValueError(
                    "LLM with bound tools is not supported. "
                    "Use a basic LLM without tool binding or wrap your entire agent/chain instead."
                ) from e

            elif "RunnableBinding" in error_msg and "tool" in error_msg.lower():
                raise ValueError(
                    "Tool binding at LLM level is not supported. "
                    "Use gateway mode or wrap your entire agent/chain instead."
                ) from e

            elif "stream" in error_msg.lower() or "async" in error_msg.lower():
                raise ValueError(
                    "Streaming functionality has limitations. "
                    "Try non-streaming mode or use simpler chain patterns for streaming."
                ) from e

            # Re-raise ValueError exceptions (these are already user-friendly)
            elif isinstance(e, ValueError):
                raise

            # For other exceptions, provide a generic helpful message
            else:
                raise ValueError(
                    "Guardrails error: {}. Input type: {} ".format(
                        str(e), type(input).__name__
                    )
                ) from e

    def _input_to_rails_messages(self, input: Input) -> List[dict]:
        """Convert various input formats to rails message format."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        elif isinstance(input, dict):
            if "input" in input:
                return [{"role": "user", "content": str(input["input"])}]
            elif "messages" in input:
                # Convert LangChain message format to rails format if needed
                if isinstance(input["messages"], list) and len(input["messages"]) > 0:
                    return self._convert_messages_to_rails_format(input["messages"])
                return []
            else:
                return [{"role": "user", "content": str(input)}]
        elif isinstance(input, list):
            # Convert LangChain messages to rails format
            return self._convert_messages_to_rails_format(input)
        else:
            return [{"role": "user", "content": str(input)}]

    def _convert_messages_to_rails_format(self, messages) -> List[dict]:
        """Convert LangChain messages to rails message format."""
        rails_messages = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # LangChain BaseMessage format
                rails_messages.append(
                    {
                        "role": (
                            msg.role
                            if msg.role in ["user", "assistant", "system"]
                            else "user"
                        ),
                        "content": str(msg.content),
                    }
                )
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Already in rails format
                rails_messages.append(
                    {
                        "role": (
                            msg["role"]
                            if msg["role"] in ["user", "assistant", "system"]
                            else "user"
                        ),
                        "content": str(msg["content"]),
                    }
                )
            else:
                # Fallback: treat as user message
                rails_messages.append({"role": "user", "content": str(msg)})
        return rails_messages

    def _extract_output_content(self, output: Output) -> str:
        """Extract content from output for rails checking."""
        if isinstance(output, str):
            return output
        elif hasattr(output, "content"):  # LangChain AIMessage
            return str(output.content)
        elif isinstance(output, dict):
            if "output" in output:
                return str(output["output"])
            elif "content" in output:
                return str(output["content"])
            else:
                return str(output)
        else:
            return str(output)

    def _full_rails_invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Full rails mode: existing LLMRails processing."""
        input_messages = self._transform_input_to_rails_format(input)

        # Store run manager if available for callbacks
        run_manager = kwargs.get("run_manager", None)

        # Generate response from rails
        res = self.rails.generate(
            messages=input_messages, options=GenerationOptions(output_vars=True)
        )
        context = res.output_data
        result = res.response

        # If more than one message is returned, we only take the first one.
        # This can happen for advanced use cases, e.g., when the LLM could predict
        # multiple function calls at the same time. We'll deal with these later.
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        # Format and return the output based in input type
        return self._format_output(
            input, result, context, res.tool_calls, res.llm_metadata
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this runnable asynchronously."""
        self.config = config
        self.kwargs = kwargs

        try:
            return await self._full_rails_ainvoke(input, config, **kwargs)
        except Exception as e:
            # Provide helpful error messages based on the error type
            error_msg = str(e)

            if "RunnableBinding" in error_msg and "model_kwargs" in error_msg:
                raise ValueError(
                    "LLM with bound tools is not supported. "
                    "Use a basic LLM without tool binding or wrap your entire agent/chain instead."
                ) from e

            elif "RunnableBinding" in error_msg and "tool" in error_msg.lower():
                raise ValueError(
                    "Tool binding at LLM level is not supported. "
                    "Use gateway mode or wrap your entire agent/chain instead."
                ) from e

            # Re-raise ValueError exceptions (these are already user-friendly)
            elif isinstance(e, ValueError):
                raise

            # For other exceptions, provide a generic helpful message
            else:
                raise ValueError(
                    "Async guardrails error: {}. Input type: {}".format(
                        str(e), type(input).__name__
                    )
                ) from e

    async def _full_rails_ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Full rails mode async: existing LLMRails processing."""
        input_messages = self._transform_input_to_rails_format(input)

        # Store run manager if available for callbacks
        run_manager = kwargs.get("run_manager", None)

        # Generate response from rails asynchronously
        res = await self.rails.generate_async(
            messages=input_messages, options=GenerationOptions(output_vars=True)
        )
        context = res.output_data
        result = res.response

        # Format and return the output based on input type
        return self._format_output(
            input, result, context, res.tool_calls, res.llm_metadata
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """Stream the output of this runnable synchronously.

        Provides token-by-token streaming of the LLM response with guardrails applied.
        Handles async context properly by running astream in a separate event loop.
        """
        from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
        from nemoguardrails.utils import get_or_create_event_loop

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "Cannot use sync stream() inside async code. Use astream() instead."
            )

        async def _collect_all_chunks():
            chunks = []
            async for chunk in self.astream(input, config, **kwargs):
                chunks.append(chunk)
            return chunks

        loop = get_or_create_event_loop()
        all_chunks = loop.run_until_complete(_collect_all_chunks())

        for chunk in all_chunks:
            yield chunk

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """Stream the output of this runnable asynchronously.

        Provides token-by-token streaming of the LLM response with guardrails applied.
        Uses LLMRails.stream_async() directly for efficient streaming.
        """
        self.config = config
        self.kwargs = kwargs

        input_messages = self._transform_input_to_rails_format(input)

        original_streaming = getattr(self.rails.llm, "streaming", False)
        streaming_enabled = False

        if hasattr(self.rails.llm, "streaming") and not original_streaming:
            self.rails.llm.streaming = True
            streaming_enabled = True

        try:
            async for chunk in self.rails.stream_async(messages=input_messages):
                # Format the chunk based on the input type for streaming
                formatted_chunk = self._format_streaming_chunk(input, chunk)
                yield formatted_chunk
        finally:
            if streaming_enabled and hasattr(self.rails.llm, "streaming"):
                self.rails.llm.streaming = original_streaming

    def _format_streaming_chunk(self, input: Any, chunk: str) -> Any:
        """Format a streaming chunk based on the input type.

        Args:
            input: The original input
            chunk: The current text chunk

        Returns:
            The formatted streaming chunk (using AIMessageChunk for LangChain compatibility)
        """
        if isinstance(input, ChatPromptValue):
            return AIMessageChunk(content=chunk)
        elif isinstance(input, StringPromptValue):
            return chunk
        elif isinstance(input, (HumanMessage, AIMessage, BaseMessage)):
            return AIMessageChunk(content=chunk)
        elif isinstance(input, list) and all(
            isinstance(msg, BaseMessage) for msg in input
        ):
            return AIMessageChunk(content=chunk)
        elif isinstance(input, dict):
            output_key = self.passthrough_bot_output_key
            if self.passthrough_user_input_key in input or "input" in input:
                user_input = input.get(
                    self.passthrough_user_input_key, input.get("input")
                )
                if isinstance(user_input, str):
                    return {output_key: chunk}
                elif isinstance(user_input, list):
                    if all(
                        isinstance(msg, dict) and "role" in msg for msg in user_input
                    ):
                        return {output_key: {"role": "assistant", "content": chunk}}
                    elif all(isinstance(msg, BaseMessage) for msg in user_input):
                        return {output_key: AIMessageChunk(content=chunk)}
                    return {output_key: chunk}
                elif isinstance(user_input, BaseMessage):
                    return {output_key: AIMessageChunk(content=chunk)}
            return {output_key: chunk}
        elif isinstance(input, str):
            return AIMessageChunk(content=chunk)
        else:
            raise ValueError(f"Unexpected input type: {type(input)}")

    def batch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """Batch inputs and process them synchronously."""
        # Process inputs sequentially to maintain state consistency
        return [self.invoke(input, config, **kwargs) for input in inputs]

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        """Batch inputs and process them asynchronously.

        Concurrency is controlled via config['max_concurrency'] following LangChain best practices.
        """
        max_concurrency = None
        if config and "max_concurrency" in config:
            max_concurrency = config["max_concurrency"]

        return await gather_with_concurrency(
            max_concurrency,
            *[self.ainvoke(input_item, config, **kwargs) for input_item in inputs],
        )

    def transform(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Transform the input.

        This is just an alias for invoke.
        """
        return self.invoke(input, config, **kwargs)

    async def atransform(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Transform the input asynchronously.

        This is just an alias for ainvoke.
        """
        return await self.ainvoke(input, config, **kwargs)
