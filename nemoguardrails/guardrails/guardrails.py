# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Top-level Guardrails interface module.

This module provides a simplified, user-friendly interface for interacting with
NeMo Guardrails. The Guardrails class wraps the LLMRails functionality and provides
a streamlined API for generating LLM responses with programmable guardrails.
"""

from typing import AsyncIterator, Optional, Tuple, TypeAlias, Union, overload

from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.guardrails.guardrails_models import GuardrailsRequest
from nemoguardrails.guardrails.guardrails_types import MessageRole
from nemoguardrails.rails.llm.config import (
    RailsConfig,
)
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse

LLMMessages: TypeAlias = list[dict[MessageRole, str]]


class Guardrails:
    """Top-level interface for NeMo Guardrails functionality.

    This class provides a simplified API for adding programmable guardrails to
    LLM-based conversational applications. It wraps the LLMRails class and offers
    a more intuitive interface for common use cases.

    The Guardrails class supports multiple input formats and provides both synchronous
    and asynchronous generation methods, with optional streaming support.

    Attributes:
        config (RailsConfig): The guardrails configuration containing models, rails,
            and other settings.
        llm (Optional[Union[BaseLLM, BaseChatModel]]): The language model to use for
            generation. If not provided, the model from the config will be used.
        verbose (bool): Whether to enable verbose logging.
        llmrails (LLMRails): The underlying LLMRails instance that handles the actual
            generation logic.

    Example:
        Basic usage with string prompt:

        >>> from nemoguardrails import Guardrails, RailsConfig
        >>> config = RailsConfig.from_path("path/to/config")
        >>> guardrails = Guardrails(config)
        >>> request = GuardrailsRequest(prompt="Hello!")
        >>> response = await guardrails.generate_async(request)

        Using with message history:

        >>> request = GuardrailsRequest(prompt=[
        ...     {"user": "What is the weather?"},
        ...     {"assistant": "I don't have access to weather data."},
        ...     {"user": "Ok, tell me a joke instead."}
        ... ])
        >>> response = await guardrails.generate_async(request)

        Streaming responses:

        >>> async for chunk in guardrails.stream_async(request):
        ...     print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
    ):
        """Initialize a Guardrails instance.

        Args:
            config (RailsConfig): The guardrails configuration containing models,
                rails, prompts, and other settings. Can be loaded from a path using
                RailsConfig.from_path() or constructed programmatically.
            llm (Optional[Union[BaseLLM, BaseChatModel]]): An optional language model
                instance to use for generation. If provided, this LLM will be used
                instead of the one specified in the config. Accepts LangChain-compatible
                LLM or ChatModel instances.
            verbose (bool): If True, enables verbose logging for debugging. Default is False.

        Example:
            >>> from nemoguardrails import Guardrails, RailsConfig
            >>> config = RailsConfig.from_path("path/to/config")
            >>> guardrails = Guardrails(config, verbose=True)
        """

        self.config = config
        self.llm = llm
        self.verbose = verbose
        if llm:
            self.llmrails = LLMRails(config, llm=llm)
        else:
            self.llmrails = LLMRails(config)

    @staticmethod
    def _request_messages(request: GuardrailsRequest) -> LLMMessages:
        """Convert a GuardrailsRequest prompt into LLMMessages format.

        This internal method normalizes different prompt formats into a standard
        list of message dictionaries that can be consumed by the underlying LLMRails.

        Args:
            request (GuardrailsRequest): The request object containing the prompt
                to be converted. The prompt can be either a string or a list of
                message dictionaries.

        Returns:
            LLMMessages: A list of message dictionaries where each dictionary has
                a MessageRole key (e.g., 'user', 'assistant') and a string value
                containing the message content.

        Raises:
            ValueError: If the prompt format cannot be converted to LLMMessages.
                This occurs when the prompt is neither a string nor a list.

        Example:
            >>> request = GuardrailsRequest(prompt="Hello")
            >>> messages = Guardrails._request_messages(request)
            >>> # Returns: [{"user": "Hello"}]

            >>> request = GuardrailsRequest(prompt=[
            ...     {"user": "Hi"}, {"assistant": "Hello!"}
            ... ])
            >>> messages = Guardrails._request_messages(request)
            >>> # Returns: [{"user": "Hi"}, {"assistant": "Hello!"}]
        """

        if isinstance(request.prompt, str):
            return LLMMessages([{MessageRole.USER: request.prompt}])
        if isinstance(request.prompt, list):
            return LLMMessages(request.prompt)

        raise ValueError(f"Can't unpack messages from {request.prompt}")

    def generate(
        self,
        request: GuardrailsRequest,
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate an LLM response synchronously with guardrails applied.

        This method processes the request through configured guardrails (input rails,
        dialog rails, retrieval rails, execution rails, and output rails) before
        generating the final response from the LLM.

        Note: This is a synchronous method. For better performance in async contexts,
        use generate_async() instead.

        Args:
            request (GuardrailsRequest): The request containing the prompt to process.
                The prompt can be a string or a list of message dictionaries.

        Returns:
            Union[str, dict, GenerationResponse, Tuple[dict, dict]]: The generated
                response. The return type depends on the configuration and options:
                - str: Simple text response (most common)
                - dict: Response with additional metadata
                - GenerationResponse: Full response object with all details
                - Tuple[dict, dict]: Response with state information

        Example:
            >>> from nemoguardrails import Guardrails, RailsConfig
            >>> from nemoguardrails.guardrails.guardrails_models import GuardrailsRequest
            >>> config = RailsConfig.from_path("path/to/config")
            >>> guardrails = Guardrails(config)
            >>> request = GuardrailsRequest(prompt="Tell me about AI safety")
            >>> response = guardrails.generate(request)
            >>> print(response)
        """

        messages = self._request_messages(request)
        return self.llmrails.generate(messages=messages)

    @overload
    async def generate_async(
        self,
        request: GuardrailsRequest,
    ) -> str: ...

    @overload
    async def generate_async(
        self,
        request: GuardrailsRequest,
    ) -> dict: ...

    @overload
    async def generate_async(
        self,
        request: GuardrailsRequest,
    ) -> GenerationResponse: ...

    @overload
    async def generate_async(
        self,
        request: GuardrailsRequest,
    ) -> tuple[dict, dict]: ...

    async def generate_async(
        self,
        request: GuardrailsRequest,
    ) -> str | dict | GenerationResponse | tuple[dict, dict]:
        """Generate an LLM response asynchronously with guardrails applied.

        This is the preferred method for generating responses in async contexts.
        It processes the request through all configured guardrails (input rails,
        dialog rails, retrieval rails, execution rails, and output rails) before
        generating the final response from the LLM.

        The method supports multiple input formats for flexibility:
        - String: A single string containing the user's message
        - List of dicts: Message history with role-content pairs

        Args:
            request (GuardrailsRequest): The request containing the prompt to process.
                The prompt field can be:
                - A string (e.g., "Hello!")
                - A list of message dicts (e.g., [{"user": "Hi"}, {"assistant": "Hello"}])

        Returns:
            str | dict | GenerationResponse | tuple[dict, dict]: The generated response.
                The return type depends on the configuration and generation options:
                - str: Simple text response (default and most common)
                - dict: Response with metadata (when return_context_data=True)
                - GenerationResponse: Full response object with all generation details
                - tuple[dict, dict]: Response with state information (advanced usage)

        Example:
            Basic usage:

            >>> request = GuardrailsRequest(prompt="What is machine learning?")
            >>> response = await guardrails.generate_async(request)
            >>> print(response)  # "Machine learning is..."

            With message history:

            >>> request = GuardrailsRequest(prompt=[
            ...     {"user": "What is AI?"},
            ...     {"assistant": "AI is artificial intelligence..."},
            ...     {"user": "Tell me more"}
            ... ])
            >>> response = await guardrails.generate_async(request)
        """

        messages = self._request_messages(request)
        response = await self.llmrails.generate_async(messages=messages)
        return response

    def stream_async(
        self,
        request: GuardrailsRequest,
    ) -> AsyncIterator[str | dict]:
        """Generate an LLM response asynchronously with streaming support.

        This method initiates LLM inference and returns an AsyncIterator that yields
        response chunks as they are generated. This is useful for providing real-time
        feedback to users in conversational applications, allowing them to see the
        response as it's being generated rather than waiting for the complete response.

        The method applies all configured guardrails before streaming begins and can
        also apply output rails to streamed content depending on configuration.

        Args:
            request (GuardrailsRequest): The request containing the prompt to process.
                The prompt can be a string or a list of message dictionaries.

        Returns:
            AsyncIterator[str | dict]: An async iterator that yields response chunks.
                Each chunk can be:
                - str: A text chunk of the response
                - dict: A chunk with metadata (e.g., for Server-Sent Events)

        Example:
            Basic streaming:

            >>> request = GuardrailsRequest(prompt="Explain quantum computing")
            >>> async for chunk in guardrails.stream_async(request):
            ...     print(chunk, end="", flush=True)

            Streaming with SSE format:

            >>> async for event in guardrails.stream_async(request):
            ...     if isinstance(event, dict):
            ...         # Handle metadata or special events
            ...         print(f"Event: {event}")
            ...     else:
            ...         # Handle text chunks
            ...         print(event, end="")

        Note:
            The streaming behavior and chunk format may vary depending on the
            underlying LLM provider and guardrails configuration.
        """

        messages = self._request_messages(request)
        return self.llmrails.stream_async(messages=messages)
