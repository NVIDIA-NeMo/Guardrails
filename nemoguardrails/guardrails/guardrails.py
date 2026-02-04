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

import logging
from enum import Enum
from typing import AsyncIterator, Optional, Tuple, TypeAlias, Union, overload

from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse

# Queue configuration constants
MAX_QUEUE_SIZE = 100
MAX_CONCURRENCY = 10

log = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Enumeration of message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CONTEXT = "context"
    EVENT = "event"
    TOOL = "tool"


LLMMessages: TypeAlias = list[dict[str, str]]


class Guardrails:
    """Top-level interface for NeMo Guardrails functionality."""

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
    ):
        """Initialize a Guardrails instance."""

        self.config = config
        self.llm = llm
        self.verbose = verbose

        self.llmrails = LLMRails(config, llm, verbose)

        # Async work queue for managing concurrent generate_async requests
        self._generate_async_queue: AsyncWorkQueue = AsyncWorkQueue(
            name="generate_async_queue",
            max_queue_size=MAX_QUEUE_SIZE,
            max_concurrency=MAX_CONCURRENCY,
            reject_on_full=True,
        )

        # List of all queues for lifecycle management
        self._queues = [self._generate_async_queue]

    @staticmethod
    def _convert_to_messages(prompt: str | None = None, messages: LLMMessages | None = None) -> LLMMessages:
        """Convert prompt or simplified messages to LLMRails standard format.

        Converts from Guardrails simplified format to LLMRails standard format:
        - Simplified: [{"user": "text"}]
        - Standard: [{"role": "user", "content": "Hello"}]
        """

        # Priority: messages first, then prompt
        if messages:
            return messages

        if prompt:
            # Convert string prompt to standard format
            return [{"role": "user", "content": prompt}]

        raise ValueError("Neither prompt nor messages provided for generation")

    def generate(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate an LLM response synchronously with guardrails applied."""

        messages = self._convert_to_messages(prompt, messages)
        return self.llmrails.generate(messages=messages, **kwargs)

    @overload
    async def generate_async(self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs) -> str: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> dict: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> GenerationResponse: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> tuple[dict, dict]: ...

    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> str | dict | GenerationResponse | tuple[dict, dict]:
        """Generate an LLM response asynchronously with guardrails applied."""

        messages = self._convert_to_messages(prompt, messages)

        # Submit to work queue for concurrency control
        response = await self._generate_async_queue.submit(self.llmrails.generate_async, messages=messages, **kwargs)
        return response

    def stream_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> AsyncIterator[str | dict]:
        """Generate an LLM response asynchronously with streaming support."""

        messages = self._convert_to_messages(prompt, messages)
        return self.llmrails.stream_async(messages=messages, **kwargs)

    def explain(self) -> ExplainInfo:
        """Get the latest ExplainInfo object for debugging."""
        return self.llmrails.explain()

    def update_llm(self, llm: Union[BaseLLM, BaseChatModel]) -> None:
        """Replace the main LLM with a new one."""
        self.llm = llm
        self.llmrails.update_llm(llm)

    async def startup(self) -> None:
        """Lifecycle method to create worker threads and infrastructure"""
        for queue in self._queues:
            await queue.start()

    async def shutdown(self) -> None:
        """Lifecycle method to cleanly shutdown worker threads and infrastructure"""
        for queue in self._queues:
            await queue.stop()

    async def __aenter__(self):
        """Async context manager entry - starts the queues."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - shuts down the queues."""
        await self.shutdown()
