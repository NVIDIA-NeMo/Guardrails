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
NeMo Guardrails. The Guardrails class wraps either IORails or LLMRails (chosen
automatically based on config) and provides a streamlined API for generating
LLM responses with programmable guardrails.
"""

import logging
from typing import AsyncIterator, Optional, Tuple, Union, cast, overload

from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.guardrails import configure_logging
from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.guardrails_types import LLMMessages
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import RailsConfig, _get_flow_name
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse

# Queue configuration constants
MAX_QUEUE_SIZE = 256
MAX_CONCURRENCY = 256

log = logging.getLogger(__name__)


# Set with flows supported by the IORailsEngine
IORAILS_RAILS = {"input", "output", "config"}
IORAILS_INPUT_FLOWS = {"content safety check input", "topic safety check input", "jailbreak detection model"}
IORAILS_OUTPUT_FLOWS = {"content safety check output"}


class Guardrails:
    """Top-level interface for NeMo Guardrails functionality."""

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
        *,
        use_iorails: bool = True,  # False -> fall back to LLMRails instead
    ):
        """Initialize a Guardrails instance."""

        self.config = config
        self.verbose = verbose

        if verbose:
            configure_logging(logging.DEBUG)
        else:
            configure_logging(logging.INFO)

        # Whether to use IORailsEngine for inference requests
        use_iorails_engine = use_iorails and self._has_only_iorails_flows()
        self._rails_engine = IORails(config) if use_iorails_engine else LLMRails(config, llm, verbose)

        # Async work queue for managing concurrent generate_async requests
        self._generate_async_queue: AsyncWorkQueue = AsyncWorkQueue(
            name="generate_async_queue",
            max_queue_size=MAX_QUEUE_SIZE,
            max_concurrency=MAX_CONCURRENCY,
            reject_on_full=True,
        )

        # List of all queues for lifecycle management
        self._queues = [self._generate_async_queue]

        # Track whether startup() has been called (supports lazy initialization)
        self._started = False

    @property
    def rails_engine(self) -> IORails | LLMRails:
        """Get immutable LLMRails object"""
        return self._rails_engine

    @staticmethod
    def _convert_to_messages(prompt: str | None = None, messages: LLMMessages | None = None) -> LLMMessages:
        """Return messages in standard format, converting a prompt string if needed.

        If messages is provided, returns it as-is.
        If prompt is provided, wraps it as [{"role": "user", "content": prompt}].
        """

        # Priority: messages first, then prompt
        if messages:
            return messages

        if prompt:
            # Convert string prompt to standard format
            return [{"role": "user", "content": prompt}]

        raise ValueError("Neither prompt nor messages provided for generation")

    def _has_only_iorails_flows(self):
        """Check if all the flows in the config can be supported by IORails"""

        # If we have any rails outside of `input` and `output` we don't support them
        rails_set = self.config.rails.model_fields_set
        if rails_set - IORAILS_RAILS:
            return False

        for flow in self.config.rails.input.flows:
            flow_name = _get_flow_name(flow)
            if flow_name not in IORAILS_INPUT_FLOWS:
                return False

        for flow in self.config.rails.output.flows:
            flow_name = _get_flow_name(flow)
            if flow_name not in IORAILS_OUTPUT_FLOWS:
                return False

        return True

    async def _ensure_started(self) -> None:
        """Lazy initialization: call startup() on first use if not already started."""
        if not self._started:
            await self.startup()

    def generate(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate an LLM response synchronously with guardrails applied.
        Supported in both IORails and LLMRails
        """

        generate_messages = self._convert_to_messages(prompt, messages)
        return self.rails_engine.generate(messages=generate_messages, **kwargs)

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
        """Generate an LLM response asynchronously with guardrails applied.
        Supported by both LLMRails and IORails
        """
        await self._ensure_started()

        generate_messages = self._convert_to_messages(prompt, messages)
        response = await self._generate_async_queue.submit(
            self.rails_engine.generate_async, messages=generate_messages, **kwargs
        )
        return response

    def stream_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> AsyncIterator[str | dict]:
        """Generate an LLM response asynchronously with streaming support.
        Only supported when using LLMRails
        """

        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support stream_async()")

        stream_messages = self._convert_to_messages(prompt, messages)
        # self.rails_engine must be LLMRails since we raise above if we're using IORails
        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.stream_async(messages=stream_messages, **kwargs)

    def explain(self) -> ExplainInfo:
        """Get the latest ExplainInfo object for debugging.
        Only supported for LLMRails
        """

        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support explain()")

        # self.rails_engine must be LLMRails since we raise above if we're using IORails
        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.explain()

    def update_llm(self, llm: Union[BaseLLM, BaseChatModel]) -> None:
        """Replace the main LLM with a new one.
        Only supported for LLMRails, since IORails doesn't take LLM as argument
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support update_llm()")

        # self.rails_engine must be LLMRails since we raise above if we're using IORails
        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.update_llm(llm)

    async def startup(self) -> None:
        """Lifecycle method to start async worker tasks and the rails engine.

        Idempotent: safe to call multiple times. Also called automatically
        on first generate_async() if not called explicitly, so callers are
        not required to manage the lifecycle.
        """
        if self._started:
            return
        for queue in self._queues:
            await queue.start()
        if isinstance(self.rails_engine, IORails):
            await self.rails_engine.start()
        self._started = True

    async def shutdown(self) -> None:
        """Lifecycle method to stop async worker tasks and the rails engine.

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            return
        for queue in self._queues:
            await queue.stop()
        if isinstance(self.rails_engine, IORails):
            await self.rails_engine.stop()
        self._started = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
