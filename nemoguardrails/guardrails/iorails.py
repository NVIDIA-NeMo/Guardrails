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

"""Optimized IORails Engine for specific guardrail configurations.

This module provides an optimized inference path for guardrail configurations that
only use specific supported flows (input/output content safety). For configurations
outside this supported set, the standard LLMRails engine should be used instead.
"""

import asyncio
import logging
import time

from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    get_request_id,
    reset_request_id,
    set_new_request_id,
    truncate,
)
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions

log = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I'm sorry, I can't respond to that."


class IORails:
    """Workflow engine for accelerated Input/Output rails inference."""

    def __init__(self, config: RailsConfig) -> None:
        self._running = False
        self.config = config

        # Model Manager has one or more ModelEngine inside. Each ModelEngine calls a single model or API
        self.model_manager = ModelManager(config)

        # Rails Manager is responsible for running rails by making calls to Model Manager
        self.rails_manager = RailsManager(config, self.model_manager)

    async def start(self) -> None:
        """Start the IORails engine. Call this during service startup."""
        if self._running:
            return

        # When starting up, make sure self._running is always set to True even on exceptions.
        # This allows the stop() method to clean up any state
        try:
            await self.model_manager.start()
        finally:
            self._running = True

    async def stop(self) -> None:
        """Stop the IORails engine. Call this during service shutdown."""
        if not self._running:
            return

        # If any exceptions are thrown when stopping ModelManager, set the _running to False
        try:
            await self.model_manager.stop()
        finally:
            self._running = False

    async def __aenter__(self):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager (used for testing rather than long-lived instance)"""
        await self.stop()

    def generate(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Synchronous version of generate_async."""

        async def _run_sync_iorails():
            async with IORails(self.config) as iorails_engine:
                return await iorails_engine.generate_async(messages, **kwargs)

        return asyncio.run(_run_sync_iorails())

    async def generate_async(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Run input rails, generation, and output rails. Return response if safe."""
        token = set_new_request_id()
        req_id = get_request_id()
        t0 = time.monotonic()
        try:
            log.info("[%s] generate_async called", req_id)
            log.debug("[%s] generate_async messages=%s", req_id, truncate(messages))

            # Step 1: Check input rails
            log.info("[%s] Running input rails", req_id)
            input_result = await self.rails_manager.is_input_safe(messages)
            if not input_result.is_safe:
                log.info("[%s] Input blocked: %s", req_id, input_result.reason)
                return {"role": "assistant", "content": REFUSAL_MESSAGE}

            # Step 2: Generate response from main LLM
            # If we got an `options=GenerationOptions`, then unpack GenerationOptions.llm_params and add
            # that to the main LLM call
            log.info("[%s] Calling main LLM", req_id)
            llm_kwargs = {}
            if kwargs.get("options") and isinstance(kwargs["options"], GenerationOptions):
                generation_options = kwargs["options"]
                llm_kwargs = generation_options.llm_params if generation_options.llm_params else {}

            response_text = await self.model_manager.generate_async("main", messages, **llm_kwargs)
            log.debug("[%s] Main LLM response: %s", req_id, truncate(response_text))

            # Step 3: Check output rails
            log.info("[%s] Running output rails", req_id)
            output_result = await self.rails_manager.is_output_safe(messages, response_text)
            if not output_result.is_safe:
                log.info("[%s] Output blocked: %s", req_id, output_result.reason)
                return {"role": "assistant", "content": REFUSAL_MESSAGE}

            return {"role": "assistant", "content": response_text}
        except Exception:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.error("[%s] generate_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
            raise
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.info("[%s] generate_async completed time=%.1fms", req_id, elapsed_ms)
            reset_request_id(token)
