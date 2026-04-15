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
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from typing import Optional, Union

from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    get_request_id,
    reset_request_id,
    set_new_request_id,
    truncate,
)
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.guardrails.telemetry import get_tracer, is_tracing_enabled, traced_request
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler

log = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I'm sorry, I can't respond to that."

# Default concurrency budget for streaming requests (separate from the AsyncWorkQueue for generate_async)
STREAM_MAX_CONCURRENCY = 256

# Error type used by _generation_task when pushing error JSON into the stream
_GENERATION_ERROR_TYPE = "generation_error"


class IORails:
    """Workflow engine for accelerated Input/Output rails inference."""

    def __init__(self, config: RailsConfig) -> None:
        """Build the engine registry and rails manager from the given config."""
        self._running = False
        self.config = config

        self.engine_registry = EngineRegistry(config.models, config.rails.config)
        self.rails_manager = RailsManager(
            engine_registry=self.engine_registry,
            task_manager=LLMTaskManager(config),
            input_flows=config.rails.input.flows,
            output_flows=config.rails.output.flows,
            input_parallel=config.rails.input.parallel or False,
            output_parallel=config.rails.output.parallel or False,
        )

        # Semaphore for streaming concurrency control / load shedding
        self._stream_semaphore = asyncio.Semaphore(STREAM_MAX_CONCURRENCY)

        # Inline OTEL instrumentation
        self._tracing_enabled = is_tracing_enabled(config.tracing)
        self._tracer = get_tracer() if self._tracing_enabled else None

    @property
    def _has_streaming_output_rails(self) -> bool:
        """True when output rails are configured and streaming is enabled for them."""
        streaming = self.config.rails.output.streaming
        return streaming is not None and streaming.enabled and len(self.config.rails.output.flows) > 0

    async def start(self) -> None:
        """Start the IORails engine. Call this during service startup."""
        if self._running:
            return

        # When starting up, make sure self._running is always set to True even on exceptions.
        # This allows the stop() method to clean up any state
        try:
            await self.engine_registry.start()
        finally:
            self._running = True

    async def stop(self) -> None:
        """Stop the IORails engine. Call this during service shutdown."""
        if not self._running:
            return

        # If any exceptions are thrown when stopping EngineRegistry, set the _running to False
        try:
            await self.engine_registry.stop()
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
        await self.start()

        tracer = self._tracer if self._tracing_enabled else None
        with traced_request(tracer) as req_id:
            t0 = time.monotonic()
            try:
                return await self._do_generate(messages, req_id, **kwargs)
            except Exception:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error("[%s] generate_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                raise
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info("[%s] generate_async completed time=%.1fms", req_id, elapsed_ms)

    async def _do_generate(self, messages: LLMMessages, req_id: str, **kwargs) -> LLMMessage:
        """Core pipeline: input rails -> LLM call -> output rails."""
        log.info("[%s] generate_async called", req_id)
        log.debug("[%s] generate_async messages=%s", req_id, truncate(messages))

        # Step 1: Check input rails
        log.info("[%s] Running input rails", req_id)
        input_result = await self.rails_manager.is_input_safe(messages)
        if not input_result.is_safe:
            log.info("[%s] Input blocked: %s", req_id, input_result.reason)
            return {"role": "assistant", "content": REFUSAL_MESSAGE}

        # Step 2: Generate response from main LLM
        log.info("[%s] Calling main LLM", req_id)
        llm_kwargs = {}
        options = kwargs.get("options")
        if options and isinstance(options, dict):
            options = GenerationOptions(**options)
        if isinstance(options, GenerationOptions) and options.llm_params:
            llm_kwargs = options.llm_params

        response_text = await self.engine_registry.model_call("main", messages, **llm_kwargs)
        log.debug("[%s] Main LLM response: %s", req_id, truncate(response_text))

        # Step 3: Check output rails
        log.info("[%s] Running output rails", req_id)
        output_result = await self.rails_manager.is_output_safe(messages, response_text)
        if not output_result.is_safe:
            log.info("[%s] Output blocked: %s", req_id, output_result.reason)
            return {"role": "assistant", "content": REFUSAL_MESSAGE}

        return {"role": "assistant", "content": response_text}

    def _validate_streaming_with_output_rails(self) -> None:
        """Raise if output rails exist but streaming is not enabled for them."""
        if len(self.config.rails.output.flows) > 0 and not self._has_streaming_output_rails:
            raise StreamingNotSupportedError(
                "stream_async() cannot be used when output rails are configured but "
                "rails.output.streaming.enabled is False. Either set "
                "rails.output.streaming.enabled to True in your configuration, or use "
                "generate_async() instead of stream_async()."
            )

    def stream_async(
        self,
        messages: LLMMessages,
        options: Optional[Union[dict, GenerationOptions]] = None,
        include_metadata: Optional[bool] = False,
    ) -> AsyncIterator[Union[str, dict]]:
        """Stream LLM response tokens with input/output rails applied.

        Returns an async iterator that yields string chunks (or dicts when
        ``include_metadata=True``).  Input rails run before any tokens are
        streamed.  If output rails are configured and streaming is enabled,
        tokens are buffered and checked using the same ``RollingBuffer`` /
        ``stream_first`` semantics as LLMRails.

        Args:
            messages: Conversation messages in OpenAI format.
            options: Optional GenerationOptions (llm_params are forwarded to
                the main LLM call).
            include_metadata: When True, chunks are dicts with ``text`` and
                ``metadata`` keys instead of plain strings.

        Returns:
            An async iterator of string chunks (or dicts).

        Raises:
            StreamingNotSupportedError: If output rails are present but
                ``rails.output.streaming.enabled`` is False.
            ValueError: If ``include_metadata=True`` with output rails
                streaming enabled (BufferStrategy requires plain string chunks).
            asyncio.QueueFull: If the streaming concurrency limit is
                reached (load shedding).
        """
        self._validate_streaming_with_output_rails()

        if include_metadata and self._has_streaming_output_rails:
            raise ValueError(
                "include_metadata=True is not supported when output rails streaming is enabled. "
                "BufferStrategy requires plain string chunks. Use include_metadata=False or "
                "disable output rails streaming."
            )

        # Extract llm_params from GenerationOptions if provided
        llm_kwargs: dict = {}
        if options and isinstance(options, dict):
            options = GenerationOptions(**options)
        if isinstance(options, GenerationOptions) and options.llm_params:
            llm_kwargs = options.llm_params

        streaming_handler = StreamingHandler(include_metadata=include_metadata)

        async def _generation_task():
            """Background task: input rails → stream LLM chunks → push to handler.

            Inherits the request ID from the caller context via create_task().
            """
            req_id = get_request_id()
            t0 = time.monotonic()
            try:
                # Step 1: Input rails (non-streaming)
                log.info("[%s] Running input rails", req_id)
                input_result = await self.rails_manager.is_input_safe(messages)
                if not input_result.is_safe:
                    log.info("[%s] Input blocked: %s", req_id, input_result.reason)
                    await streaming_handler.push_chunk(REFUSAL_MESSAGE)
                    await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
                    return

                # Step 2: Stream main LLM
                log.info("[%s] Streaming main LLM", req_id)
                async for chunk in self.engine_registry.stream_model_call("main", messages, **llm_kwargs):
                    await streaming_handler.push_chunk(chunk)

                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            except Exception as e:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error(
                    "[%s] generation task failed time=%.1fms",
                    req_id,
                    elapsed_ms,
                    exc_info=True,
                )
                error_payload = json.dumps(
                    {"error": {"message": str(e), "type": _GENERATION_ERROR_TYPE, "code": "generation_failed"}}
                )
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info("[%s] generation task completed time=%.1fms", req_id, elapsed_ms)

        async def _wrapped_iterator():
            """Wrap the base iterator with semaphore-based concurrency control."""
            # Ensure engines are running (idempotent if already started).
            await self.start()

            # Non-blocking acquire; raises immediately if all slots are taken.
            # locked() returns True when the semaphore value is 0.  Because there
            # is no await between the check and acquire(), no other coroutine can
            # interleave in asyncio's cooperative model, so this is race-free.
            if self._stream_semaphore.locked():
                raise asyncio.QueueFull("Streaming concurrency limit reached")
            await self._stream_semaphore.acquire()

            token = set_new_request_id()
            req_id = get_request_id()
            t0 = time.monotonic()
            try:
                log.info("[%s] stream_async called", req_id)
                log.debug("[%s] stream_async messages=%s", req_id, truncate(messages))

                task = asyncio.create_task(_generation_task())
                try:
                    # Determine base iterator: with or without output rails
                    if self._has_streaming_output_rails:
                        base_iterator = self._run_output_rails_in_streaming(
                            streaming_handler=streaming_handler,
                            messages=messages,
                        )
                    else:
                        base_iterator = streaming_handler

                    async for chunk in base_iterator:
                        if chunk is not None:
                            yield chunk
                finally:
                    try:
                        if not task.done():
                            task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                    finally:
                        try:
                            reset_request_id(token)
                        except ValueError:
                            # GeneratorExit triggers cleanup in a different context
                            # where the token is no longer valid — safe to ignore.
                            pass
            except Exception:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error("[%s] stream_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                raise
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info("[%s] stream_async completed time=%.1fms", req_id, elapsed_ms)
                self._stream_semaphore.release()

        return _wrapped_iterator()

    async def _run_output_rails_in_streaming(
        self,
        streaming_handler: AsyncIterator[Union[str, dict]],
        messages: LLMMessages,
    ) -> AsyncGenerator[Union[str, dict], None]:
        """Buffer streamed chunks and run output rails on each batch.

        Uses the same ``RollingBuffer`` and ``stream_first`` semantics as
        LLMRails:
        - ``stream_first=True``: yield chunks immediately, then run output
          rails.  If unsafe, inject an error and stop.
        - ``stream_first=False``: run output rails first, only yield chunks
          if safe.
        """

        # Unpack streaming config and get the buffer strategy
        output_streaming_config = self.config.rails.output.streaming
        stream_first = output_streaming_config.stream_first
        buffer_strategy = get_buffer_strategy(output_streaming_config)

        async for chunk_batch in buffer_strategy(streaming_handler):
            user_output_chunks = chunk_batch.user_output_chunks
            bot_response_chunk = buffer_strategy.format_chunks(chunk_batch.processing_context)

            # If the batch contains a generation error from _generation_task,
            # yield it directly and stop — don't feed error JSON through output rails.
            for chunk in user_output_chunks:
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict) and parsed.get("error", {}).get("type") == _GENERATION_ERROR_TYPE:
                        yield chunk
                        return
                except (json.JSONDecodeError, TypeError):
                    pass

            if stream_first:
                for chunk in user_output_chunks:
                    yield chunk

            # Run output rails on the accumulated context
            req_id = get_request_id()
            log.info("[%s] Running output rails", req_id)
            output_result = await self.rails_manager.is_output_safe(messages, bot_response_chunk)
            if not output_result.is_safe:
                log.info("[%s] Output blocked: %s", req_id, output_result.reason)
                error_data = {
                    "error": {
                        "message": f"Blocked by output rails: {output_result.reason}",
                        "type": "guardrails_violation",
                        "param": "output_rails",
                        "code": "content_blocked",
                    }
                }
                yield json.dumps(error_data)
                return

            if not stream_first:
                for chunk in user_output_chunks:
                    yield chunk
