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
from contextlib import nullcontext, suppress
from typing import Optional, Union

from nemoguardrails.actions.llm.utils import _extract_and_remove_think_tags
from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailDirection,
    get_request_id,
    truncate,
)
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.guardrails.telemetry import (
    are_metrics_enabled,
    get_tracer,
    is_tracing_enabled,
    record_nonstream_rejected,
    record_request_blocked,
    record_request_error,
    record_span_error,
    record_stream_rejected,
    register_nonstream_saturation_gauges,
    request_metrics,
    stream_active_metric,
    traced_request,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler

log = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I'm sorry, I can't respond to that."

# Concurrency budgets for the non-streaming AsyncWorkQueue:
# NONSTREAM_QUEUE_DEPTH      — max pending items before submit raises QueueFull
# NONSTREAM_MAX_CONCURRENCY  — max concurrent worker tasks draining the queue
NONSTREAM_QUEUE_DEPTH = 256
NONSTREAM_MAX_CONCURRENCY = 256

# Concurrency budget for streaming requests (separate from the non-streaming
# AsyncWorkQueue — streams have no admission buffer, just fail-fast on the
# semaphore).
STREAM_MAX_CONCURRENCY = 256

# Error type used by _generation_task when pushing error JSON into the stream
_GENERATION_ERROR_TYPE = "generation_error"


class IORails:
    """Workflow engine for accelerated Input/Output rails inference."""

    def __init__(self, config: RailsConfig) -> None:
        """Build the engine registry and rails manager from the given config."""
        self._running = False
        self.config = config

        # Create the OTEL tracer (if enabled in config).
        # Pass to EngineRegistry and RailsManager to keep all spans consistent under parent
        self._tracing_enabled = is_tracing_enabled(config.tracing)
        self._tracer = get_tracer() if self._tracing_enabled else None
        self._metrics_enabled = are_metrics_enabled(config.metrics)

        self.engine_registry = EngineRegistry(config.models, config.rails.config, tracer=self._tracer)
        self.rails_manager = RailsManager(
            engine_registry=self.engine_registry,
            task_manager=LLMTaskManager(config),
            input_flows=config.rails.input.flows,
            output_flows=config.rails.output.flows,
            input_parallel=config.rails.input.parallel or False,
            output_parallel=config.rails.output.parallel or False,
            tracer=self._tracer,
        )

        # Non-streaming admission queue + worker pool (owned by IORails so
        # all request-path concurrency controls sit under one roof).  The
        # queue auto-starts lazily on first submit(); ``start()`` below
        # starts it explicitly alongside the engine registry.
        self._generate_async_queue = AsyncWorkQueue(
            name="iorails_generate_queue",
            max_queue_size=NONSTREAM_QUEUE_DEPTH,
            max_concurrency=NONSTREAM_MAX_CONCURRENCY,
            reject_on_full=True,
        )

        # Semaphore for streaming concurrency control / load shedding
        self._stream_semaphore = asyncio.Semaphore(STREAM_MAX_CONCURRENCY)

        # ObservableGauges are created lazily on first ``start()`` because
        # they need a reference to an AsyncWorkQueue which has been started.
        self._gauges_registered = False

    @property
    def _has_streaming_output_rails(self) -> bool:
        """True when output rails are configured and streaming is enabled for them."""
        streaming = self.config.rails.output.streaming
        return streaming is not None and streaming.enabled and len(self.config.rails.output.flows) > 0

    async def start(self) -> None:
        """Start the IORails engine. Call this during service startup."""
        if self._running:
            return

        #  The EngineRegistry cleans up all its Engines if there's an exception on startup
        #  so no need to catch exceptions and clean up here
        await self.engine_registry.start()
        try:
            await self._generate_async_queue.start()
            try:
                # Queue is now live; register the state-observing ObservableGauges.
                # ``lambda: self._running`` is checked at collect time so the gauges
                # report empty lists once the engine has been stopped.
                if self._metrics_enabled and not self._gauges_registered:
                    register_nonstream_saturation_gauges(
                        self._generate_async_queue,
                        is_running=lambda: self._running,
                    )
                    self._gauges_registered = True
            except BaseException:
                # Gauge registration failed after the queue was started — roll
                # the queue back so a retry of start() comes from a clean state
                # rather than leaving the queue running with ``_running=False``
                # (which would make stop() a no-op and leak worker tasks).
                try:
                    await self._generate_async_queue.stop()
                except BaseException:
                    log.exception("queue rollback failed during IORails.start()")
                raise
        except BaseException:
            # Log but suppress rollback failures so we propagate the original
            # queue-start (or gauge-registration) error as the actionable root cause.
            try:
                await self.engine_registry.stop()
            except BaseException:
                log.exception("engine_registry rollback failed during IORails.start()")
            raise

        self._running = True

    async def stop(self) -> None:
        """Stop the IORails engine. Call this during service shutdown."""
        if not self._running:
            return

        # Each shutdown step runs independently so a failure in one does not
        # leak the other. _running is cleared regardless so a retry of stop()
        # is a no-op and we don't leak worker tasks.
        try:
            try:
                await self._generate_async_queue.stop()
            finally:
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
        """Synchronous version of generate_async.

        Telemetry is disabled for the ephemeral IORails object used for
        the ``generate()`` call. For production use, use the asynchronous
        `generate_async()` and `stream_async()` methods for non-streaming
        and streaming requests respectively.
        """

        # Disable tracing and metrics for synchronous generation calls
        sync_config = self.config.model_copy(deep=True)
        if sync_config.tracing is not None:
            sync_config.tracing.enabled = False
        if sync_config.metrics is not None:
            sync_config.metrics.enabled = False

        async def _run_sync_iorails():
            """Spin up a short-lived IORails engine for one synchronous generate call."""
            async with IORails(sync_config) as iorails_engine:
                return await iorails_engine.generate_async(messages, **kwargs)

        return asyncio.run(_run_sync_iorails())

    async def generate_async(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Public entry: submit the request to the internal work queue.

        The queue enforces non-streaming concurrency limits
        (``NONSTREAM_MAX_CONCURRENCY`` workers draining up to
        ``NONSTREAM_QUEUE_DEPTH`` pending items).  Callers receive
        ``asyncio.QueueFull`` when the admission buffer is full and
        ``guardrails.nonstream.rejections`` increments if metrics are enabled.

        Request-level metrics (``guardrails.requests``,
        ``guardrails.request.duration``, ``guardrails.requests.errors``)
        wrap the queue submission, so duration includes queue-wait time
        (OTEL HTTP semconv).  A ``QueueFull`` rejection shows up in BOTH
        ``requests.errors{error.type=QueueFull}`` and
        ``nonstream.rejections`` — honest dual-signal reporting.
        """
        await self.start()
        metrics_ctx = request_metrics() if self._metrics_enabled else nullcontext()
        with metrics_ctx:
            try:
                return await self._generate_async_queue.submit(self._run_generate, messages, **kwargs)
            except asyncio.QueueFull:
                if self._metrics_enabled:
                    record_nonstream_rejected()
                raise

    async def _run_generate(self, messages: LLMMessages, **kwargs) -> LLMMessage:
        """Runs inside a queue worker task.  Wraps the pipeline in
        ``traced_request`` so each request gets its own span + request ID,
        then delegates to ``_do_generate`` for the actual input rails →
        LLM → output rails flow.  Metrics are emitted at the outer
        lifecycle scope by ``generate_async``, not here.
        """
        tracer = self._tracer if self._tracing_enabled else None
        with traced_request(tracer) as (_, req_id):
            t0 = time.monotonic()
            try:
                result = await self._do_generate(messages, req_id, **kwargs)
            except Exception:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error("[%s] generate_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                raise
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.info("[%s] generate_async completed time=%.1fms", req_id, elapsed_ms)
            return result

    async def _do_generate(self, messages: LLMMessages, req_id: str, **kwargs) -> LLMMessage:
        """Core pipeline: input rails -> LLM call -> output rails."""
        log.info("[%s] generate_async called", req_id)
        log.debug("[%s] generate_async messages=%s", req_id, truncate(messages))

        # Step 1: Check input rails
        log.info("[%s] Running input rails", req_id)
        input_result = await self.rails_manager.is_input_safe(messages)
        if not input_result.is_safe:
            log.info("[%s] Input blocked: %s", req_id, input_result.reason)
            if self._metrics_enabled:
                record_request_blocked(RailDirection.INPUT)
            return {"role": "assistant", "content": REFUSAL_MESSAGE}

        # Step 2: Generate response from main LLM
        log.info("[%s] Calling main LLM", req_id)
        llm_kwargs = {}
        options = kwargs.get("options")
        if options and isinstance(options, dict):
            options = GenerationOptions(**options)
        if isinstance(options, GenerationOptions) and options.llm_params:
            llm_kwargs = options.llm_params

        response = await self.engine_registry.model_call("main", messages, **llm_kwargs)
        # Log raw content before reasoning extraction and think-token removal
        log.debug("[%s] Raw LLM response: %s", req_id, truncate(response.content))

        # Reasoning extraction prefers LLMResponse `reasoning` field if the provider
        # supports it, falling back to extracting <think>...</think> tags otherwise.
        # The fallback mutates response.content to remove reasoning content.
        reasoning_content = response.reasoning or _extract_and_remove_think_tags(response)
        response_text = response.content

        # Step 3: Check output rails
        log.info("[%s] Running output rails", req_id)
        output_result = await self.rails_manager.is_output_safe(messages, response_text)
        if not output_result.is_safe:
            log.info("[%s] Output blocked: %s", req_id, output_result.reason)
            if self._metrics_enabled:
                record_request_blocked(RailDirection.OUTPUT)
            return {"role": "assistant", "content": REFUSAL_MESSAGE}

        # TODO: Support returning GenerationResponse `reasoning_content` to match LLMRails
        # For now, embed the reasoning on the content with think-tags
        if reasoning_content:
            response_text = f"<think>{reasoning_content}</think>\n" + response_text

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

        async def _generation_task(request_span):
            """Background task: input rails → stream LLM chunks → push to handler.

            ``request_span`` is the IORails request span (or ``None`` when
            tracing is disabled), captured by the caller from
            ``traced_request`` and passed in explicitly — never fetched via
            ``trace.get_current_span()`` which could return the host app's
            ambient span and pollute unrelated traces.

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
                    if self._metrics_enabled:
                        record_request_blocked(RailDirection.INPUT)
                    await streaming_handler.push_chunk(REFUSAL_MESSAGE)
                    await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
                    return

                # Step 2: Stream main LLM content from structured response.
                # Only delta_content is forwarded. Reasoning is dropped for compatibility
                # with LLMRails. Tool-calls are not yet supported by IORails
                log.info("[%s] Streaming main LLM", req_id)
                content_parts: list[str] = []
                async for chunk in self.engine_registry.stream_model_call("main", messages, **llm_kwargs):
                    if chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                        await streaming_handler.push_chunk(chunk.delta_content)

                # While LLMResponseChunk.delta_reasoning is dropped explicitly,
                # think-tags embedded in delta_content are not. Give a warning
                # to reflect this asymmetry (once-per-request).
                full_content = "".join(content_parts)
                if "<think>" in full_content or "</think>" in full_content:
                    log.warning(
                        "[%s] Streamed content contains <think> tags; model is leaking "
                        "reasoning via delta_content rather than delta_reasoning "
                        "(output rails will process reasoning tokens)",
                        req_id,
                    )

                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            except Exception as e:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.error(
                    "[%s] generation task failed time=%.1fms",
                    req_id,
                    elapsed_ms,
                    exc_info=True,
                )
                # Mark the request span ERROR; record_span_error no-ops when
                # request_span is None (tracing disabled), so no extra guard
                # is needed and there's no ambient-context lookup to worry about.
                record_span_error(request_span, e)
                # Bump guardrails.requests.errors explicitly: the exception is
                # about to be swallowed (converted to an error-payload chunk),
                # so request_metrics's except branch never fires for the
                # streaming path.
                if self._metrics_enabled:
                    record_request_error(e)
                error_payload = json.dumps(
                    {"error": {"message": str(e), "type": _GENERATION_ERROR_TYPE, "code": "generation_failed"}}
                )
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore[arg-type]
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info("[%s] generation task completed time=%.1fms", req_id, elapsed_ms)

        async def _wrapped_iterator():
            """Wrap the base iterator with semaphore-based concurrency control.

            Request-level metrics (``guardrails.requests``,
            ``guardrails.request.duration``, ``guardrails.requests.errors``)
            wrap the entire stream lifecycle, so a ``QueueFull`` on the
            semaphore check bumps BOTH ``stream.rejections`` and
            ``requests.errors{error.type=QueueFull}`` — dual-signal
            semantics matching the non-streaming path.
            """
            # Ensure engines are running (idempotent if already started).
            # Kept outside ``request_metrics`` so duration matches the
            # non-streaming path (excludes one-time engine startup cost).
            await self.start()

            metrics_ctx = request_metrics() if self._metrics_enabled else nullcontext()
            with metrics_ctx:
                # Non-blocking acquire; raises immediately if all slots are taken.
                # locked() returns True when the semaphore value is 0.  Because there
                # is no await between the check and acquire(), no other coroutine can
                # interleave in asyncio's cooperative model, so this is race-free.
                if self._stream_semaphore.locked():
                    if self._metrics_enabled:
                        record_stream_rejected()
                    raise asyncio.QueueFull("Streaming concurrency limit reached")
                await self._stream_semaphore.acquire()

                tracer = self._tracer if self._tracing_enabled else None
                # Track this stream as active while it holds the semaphore
                # permit; the CM decrements in its finally, just before the
                # outer ``semaphore.release()`` below.
                stream_active_ctx = stream_active_metric() if self._metrics_enabled else nullcontext()
                try:
                    with stream_active_ctx:
                        # traced_request is entered inside the async generator so the
                        # request span is the current OTEL context when create_task()
                        # below snapshots contextvars — that's what makes rail / LLM
                        # spans raised inside _generation_task attach as children.
                        with traced_request(tracer) as (request_span, req_id):
                            t0 = time.monotonic()
                            try:
                                log.info("[%s] stream_async called", req_id)
                                log.debug("[%s] stream_async messages=%s", req_id, truncate(messages))

                                task = asyncio.create_task(_generation_task(request_span))
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
                                    if not task.done():
                                        task.cancel()
                                    with suppress(asyncio.CancelledError):
                                        await task
                            except Exception:
                                elapsed_ms = (time.monotonic() - t0) * 1000
                                log.error("[%s] stream_async failed time=%.1fms", req_id, elapsed_ms, exc_info=True)
                                raise
                            finally:
                                elapsed_ms = (time.monotonic() - t0) * 1000
                                log.info("[%s] stream_async completed time=%.1fms", req_id, elapsed_ms)
                finally:
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
                if self._metrics_enabled:
                    record_request_blocked(RailDirection.OUTPUT)
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
