# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Apply buffered output checks to provider-native event streams."""

import inspect
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from nemoguardrails.server.experimental._buffered_kernel import (
    InspectionStage,
    OperationBlocked,
    OperationCheckFailed,
    OperationModificationUnsupported,
    OperationProjectionFailed,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentChecker,
    ContentCheckFailed,
    OutputContentCheck,
    StreamBufferingPolicy,
    UnsupportedContentModification,
    validate_content_check_decision,
)
from nemoguardrails.server.experimental._guarded_operation import UnsupportedGuardedPayload
from nemoguardrails.server.experimental._http_kernel import BufferedHttpResponse
from nemoguardrails.server.experimental.provider.sse import (
    ServerSentEvent,
    ServerSentEventTooLarge,
    TruncatedServerSentEvent,
    iter_sse_events,
)
from nemoguardrails.server.experimental.provider.stream import (
    ProviderStreamAdapter,
    StreamEventRole,
    UnsupportedProviderStream,
    validate_stream_event,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage


class UnsupportedStreamInspection(ValueError):
    """Report an output policy that cannot safely release a guarded stream."""


class PendingStreamBufferTooLarge(ValueError):
    """Report unchecked provider events that exceed the buffering limit."""


class StreamSnapshotHistoryTooLarge(ValueError):
    """Report snapshot verification state that exceeds the stream limit."""


@dataclass(frozen=True, slots=True)
class StreamUpstreamFailed:
    """Stop a stream when its upstream byte iterator fails."""

    failure: BaseException


@dataclass(frozen=True, slots=True)
class StreamProcessingFailed:
    """Stop a stream when guarded stream processing fails unexpectedly."""

    failure: BaseException


StreamOutcome = (
    OperationBlocked
    | OperationCheckFailed
    | OperationModificationUnsupported
    | OperationProjectionFailed
    | StreamUpstreamFailed
    | StreamProcessingFailed
)
StreamOutcomeRenderer = Callable[[StreamOutcome], BufferedHttpResponse]


@dataclass(frozen=True, slots=True)
class _PendingStreamEvent:
    event: ServerSentEvent
    has_guarded_text: bool


@dataclass(frozen=True, slots=True)
class _TextWindow:
    processing: tuple[str, ...]
    release_count: int


class _UpstreamStreamFailure(Exception):
    def __init__(self, failure: BaseException):
        super().__init__(str(failure))
        self.failure = failure


def validate_streaming_policy(streaming_policy: StreamBufferingPolicy | None) -> None:
    """Require output streams to buffer every guarded delta before release."""

    if streaming_policy is None:
        raise UnsupportedStreamInspection("Guarded output streaming requires a buffering policy.")
    if streaming_policy.release_before_check:
        raise UnsupportedStreamInspection("Guarded streams cannot release output before checking it.")


def _take_checked_events(
    pending: list[_PendingStreamEvent],
    text_count: int,
) -> tuple[ServerSentEvent, ...]:
    checked_count = 0
    end = 0
    for end, event in enumerate(pending, start=1):
        if event.has_guarded_text:
            checked_count += 1
        if checked_count == text_count:
            break
    checked = pending[:end]
    del pending[:end]
    return tuple(event.event for event in checked)


async def _text_windows(
    source: AsyncIterator[str | ServerSentEvent],
    *,
    chunk_size: int,
    context_size: int,
) -> AsyncIterator[_TextWindow | ServerSentEvent]:
    context: list[str] = []
    pending: list[str] = []
    async for text in source:
        if isinstance(text, ServerSentEvent):
            yield text
            continue
        pending.append(text)
        if len(pending) < chunk_size:
            continue
        processing = context + pending
        yield _TextWindow(tuple(processing), len(pending))
        context = processing[-context_size:] if context_size else []
        pending = []
    if pending:
        yield _TextWindow(tuple(context + pending), len(pending))


async def _close_source(source: AsyncIterator[bytes]) -> None:
    close = getattr(source, "aclose", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


async def _guard_upstream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    try:
        async for chunk in source:
            if not isinstance(chunk, bytes):
                raise TypeError("A provider stream must yield bytes.")
            yield chunk
    except Exception as failure:
        raise _UpstreamStreamFailure(failure) from failure


def _encoded_outcome(
    outcome: StreamOutcome,
    *,
    adapter: ProviderStreamAdapter,
    render_outcome: StreamOutcomeRenderer,
) -> tuple[bytes, ...]:
    response = render_outcome(outcome)
    return tuple(adapter.encode_error(response.body))


async def guard_provider_stream(
    source: AsyncIterator[bytes],
    *,
    checker: ContentChecker,
    streaming_policy: StreamBufferingPolicy | None,
    input_message: GuardedMessage,
    adapter: ProviderStreamAdapter,
    render_outcome: StreamOutcomeRenderer,
    max_event_bytes: int = 1024 * 1024,
    max_pending_bytes: int = 10 * 1024 * 1024,
) -> AsyncIterator[bytes]:
    """Release provider SSE events only after required output checks pass.

    The current execution model requires one projected user input subject as
    context for assistant-output checks. This is a provider-neutral execution
    invariant, not a restriction on roles in the provider's raw conversation.
    Bindings supporting other input subjects require corresponding checker and
    execution support.

    Change this invariant only when the framework implements a capability for
    other input roles or multiple guarded input subjects, with matching bindings,
    checker semantics, and buffered and streaming execution tests. Adding a
    provider or changing a profile declaration alone does not relax it.

    Iteration closes ``source`` when it ends, including when arguments are
    rejected. A stream that is never iterated does not run that cleanup, so its
    caller must close ``source``. Validated data-less keepalive prefixes are
    released promptly; they never overtake pending payloads or terminal events.
    """

    try:
        if input_message.role != "user":
            raise ValueError("A guarded stream requires a user input message.")
        if any(type(limit) is not int or limit <= 0 for limit in (max_event_bytes, max_pending_bytes)):
            raise ValueError("Stream byte limits must be positive.")
    except ValueError:
        await _close_source(source)
        raise

    try:
        if streaming_policy is None:
            async for chunk in source:
                yield chunk
            return
        validate_streaming_policy(streaming_policy)

        pending: list[_PendingStreamEvent] = []
        pending_bytes = 0
        observed_text: list[str] = []
        observed_text_bytes = 0
        snapshot_history_complete = True

        async def text_deltas() -> AsyncIterator[str | ServerSentEvent]:
            nonlocal observed_text_bytes, pending_bytes, snapshot_history_complete
            async for event in iter_sse_events(_guard_upstream(source), max_event_bytes=max_event_bytes):
                classified = adapter.classify_event(event)
                validate_stream_event(adapter.contract, classified)
                if not pending and event.data is None and classified.role is StreamEventRole.OPAQUE_METADATA:
                    yield event
                    continue
                if classified.role is StreamEventRole.TEXT_SNAPSHOT:
                    if not snapshot_history_complete:
                        raise StreamSnapshotHistoryTooLarge(
                            "Provider stream text exceeds the configured snapshot verification limit."
                        )
                    if classified.text != "".join(observed_text):
                        raise UnsupportedProviderStream(
                            "A provider stream text snapshot does not match the guarded text deltas."
                        )
                text = classified.text if classified.role is StreamEventRole.GUARDED_TEXT else None
                pending.append(_PendingStreamEvent(event, has_guarded_text=bool(text)))
                pending_bytes += len(event.raw)
                if pending_bytes > max_pending_bytes:
                    raise PendingStreamBufferTooLarge(
                        "Unchecked provider stream events exceed the configured buffering limit."
                    )
                if text:
                    if snapshot_history_complete:
                        observed_text_bytes += len(text.encode())
                        if observed_text_bytes > max_event_bytes:
                            observed_text.clear()
                            snapshot_history_complete = False
                        else:
                            observed_text.append(text)
                    yield text
            adapter.validate_end_of_stream()

        async for window in _text_windows(
            text_deltas(),
            chunk_size=streaming_policy.chunk_size,
            context_size=streaming_policy.context_size,
        ):
            if isinstance(window, ServerSentEvent):
                yield window.raw
                continue
            checked_events = _take_checked_events(pending, window.release_count)
            pending_bytes -= sum(len(event.raw) for event in checked_events)
            try:
                decision = validate_content_check_decision(
                    await checker.check_output(OutputContentCheck(input_message, "".join(window.processing)))
                )
            except UnsupportedContentModification as failure:
                outcome: StreamOutcome = OperationModificationUnsupported(InspectionStage.OUTPUT, failure)
            except Exception as failure:
                outcome = OperationCheckFailed(
                    InspectionStage.OUTPUT,
                    ContentCheckFailed("The output content check failed.", failure),
                )
            else:
                if isinstance(decision, ContentAllowed):
                    for event in checked_events:
                        yield event.raw
                    continue
                if isinstance(decision, ContentBlocked):
                    outcome = OperationBlocked(InspectionStage.OUTPUT, decision)
                else:
                    outcome = OperationCheckFailed(InspectionStage.OUTPUT, decision)
            for chunk in _encoded_outcome(outcome, adapter=adapter, render_outcome=render_outcome):
                yield chunk
            return

        for event in pending:
            yield event.event.raw
    except (
        PendingStreamBufferTooLarge,
        ServerSentEventTooLarge,
        TruncatedServerSentEvent,
        StreamSnapshotHistoryTooLarge,
        UnsupportedProviderStream,
    ) as failure:
        outcome = OperationProjectionFailed(
            InspectionStage.OUTPUT,
            UnsupportedGuardedPayload(str(failure) or "The provider stream does not match the guarded content schema."),
        )
        for chunk in _encoded_outcome(outcome, adapter=adapter, render_outcome=render_outcome):
            yield chunk
    except _UpstreamStreamFailure as failure:
        for chunk in _encoded_outcome(
            StreamUpstreamFailed(failure.failure),
            adapter=adapter,
            render_outcome=render_outcome,
        ):
            yield chunk
    except Exception as failure:
        for chunk in _encoded_outcome(
            StreamProcessingFailed(failure),
            adapter=adapter,
            render_outcome=render_outcome,
        ):
            yield chunk
    finally:
        await _close_source(source)
