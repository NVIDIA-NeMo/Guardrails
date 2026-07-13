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

"""Streaming cancellation / teardown / partial-on-abort.

Cassette replay returns the whole recorded body in one shot, so it can never reproduce
a consumer that stops early or a cancelled task. ``stream_async`` returns the external
``generator`` directly when there are no streaming output rails, and wraps it via
``_run_output_rails_in_streaming`` otherwise; both paths must tear the source generator
down cleanly when the consumer aborts, delivering only the chunks produced before the
abort point (no buffered tail).

These tests drive a *tracking* async generator that yields a known prefix and then
blocks forever, so the consumer is the only thing that can end the stream. The
generator records (via a ``finally``) whether it was closed, which is the observable
proof that ``aclose()`` / cancellation propagated to the source. Fully deterministic
(no cassette), and identical behavior is expected on both pre-refactor and the
decomposed refactor.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator

import pytest

from nemoguardrails import LLMRails
from tests.recorded.rails.public_api.configs import (
    STREAMING_OUTPUT_RAILS_CONFIG,
    STREAMING_PASSTHROUGH_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


def _tracking_generator(values: list[str], teardown: dict) -> AsyncIterator[str]:
    """Yield ``values``, then block forever; record teardown via ``finally``.

    ``teardown["closed"]`` flips to True only when the generator is closed
    (``aclose()`` or a cancellation thrown in at the blocking ``await``), which is the
    observable proof that the consumer's abort propagated to the source.
    """

    async def gen() -> AsyncIterator[str]:
        try:
            for value in values:
                yield value
                await asyncio.sleep(0)
            while True:  # block: only the consumer can end this stream
                await asyncio.sleep(0.01)
        finally:
            teardown["closed"] = True

    return gen()


async def test_consumer_break_closes_external_generator():
    """Breaking the consumer delivers only the pre-abort prefix and closes the source."""
    teardown = {"closed": False}
    rails = LLMRails(load_config(STREAMING_PASSTHROUGH_CONFIG), llm=FakeLLMModel(responses=["unused"]), verbose=False)

    stream = rails.stream_async(
        messages=[{"role": "user", "content": "hi"}],
        generator=_tracking_generator(["alpha", "beta", "gamma"], teardown),
    )
    assert isinstance(stream, AsyncGenerator)
    received = []
    async for chunk in stream:
        received.append(chunk)
        if len(received) == 2:
            break
    await stream.aclose()

    # partial-on-abort: only the chunks produced before the break reached the consumer.
    assert received == ["alpha", "beta"]
    assert teardown["closed"] is True


async def test_consumer_break_through_output_rail_does_not_close_source_generator():
    """Pins a KNOWN GAP: aborting a streaming-output-rail-wrapped external generator
    does NOT propagate ``aclose()`` to the source generator.

    On the passthrough path (test above) breaking the consumer closes the source
    generator's ``finally`` synchronously. When streaming output rails wrap the
    generator (``run_output_rails_in_streaming``), that propagation is lost: closing the
    wrapper does not close the wrapped source, so a real source holding a network
    connection would leak until garbage collection. This behavior is **identical on
    pre-refactor and the decomposed refactor** (verified), i.e. a pre-existing gap, not a
    decomposition regression. The break itself is still clean (no exception, only the
    pre-abort prefix delivered).

    When the wrapper is fixed to propagate teardown, flip the final assertion to
    ``teardown["closed"] is True`` and rename this test.
    """
    teardown = {"closed": False}
    rails = LLMRails(load_config(STREAMING_OUTPUT_RAILS_CONFIG), verbose=False)

    stream = rails.stream_async(
        messages=[{"role": "user", "content": "stream"}],
        generator=_tracking_generator(["safe ", "more ", "tail "], teardown),
    )
    assert isinstance(stream, AsyncGenerator)
    received = []
    async for chunk in stream:
        received.append(chunk)
        if len(received) == 1:
            break
    await stream.aclose()  # clean abort: must not raise

    # KNOWN GAP (both branches): the wrapper did not close the source generator.
    assert teardown["closed"] is False


async def test_task_cancellation_propagates_cleanly():
    """Cancelling the consuming task raises CancelledError and closes the source generator."""
    teardown = {"closed": False}
    first_chunk = asyncio.Event()
    rails = LLMRails(load_config(STREAMING_PASSTHROUGH_CONFIG), llm=FakeLLMModel(responses=["unused"]), verbose=False)

    stream = rails.stream_async(
        messages=[{"role": "user", "content": "hi"}],
        generator=_tracking_generator(["alpha", "beta"], teardown),
    )
    received = []

    async def consume():
        async for chunk in stream:
            received.append(chunk)
            first_chunk.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_chunk.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert received[0] == "alpha"
    assert teardown["closed"] is True
