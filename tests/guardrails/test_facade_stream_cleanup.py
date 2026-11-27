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

"""Public stream closure awaits delegated provider and producer cleanup."""

import asyncio

import pytest

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponseChunk


class PausedModel(FakeLLMModel):
    def __init__(self, leading_content):
        super().__init__(responses=[])
        self.leading_content = leading_content
        self.paused = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        try:
            if self.leading_content:
                yield LLMResponseChunk(delta_content="first")
                yield LLMResponseChunk(delta_content="buffered")
            self.paused.set()
            await self.release.wait()
        finally:
            self.closed.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["close_after_content", "cancel_after_content", "cancel_before_content"])
async def test_public_stream_closes_provider_and_producer_immediately(mode):
    model = PausedModel(leading_content=mode != "cancel_before_content")
    rails = Guardrails(RailsConfig(models=[]), llm=model, require_iorails=True)
    stream = rails.stream_async(prompt="hello")
    initial_slots = rails.rails_engine._stream_semaphore._value
    consumer = None
    try:
        if model.leading_content:
            assert await asyncio.wait_for(anext(stream), 2) == "first"
        if mode == "cancel_after_content":
            assert await asyncio.wait_for(anext(stream), 2) == "buffered"
        if mode.startswith("cancel"):
            consumer = asyncio.create_task(anext(stream))
        await asyncio.wait_for(model.paused.wait(), 2)
        producers = {
            task for task in asyncio.all_tasks() if "_generation_task" in getattr(task.get_coro(), "__qualname__", "")
        }
        assert len(producers) == 1
        if consumer is None:
            await stream.aclose()
        else:
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
        assert model.closed.is_set()
        assert all(task.done() for task in producers)
        assert rails.rails_engine._stream_semaphore._value == initial_slots
        assert not model.release.is_set()
    finally:
        model.release.set()
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        await stream.aclose()
        await asyncio.wait_for(model.closed.wait(), 2)
        await rails.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_iorails", [True, False])
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_delegated_close_error_preserves_primary_outcome(monkeypatch, use_iorails, outcome):
    primary = ValueError("iteration failed") if outcome == "failed" else asyncio.CancelledError()
    cleanup = RuntimeError("cleanup failed")

    class Iterator:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if outcome == "completed":
                raise StopAsyncIteration
            raise primary

        async def aclose(self):
            self.closed = True
            raise cleanup

    rails = Guardrails(RailsConfig(models=[]), llm=FakeLLMModel(responses=[]), use_iorails=use_iorails)
    iterator = Iterator()
    monkeypatch.setattr(rails.rails_engine, "stream_async", lambda **kwargs: iterator)
    expected = cleanup if outcome == "completed" else primary
    try:
        with pytest.raises(type(expected)) as caught:
            await anext(rails.stream_async(prompt="hello"))
        assert caught.value is expected
        assert iterator.closed
    finally:
        await rails.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_iorails", [True, False])
async def test_generic_async_iterator_without_aclose_is_supported(monkeypatch, use_iorails):
    class Iterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    rails = Guardrails(RailsConfig(models=[]), llm=FakeLLMModel(responses=[]), use_iorails=use_iorails)
    monkeypatch.setattr(rails.rails_engine, "stream_async", lambda **kwargs: Iterator())
    try:
        assert [chunk async for chunk in rails.stream_async(prompt="hello")] == []
    finally:
        await rails.shutdown()


@pytest.mark.asyncio
async def test_failed_startup_does_not_create_legacy_stream_producer(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    rails = Guardrails(RailsConfig(models=[]), llm=FakeLLMModel(responses=[]), use_iorails=False)
    failure = RuntimeError("startup failed")
    factory = Mock(wraps=rails.rails_engine.stream_async)
    monkeypatch.setattr(rails, "_ensure_started", AsyncMock(side_effect=failure))
    monkeypatch.setattr(rails.rails_engine, "stream_async", factory)
    stream = rails.stream_async(prompt="hello")
    try:
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)
        assert caught.value is failure
        factory.assert_not_called()
        assert not getattr(rails.rails_engine, "_active_tasks", set())
    finally:
        await stream.aclose()
        await rails.shutdown()
