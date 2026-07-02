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

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nemoguardrails.context import llm_stats_var, streaming_handler_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.rails.llm import llmrails as llmrails_module
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.runtime.colang_turns import (
    generate_colang_events,
    process_colang_events,
    run_colang_turn,
)
from nemoguardrails.streaming import END_OF_STREAM


class FakeRails:
    def __init__(self, colang_version: str, runtime, instant_actions=None):
        self.config = SimpleNamespace(
            colang_version=colang_version,
            rails=SimpleNamespace(
                actions=SimpleNamespace(instant_actions=instant_actions),
            ),
        )
        self.runtime = runtime
        self.verbose = False


def make_rails(colang_version: str, runtime, instant_actions=None):
    return FakeRails(colang_version=colang_version, runtime=runtime, instant_actions=instant_actions)


@pytest.fixture(autouse=True)
def reset_llm_stats_context():
    token = llm_stats_var.set(None)
    try:
        yield
    finally:
        llm_stats_var.reset(token)


@pytest.mark.asyncio
async def test_run_colang_turn_v1_prepends_state_events_and_passes_processing_log():
    class Runtime:
        def __init__(self):
            self.events = None
            self.processing_log = None

        async def generate_events(self, events, processing_log):
            self.events = events
            self.processing_log = processing_log
            processing_log.append({"event": "generated"})
            return [{"type": "BotIntent", "intent": "express greeting"}]

    runtime = Runtime()
    processing_log = []

    new_events = await run_colang_turn(
        make_rails("1.0", runtime),
        events=[{"type": "UserMessage", "text": "Hi"}],
        state={"events": [{"type": "ContextUpdate"}]},
        processing_log=processing_log,
    )

    assert runtime.events == [
        {"type": "ContextUpdate"},
        {"type": "UserMessage", "text": "Hi"},
    ]
    assert runtime.processing_log is processing_log
    assert processing_log == [{"event": "generated"}]
    assert new_events == [{"type": "BotIntent", "intent": "express greeting"}]


@pytest.mark.asyncio
async def test_run_colang_turn_v1_streams_error_chunk_before_reraising():
    class Runtime:
        async def generate_events(self, events, processing_log):
            raise RuntimeError("boom")

    class StreamingHandler:
        def __init__(self):
            self.chunks = []

        async def push_chunk(self, chunk):
            self.chunks.append(chunk)

    streaming_handler = StreamingHandler()
    token = streaming_handler_var.set(streaming_handler)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await run_colang_turn(
                make_rails("1.0", Runtime()),
                events=[{"type": "UserMessage", "text": "Hi"}],
                state=None,
                processing_log=[],
            )
    finally:
        streaming_handler_var.reset(token)

    assert json.loads(streaming_handler.chunks[0]) == {"error": {"message": "boom"}}
    assert streaming_handler.chunks[1] == END_OF_STREAM


@pytest.mark.asyncio
async def test_run_colang_turn_v2_processes_events_with_default_instant_actions():
    class Runtime:
        def __init__(self):
            self.calls = []

        async def process_events(self, events, state, instant_actions, blocking):
            self.calls.append(
                {
                    "events": events,
                    "state": state,
                    "instant_actions": instant_actions,
                    "blocking": blocking,
                }
            )
            return [{"type": "StartUtteranceBotAction", "script": "Hi"}], object()

    runtime = Runtime()
    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "hi"}]
    state = object()

    new_events = await run_colang_turn(
        make_rails("2.x", runtime),
        events=events,
        state=state,
        processing_log=[],
    )

    assert runtime.calls == [
        {
            "events": events,
            "state": state,
            "instant_actions": ["UtteranceBotAction"],
            "blocking": True,
        }
    ]
    assert new_events == [{"type": "StartUtteranceBotAction", "script": "Hi"}]


@pytest.mark.asyncio
async def test_run_colang_turn_v2_honors_configured_instant_actions():
    class Runtime:
        def __init__(self):
            self.instant_actions = None

        async def process_events(self, events, state, instant_actions, blocking):
            self.instant_actions = instant_actions
            return [], object()

    runtime = Runtime()
    await run_colang_turn(
        make_rails("2.x", runtime, instant_actions=["CustomAction"]),
        events=[],
        state=None,
        processing_log=[],
    )

    assert runtime.instant_actions == ["CustomAction"]


@pytest.mark.asyncio
async def test_generate_colang_events_sets_stats_and_passes_processing_log():
    class Runtime:
        def __init__(self):
            self.processing_log = None

        async def generate_events(self, events, processing_log):
            self.processing_log = processing_log
            processing_log.append({"event": "generated"})
            return [{"type": "BotIntent", "intent": "express greeting"}]

    runtime = Runtime()
    new_events = await generate_colang_events(
        make_rails("1.0", runtime),
        events=[{"type": "UserMessage", "text": "Hi"}],
    )

    assert new_events == [{"type": "BotIntent", "intent": "express greeting"}]
    assert runtime.processing_log == [{"event": "generated"}]
    assert isinstance(llm_stats_var.get(), LLMStats)


@pytest.mark.asyncio
async def test_process_colang_events_sets_stats_and_forwards_state_and_blocking():
    class Runtime:
        def __init__(self):
            self.calls = []

        async def process_events(self, events, state, blocking):
            self.calls.append((events, state, blocking))
            return [{"type": "OutputEvent"}], {"state": "updated"}

    runtime = Runtime()
    output_events, output_state = await process_colang_events(
        make_rails("2.x", runtime),
        events=[{"type": "InputEvent"}],
        state={"state": "input"},
        blocking=True,
        semaphore=asyncio.Semaphore(1),
    )

    assert runtime.calls == [([{"type": "InputEvent"}], {"state": "input"}, True)]
    assert output_events == [{"type": "OutputEvent"}]
    assert output_state == {"state": "updated"}
    assert isinstance(llm_stats_var.get(), LLMStats)


@pytest.mark.asyncio
async def test_process_colang_events_uses_explicit_semaphore_only():
    class Runtime:
        def __init__(self):
            self.active_calls = 0
            self.max_active_calls = 0

        async def process_events(self, events, state, blocking):
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            await asyncio.sleep(0)
            self.active_calls -= 1
            return events, state

    runtime = Runtime()
    rails = make_rails("2.x", runtime)

    await asyncio.gather(
        process_colang_events(rails, [{"type": "First"}], semaphore=asyncio.Semaphore(1)),
        process_colang_events(rails, [{"type": "Second"}], semaphore=asyncio.Semaphore(1)),
    )

    assert runtime.max_active_calls == 2


@pytest.mark.asyncio
async def test_process_colang_events_serializes_runtime_calls_with_shared_semaphore():
    class Runtime:
        def __init__(self):
            self.active_calls = 0
            self.max_active_calls = 0

        async def process_events(self, events, state, blocking):
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            await asyncio.sleep(0)
            self.active_calls -= 1
            return events, state

    runtime = Runtime()
    rails = make_rails("2.x", runtime)
    semaphore = asyncio.Semaphore(1)

    await asyncio.gather(
        process_colang_events(rails, [{"type": "First"}], semaphore=semaphore),
        process_colang_events(rails, [{"type": "Second"}], semaphore=semaphore),
    )

    assert runtime.max_active_calls == 1


@pytest.mark.asyncio
async def test_llmrails_process_events_uses_module_semaphore_patch_point():
    class Runtime:
        def __init__(self):
            self.calls = 0

        async def process_events(self, events, state, blocking):
            self.calls += 1
            return events, state

    runtime = Runtime()
    rails = cast(Any, object.__new__(LLMRails))
    rails.runtime = runtime

    original_semaphore = llmrails_module.process_events_semaphore
    llmrails_module.process_events_semaphore = asyncio.Semaphore(0)
    try:
        task = asyncio.create_task(rails.process_events_async([{"type": "InputEvent"}]))
        await asyncio.sleep(0)
        assert runtime.calls == 0

        llmrails_module.process_events_semaphore.release()
        await task
    finally:
        llmrails_module.process_events_semaphore = original_semaphore

    assert runtime.calls == 1
