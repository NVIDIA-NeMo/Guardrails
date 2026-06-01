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
from typing import Any, AsyncIterator, cast

import pytest

from nemoguardrails.rails.llm.buffer import ChunkBatch
from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig
from nemoguardrails.rails.llm.streaming import streaming_output_rails
from nemoguardrails.rails.llm.streaming.streaming_output_rails import (
    run_output_rails_in_streaming,
)


class FakeBufferStrategy:
    def __init__(self, batches: list[ChunkBatch]):
        self.batches = batches

    async def __call__(self, streaming_handler: AsyncIterator[str]) -> AsyncIterator[ChunkBatch]:
        del streaming_handler
        for batch in self.batches:
            yield batch

    def format_chunks(self, chunks: list[str]) -> str:
        return "".join(chunks)


class FakeActionDispatcher:
    def __init__(
        self,
        *,
        result: Any = True,
        status: str = "success",
        parallel_result: Any = None,
        parallel_status: str = "success",
        parallel_exception: Exception | None = None,
    ):
        self.result = result
        self.status = status
        self.parallel_result = parallel_result
        self.parallel_status = parallel_status
        self.parallel_exception = parallel_exception
        self.calls = []

    async def execute_action(self, action_name: str, params: dict[str, Any]):
        self.calls.append((action_name, params))

        if action_name == "run_output_rails_in_parallel_streaming":
            if self.parallel_exception:
                raise self.parallel_exception
            return self.parallel_result, self.parallel_status

        return self.result, self.status

    def get_action(self, action_name: str):
        del action_name
        return SimpleNamespace(action_meta={})


class FakeRuntime:
    action_dispatcher: FakeActionDispatcher
    llm_task_manager: Any
    registered_action_params: dict[str, Any]

    def __init__(self, dispatcher: FakeActionDispatcher):
        self.action_dispatcher = dispatcher
        self.llm_task_manager = "task-manager"
        self.registered_action_params = {
            "llms": {"main": "registered-llm"},
            "self_check_output_llm": "rail-llm",
        }


class FakeRails:
    def __init__(self, dispatcher: FakeActionDispatcher, *, parallel: bool = False):
        self.config = SimpleNamespace(
            flows=[],
            rails=SimpleNamespace(
                output=SimpleNamespace(
                    flows=["self check output"],
                    parallel=parallel,
                )
            ),
        )
        self.runtime = FakeRuntime(dispatcher)
        self.llm = "main-llm"
        self.explain_info = None
        self.ensure_explain_info_calls = 0

    def _ensure_explain_info(self):
        self.ensure_explain_info_calls += 1
        return {"ensured": self.ensure_explain_info_calls}


async def _empty_stream() -> AsyncIterator[str]:
    if False:
        yield ""


async def _tracked_stream(closed: Any, *chunks: str) -> AsyncIterator[str]:
    try:
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)
    finally:
        closed.set()


async def _collect(iterator: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in iterator]


def _patch_buffer_strategy(monkeypatch, batches: list[ChunkBatch]) -> FakeBufferStrategy:
    strategy = FakeBufferStrategy(batches)
    monkeypatch.setattr(streaming_output_rails, "get_buffer_strategy", lambda config: strategy)
    return strategy


def _patch_action_details(monkeypatch):
    def fake_get_action_details_from_flow_id(flow_id: str, flows: list):
        del flow_id, flows
        return (
            "self_check_output",
            {
                "bot_response": "$bot_message",
                "user_input": "$user_message",
                "literal": "value",
            },
        )

    monkeypatch.setattr(
        streaming_output_rails,
        "get_action_details_from_flow_id",
        fake_get_action_details_from_flow_id,
    )


def _patch_shared_action_details(monkeypatch, action_params: dict):
    def fake_get_action_details_from_flow_id(flow_id: str, flows: list):
        del flow_id, flows
        return "self_check_output", action_params

    monkeypatch.setattr(
        streaming_output_rails,
        "get_action_details_from_flow_id",
        fake_get_action_details_from_flow_id,
    )


@pytest.mark.asyncio
async def test_output_rails_stream_runs_sequential_rails_before_yielding(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["He", "llo"], user_output_chunks=["He", "llo"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(result=True)
    rails = FakeRails(dispatcher)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
            messages=[
                {"role": "context", "content": {"account_type": "enterprise"}},
                {"role": "user", "content": "Hi"},
            ],
        )
    )

    assert chunks == ["He", "llo"]
    assert dispatcher.calls[0][0] == "self_check_output"
    params = dispatcher.calls[0][1]
    assert params["bot_response"] == "Hello"
    assert params["user_input"] == {"role": "user", "content": "Hi"}
    assert params["literal"] == "value"
    assert params["context"] == {
        "account_type": "enterprise",
        "bot_message": "Hello",
        "user_message": {"role": "user", "content": "Hi"},
    }
    assert params["llm_task_manager"] == "task-manager"
    assert params["llms"] == {"main": "registered-llm"}
    assert params["llm"] == "rail-llm"
    assert rails.explain_info == {"ensured": 1}


@pytest.mark.asyncio
async def test_output_rails_stream_does_not_mutate_action_param_templates(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["He", "llo"], user_output_chunks=["He", "llo"])],
    )
    shared_action_params = {
        "bot_response": "$bot_message",
        "user_input": "$user_message",
    }
    _patch_shared_action_details(monkeypatch, shared_action_params)
    dispatcher = FakeActionDispatcher(result=True)
    rails = FakeRails(dispatcher)

    await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
            messages=[{"role": "user", "content": "Hi"}],
        )
    )

    assert shared_action_params == {
        "bot_response": "$bot_message",
        "user_input": "$user_message",
    }
    assert dispatcher.calls[0][1]["bot_response"] == "Hello"
    assert dispatcher.calls[0][1]["user_input"] == {"role": "user", "content": "Hi"}


@pytest.mark.asyncio
async def test_output_rails_stream_passes_through_json_error_chunks(monkeypatch):
    error_chunk = '{"error": {"message": "upstream failure"}}'
    _patch_buffer_strategy(
        monkeypatch,
        [
            ChunkBatch(
                processing_context=[],
                user_output_chunks=cast(Any, error_chunk),
            )
        ],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher()
    rails = FakeRails(dispatcher)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
        )
    )

    assert chunks == [error_chunk]
    assert dispatcher.calls == []
    assert rails.explain_info is None


@pytest.mark.asyncio
async def test_output_rails_stream_yields_block_error_for_sequential_rails(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["blocked"], user_output_chunks=["blocked"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(result=False)
    rails = FakeRails(dispatcher)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
        )
    )

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == {
        "error": {
            "message": "Blocked by self check output rails.",
            "type": "guardrails_violation",
            "param": "self check output",
            "code": "content_blocked",
        }
    }
    assert rails.explain_info == {"ensured": 1}


@pytest.mark.asyncio
async def test_output_rails_stream_honors_explicit_stream_first_false(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["blocked"], user_output_chunks=["blocked"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(result=False)
    rails = FakeRails(dispatcher)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=True),
            stream_first=False,
        )
    )

    assert len(chunks) == 1
    assert json.loads(chunks[0])["error"]["code"] == "content_blocked"


@pytest.mark.asyncio
async def test_output_rails_stream_closes_external_generator_after_block(monkeypatch):
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(result=False)
    rails = FakeRails(dispatcher)
    closed = asyncio.Event()

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _tracked_stream(closed, "blocked", "unused"),
            OutputRailsStreamingConfig(stream_first=False, chunk_size=1, context_size=0),
        )
    )
    await asyncio.wait_for(closed.wait(), timeout=1)

    assert len(chunks) == 1
    assert json.loads(chunks[0])["error"]["code"] == "content_blocked"


@pytest.mark.asyncio
async def test_output_rails_stream_closes_external_generator_when_consumer_closes(monkeypatch):
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(result=True)
    rails = FakeRails(dispatcher)
    closed = asyncio.Event()
    stream = run_output_rails_in_streaming(
        rails,
        _tracked_stream(closed, "first", "unused"),
        OutputRailsStreamingConfig(stream_first=False, chunk_size=1, context_size=0),
    )

    assert await stream.__anext__() == "first"
    await asyncio.wait_for(cast(Any, stream).aclose(), timeout=1)
    await asyncio.wait_for(closed.wait(), timeout=1)


@pytest.mark.asyncio
async def test_output_rails_stream_yields_internal_error_for_sequential_action_failure(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["unsafe"], user_output_chunks=["unsafe"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(result=None, status="failed")
    rails = FakeRails(dispatcher)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
        )
    )

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == {
        "error": {
            "message": "Internal error in self check output rail: Action self_check_output failed with status: failed",
            "type": "internal_error",
            "param": "self check output",
            "code": "rail_execution_failure",
        }
    }
    assert rails.explain_info == {"ensured": 1}


@pytest.mark.asyncio
async def test_output_rails_stream_yields_parallel_stop_event_error(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["He", "llo"], user_output_chunks=["He", "llo"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(
        parallel_result=SimpleNamespace(
            events=[
                {
                    "flow_id": "self check output",
                    "error_type": "internal_error",
                    "error_message": "action failed",
                }
            ]
        )
    )
    rails = FakeRails(dispatcher, parallel=True)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
            prompt="Hi",
            messages=[{"role": "context", "content": {"account_type": "enterprise"}}],
        )
    )

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == {
        "error": {
            "message": "Internal error in self check output rail: action failed",
            "type": "internal_error",
            "param": "self check output",
            "code": "rail_execution_failure",
        }
    }
    action_name, params = dispatcher.calls[0]
    assert action_name == "run_output_rails_in_parallel_streaming"
    assert params["events"] == [
        {
            "type": "ContextUpdate",
            "data": {
                "account_type": "enterprise",
                "bot_message": "Hello",
                "user_message": "Hi",
            },
        },
        {"type": "BotMessage", "text": "Hello"},
    ]
    assert params["flows_with_params"]["self check output"]["action_name"] == "self_check_output"
    assert params["flows_with_params"]["self check output"]["params"]["bot_response"] == "Hello"


@pytest.mark.asyncio
async def test_output_rails_stream_yields_internal_error_for_parallel_action_failure(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["unsafe"], user_output_chunks=["unsafe"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(parallel_status="failed")
    rails = FakeRails(dispatcher, parallel=True)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
        )
    )

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == {
        "error": {
            "message": "Internal error in output rails rail: Parallel rails execution failed with status: failed",
            "type": "internal_error",
            "param": "output rails",
            "code": "rail_execution_failure",
        }
    }


@pytest.mark.asyncio
async def test_output_rails_stream_yields_internal_error_for_parallel_exception(monkeypatch):
    _patch_buffer_strategy(
        monkeypatch,
        [ChunkBatch(processing_context=["unsafe"], user_output_chunks=["unsafe"])],
    )
    _patch_action_details(monkeypatch)
    dispatcher = FakeActionDispatcher(parallel_exception=RuntimeError("parallel boom"))
    rails = FakeRails(dispatcher, parallel=True)

    chunks = await _collect(
        run_output_rails_in_streaming(
            rails,
            _empty_stream(),
            OutputRailsStreamingConfig(stream_first=False),
        )
    )

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == {
        "error": {
            "message": "Internal error in output rails rail: Error in parallel rail execution: parallel boom",
            "type": "internal_error",
            "param": "output rails",
            "code": "rail_execution_failure",
        }
    }
