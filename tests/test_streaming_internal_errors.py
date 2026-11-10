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

"""Tests for streaming internal error handling in parallel output rails."""

import asyncio
import json
import os
from json.decoder import JSONDecodeError
from typing import Sequence

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.exceptions import LLMCallException, StreamingNotSupportedError
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.testing import FakeLLMModel
from tests.utils import TestChat

_has_langchain_openai = check_optional_dependency("langchain_openai")

_has_openai_key = bool(os.getenv("OPENAI_API_KEY"))


async def collect_streaming_chunks(stream):
    """Helper to collect all chunks from a streaming response."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


def find_internal_error_chunks(chunks):
    """Helper to find internal config error chunks from streaming response."""
    error_chunks = []
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed and parsed["error"].get("code") == "rail_execution_failure":
                error_chunks.append(parsed)
        except JSONDecodeError:
            continue
    return error_chunks


@pytest.mark.asyncio
async def test_streaming_action_execution_failure():
    """Test streaming internal error when action execution fails."""

    @action(is_system_action=True)
    def failing_rail_action(**params):
        raise RuntimeError("Action execution failed")

    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": ["failing safety check"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                    },
                }
            },
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot express greeting

        define subflow failing safety check
          execute failing_rail_action
        """,
    )

    llm_completions = [
        'bot express greeting\n  "Hello there! How can I help you?"',
    ]

    chat = TestChat(config, llm_completions=llm_completions, streaming=True)
    chat.app.register_action(failing_rail_action)

    chunks = await collect_streaming_chunks(chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]))

    internal_error_chunks = find_internal_error_chunks(chunks)
    assert len(internal_error_chunks) == 1, (
        f"Expected exactly one internal error chunk, got {len(internal_error_chunks)}"
    )

    error = internal_error_chunks[0]
    assert error["error"]["type"] == "internal_error"
    assert error["error"]["code"] == "rail_execution_failure"
    assert "Internal error" in error["error"]["message"]
    assert "failing safety check" in error["error"]["message"]
    assert "Action failing_rail_action failed with status: failed" in error["error"]["message"]
    assert error["error"]["param"] == "failing safety check"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_error",
    [
        pytest.param(RuntimeError("Action execution failed"), id="runtime-error"),
        pytest.param(LLMCallException(RuntimeError("LLM call failed")), id="llm-call-error"),
    ],
)
async def test_sequential_streaming_action_execution_failure(action_error: Exception):
    """Test fail-closed behavior in sequential streaming when a rail action fails."""

    @action(is_system_action=True)
    def failing_rail_action(**params):
        raise action_error

    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
            "rails": {
                "output": {
                    "parallel": False,
                    "flows": ["failing safety check"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                    },
                }
            },
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot express greeting

        define subflow failing safety check
          execute failing_rail_action
        """,
    )

    llm_completions = [
        'bot express greeting\n  "Hello there! How can I help you?"',
    ]

    chat = TestChat(config, llm_completions=llm_completions, streaming=True)
    chat.app.register_action(failing_rail_action)

    chunks = await collect_streaming_chunks(chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]))

    internal_error_chunks = find_internal_error_chunks(chunks)
    assert len(internal_error_chunks) == 1
    error = internal_error_chunks[0]["error"]
    assert error == {
        "message": "Internal error in failing safety check rail: Action failing_rail_action failed with status: failed",
        "type": "internal_error",
        "param": "failing safety check",
        "code": "rail_execution_failure",
    }


@pytest.mark.asyncio
async def test_sequential_streaming_invalid_rail_outcome():
    """Test fail-closed behavior when a sequential streaming rail returns an invalid result."""

    @action(is_system_action=True)
    def invalid_rail_action(**params):
        return False

    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
            "rails": {
                "output": {
                    "parallel": False,
                    "flows": ["invalid safety check"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                    },
                }
            },
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot express greeting

        define subflow invalid safety check
          execute invalid_rail_action
        """,
    )

    llm_completions = [
        'bot express greeting\n  "Hello there! How can I help you?"',
    ]

    chat = TestChat(config, llm_completions=llm_completions, streaming=True)
    chat.app.register_action(invalid_rail_action)

    chunks = await collect_streaming_chunks(chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]))

    internal_error_chunks = find_internal_error_chunks(chunks)
    assert len(internal_error_chunks) == 1
    error = internal_error_chunks[0]["error"]
    assert error == {
        "message": "Internal error in invalid safety check rail: Rail action must return RailOutcome, got bool",
        "type": "internal_error",
        "param": "invalid safety check",
        "code": "rail_execution_failure",
    }


@pytest.mark.asyncio
async def test_streaming_internal_error_format():
    """Test that streaming internal errors have the correct format."""

    @action(is_system_action=True)
    def test_failing_action(**params):
        raise ValueError("Test error message")

    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo"}],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": ["test rail"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                    },
                }
            },
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot express greeting

        define subflow test rail
          execute test_failing_action
        """,
    )

    llm_completions = [
        'bot express greeting\n  "Test response"',
    ]

    chat = TestChat(config, llm_completions=llm_completions, streaming=True)
    chat.app.register_action(test_failing_action)

    chunks = await collect_streaming_chunks(chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]))

    internal_error_chunks = find_internal_error_chunks(chunks)
    assert len(internal_error_chunks) == 1

    error = internal_error_chunks[0]

    assert "error" in error
    error_obj = error["error"]

    assert "type" in error_obj
    assert error_obj["type"] == "internal_error"

    assert "code" in error_obj
    assert error_obj["code"] == "rail_execution_failure"

    assert "message" in error_obj
    assert "Internal error in test rail rail:" in error_obj["message"]

    assert "param" in error_obj
    assert error_obj["param"] == "test rail"


BLOCKED_TEXT = "This response has a HIGHSCORE and must never reach the client."

PARALLEL_RAILS_ACTION = "run_output_rails_in_parallel_streaming"

DIRECT_EXECUTE_COLANG = """
define user express greeting
  "hi"

define flow
  user express greeting
  bot express greeting

define subflow blocking output check
  execute check_blocked
"""

TWO_RAIL_COLANG = (
    DIRECT_EXECUTE_COLANG
    + """
define subflow hanging output check
  execute check_hanging
"""
)

TWO_RAIL_FLOWS = ["blocking output check", "hanging output check"]

SUBFLOW_DELEGATING_COLANG = """
define user express greeting
  "hi"

define flow
  user express greeting
  bot express greeting

define flow blocking output check
  if $bot_message
    do inner blocking check

define subflow inner blocking check
  bot refuse to respond
  stop
"""


@action(is_system_action=True)
def check_blocked(context=None, **params):
    if "HIGHSCORE" in (context or {}).get("bot_message", ""):
        return RailOutcome.block()
    return RailOutcome.allow()


def _blocking_rail_chat(
    colang_content: str,
    parallel: bool,
    stream_first: bool = False,
    flows: Sequence[str] = ("blocking output check",),
) -> TestChat:
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": parallel,
                    "flows": list(flows),
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": stream_first,
                    },
                }
            },
        },
        colang_content=colang_content,
    )
    chat = TestChat(
        config,
        llm_completions=[
            '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
            f'  "{BLOCKED_TEXT}"',
        ],
        streaming=True,
    )
    chat.app.register_action(check_blocked)
    return chat


def _replace_parallel_rails_dispatch(monkeypatch, chat: TestChat, dispatch):
    dispatcher = chat.app.runtime.action_dispatcher
    original_execute_action = dispatcher.execute_action

    async def execute_action(action_name, params):
        if action_name == PARALLEL_RAILS_ACTION:
            return dispatch()
        return await original_execute_action(action_name, params)

    monkeypatch.setattr(dispatcher, "execute_action", execute_action)


def _content_chunks(chunks):
    return [chunk for chunk in chunks if not find_internal_error_chunks([chunk])]


async def _stream_greeting(chat: TestChat):
    return await collect_streaming_chunks(chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_first", [False, True], ids=["check-first", "stream-first"])
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_stream_async_rejects_output_flow_without_direct_execute(parallel: bool, stream_first: bool):
    """stream_async raises StreamingNotSupportedError at the call for an output flow the streaming path cannot run."""
    chat = _blocking_rail_chat(SUBFLOW_DELEGATING_COLANG, parallel=parallel, stream_first=stream_first)

    with pytest.raises(StreamingNotSupportedError, match="blocking output check"):
        chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_stream_async_rejects_output_flow_without_direct_execute_before_pulling_generator(parallel: bool):
    """stream_async rejects an unrunnable output flow before pulling anything from an external generator."""
    chat = _blocking_rail_chat(SUBFLOW_DELEGATING_COLANG, parallel=parallel)
    pulled = asyncio.Event()

    async def external_generator():
        pulled.set()
        yield BLOCKED_TEXT

    with pytest.raises(StreamingNotSupportedError, match="blocking output check"):
        chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}], generator=external_generator())
    assert not pulled.is_set()


@pytest.mark.asyncio
async def test_parallel_streaming_dispatch_failure_ends_stream(monkeypatch):
    """A non-success status from the parallel rails dispatch ends the stream with one failure frame and no content."""
    chat = _blocking_rail_chat(DIRECT_EXECUTE_COLANG, parallel=True)
    _replace_parallel_rails_dispatch(monkeypatch, chat, lambda: (None, "failed"))

    chunks = await _stream_greeting(chat)

    assert _content_chunks(chunks) == []
    assert [frame["error"] for frame in find_internal_error_chunks(chunks)] == [
        {
            "message": f"Internal error in output rails: Action {PARALLEL_RAILS_ACTION} failed with status: failed",
            "type": "internal_error",
            "param": "output rails",
            "code": "rail_execution_failure",
        }
    ]


@pytest.mark.asyncio
async def test_parallel_streaming_dispatch_exception_ends_stream_without_exception_text(monkeypatch):
    """An exception from the parallel rails dispatch ends the stream with one failure frame that omits the exception text."""
    chat = _blocking_rail_chat(DIRECT_EXECUTE_COLANG, parallel=True)

    def raise_llm_call_exception():
        raise LLMCallException(RuntimeError("secret-detail"))

    _replace_parallel_rails_dispatch(monkeypatch, chat, raise_llm_call_exception)

    chunks = await _stream_greeting(chat)

    assert _content_chunks(chunks) == []
    frames = find_internal_error_chunks(chunks)
    assert len(frames) == 1
    assert frames[0]["error"]["param"] == "output rails"
    assert "secret-detail" not in json.dumps(frames[0])


@pytest.mark.asyncio
async def test_parallel_streaming_dispatch_without_action_result_ends_stream(monkeypatch):
    """A successful parallel rails dispatch that returns no ActionResult ends the stream with one failure frame."""
    chat = _blocking_rail_chat(DIRECT_EXECUTE_COLANG, parallel=True)
    _replace_parallel_rails_dispatch(monkeypatch, chat, lambda: (None, "success"))

    chunks = await _stream_greeting(chat)

    assert _content_chunks(chunks) == []
    frames = find_internal_error_chunks(chunks)
    assert len(frames) == 1
    assert frames[0]["error"]["param"] == "output rails"


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_streaming_rail_preparation_error_ends_stream(parallel: bool):
    """A rail-parameter preparation error ends the stream with one failure frame and no content in both modes."""
    chat = _blocking_rail_chat(DIRECT_EXECUTE_COLANG, parallel=parallel)

    async def external_generator():
        for token in BLOCKED_TEXT.split(" "):
            yield token + " "

    chunks = await collect_streaming_chunks(
        chat.app.stream_async(
            messages=[{"role": "context", "content": "not-a-dict"}, {"role": "user", "content": "Hi!"}],
            generator=external_generator(),
        )
    )

    assert _content_chunks(chunks) == []
    assert len(find_internal_error_chunks(chunks)) == 1


@pytest.mark.asyncio
async def test_sequential_streaming_runs_a_flow_listed_twice_twice_per_chunk():
    """Sequential streaming runs an output flow listed twice on each chunk twice, so its second run can block the first chunk."""
    chat = _blocking_rail_chat(DIRECT_EXECUTE_COLANG, parallel=False, flows=["blocking output check"] * 2)
    calls = 0

    async def allow_then_block(**params):
        nonlocal calls
        calls += 1
        if calls == 1:
            return RailOutcome.allow()
        return RailOutcome.block()

    chat.app.register_action(allow_then_block, "check_blocked")

    chunks = await _stream_greeting(chat)

    assert len(chunks) == 1, chunks
    assert json.loads(chunks[0]) == {
        "error": {
            "message": "Blocked by blocking output check rails.",
            "type": "guardrails_violation",
            "param": "blocking output check",
            "code": "content_blocked",
        }
    }
    assert calls == 2


class _GatedRail:
    """A fake output rail action that records the task each call runs in and holds the call until its gate opens."""

    def __init__(self):
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.tasks = []

    async def check(self, **params):
        self.tasks.append(asyncio.current_task())
        self.started.set()
        await self.gate.wait()
        return RailOutcome.allow()


async def _assert_rail_tasks_cancelled(*rails: _GatedRail):
    """Open each rail's gate and assert that every call was cancelled instead of returning."""
    for rail in rails:
        rail.gate.set()
    tasks = [task for rail in rails for task in rail.tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    assert tasks
    assert [task.cancelled() for task in tasks] == [True] * len(tasks)


@pytest.mark.asyncio
async def test_parallel_streaming_cancellation_during_rails_propagates_and_cancels_every_rail():
    """Cancelling a stream while parallel output rails run raises CancelledError, releases no chunk and cancels every rail."""
    chat = _blocking_rail_chat(TWO_RAIL_COLANG, parallel=True, flows=TWO_RAIL_FLOWS)
    blocking_rail = _GatedRail()
    hanging_rail = _GatedRail()
    chat.app.register_action(blocking_rail.check, "check_blocked")
    chat.app.register_action(hanging_rail.check, "check_hanging")
    chunks = []

    async def consume():
        async for chunk in chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]):
            chunks.append(chunk)

    consumer = asyncio.create_task(consume())
    rails_started = asyncio.gather(blocking_rail.started.wait(), hanging_rail.started.wait())
    await asyncio.wait({consumer, rails_started}, return_when=asyncio.FIRST_COMPLETED)
    rails_started.cancel()
    await asyncio.gather(rails_started, return_exceptions=True)
    assert blocking_rail.started.is_set() and hanging_rail.started.is_set(), f"The stream ended first: {chunks}"
    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert chunks == []
    await _assert_rail_tasks_cancelled(blocking_rail, hanging_rail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rail_fails", "expected_code"),
    [(False, "content_blocked"), (True, "rail_execution_failure")],
    ids=["blocks", "fails"],
)
async def test_parallel_streaming_stop_cancels_running_sibling_rail(rail_fails: bool, expected_code: str):
    """A parallel output rail that blocks or fails ends the stream with its frame and cancels the rail still running."""
    chat = _blocking_rail_chat(TWO_RAIL_COLANG, parallel=True, flows=TWO_RAIL_FLOWS)
    hanging_rail = _GatedRail()

    async def stop_once_sibling_runs(**params):
        await hanging_rail.started.wait()
        if rail_fails:
            raise RuntimeError("rail failed")
        return RailOutcome.block()

    chat.app.register_action(stop_once_sibling_runs, "check_blocked")
    chat.app.register_action(hanging_rail.check, "check_hanging")

    chunks = await _stream_greeting(chat)

    assert len(chunks) == 1, chunks
    error = json.loads(chunks[0])["error"]
    assert (error["param"], error["code"]) == ("blocking output check", expected_code)
    await _assert_rail_tasks_cancelled(hanging_rail)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_streaming_rail_action_exception_frame_omits_exception_text(parallel: bool):
    """A rail action that raises ends the stream with the same generic failure frame in both modes, without its text."""
    chat = _blocking_rail_chat(DIRECT_EXECUTE_COLANG, parallel=parallel)

    async def raising_check(**params):
        raise LLMCallException(RuntimeError("secret-detail"))

    chat.app.register_action(raising_check, "check_blocked")

    chunks = await _stream_greeting(chat)

    assert _content_chunks(chunks) == []
    assert [frame["error"] for frame in find_internal_error_chunks(chunks)] == [
        {
            "message": "Internal error in blocking output check rail: Action check_blocked failed with status: failed",
            "type": "internal_error",
            "param": "blocking output check",
            "code": "rail_execution_failure",
        }
    ]


@pytest.mark.asyncio
async def test_parallel_streaming_without_output_rail_flows_releases_chunks():
    """Parallel output rail streaming with no output rail flows releases every chunk, even on Colang 2.x."""
    config = RailsConfig.from_content(
        yaml_content="""
colang_version: "2.x"
models: []
rails:
  output:
    parallel: True
    streaming:
      enabled: True
      chunk_size: 4
      context_size: 2
      stream_first: False
""",
        colang_content='import core\n\nflow main\n  activate greeting\n\nflow greeting\n  user said "hi"\n  bot say "hello"\n',
    )
    rails = LLMRails(config, llm=FakeLLMModel(responses=[]))

    async def external_generator():
        for token in BLOCKED_TEXT.split(" "):
            yield token + " "

    chunks = await collect_streaming_chunks(
        rails.stream_async(messages=[{"role": "user", "content": "Hi!"}], generator=external_generator())
    )

    assert find_internal_error_chunks(chunks) == []
    assert "".join(chunks).strip() == BLOCKED_TEXT
