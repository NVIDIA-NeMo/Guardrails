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

"""Rails of a chat completion see the request's ``guardrails.thread_id`` as the conversation ID."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("openai", reason="openai is required for server tests")

from nemoguardrails import RailsConfig
from nemoguardrails.context import conversation_id, ensure_roundtrip_id, get_conversation_id, stream_with_request_ids
from nemoguardrails.server import api
from nemoguardrails.server.datastore.memory_store import MemoryStore
from tests.utils import TestChat

THREAD = "thread-0123456789abcdef"

CONFIG = RailsConfig.from_content(
    config={
        "models": [],
        "rails": {
            "input": {"flows": ["record input"]},
            "output": {
                "flows": ["record output"],
                "streaming": {"enabled": True, "chunk_size": 2, "context_size": 1, "stream_first": False},
            },
        },
    },
    colang_content="""
    define subflow record input
      execute record_conversation(stage="input")

    define subflow record output
      execute record_conversation(stage="output")
    """,
)


def _rails(seen: list, completions: int = 1, entered=None):
    chat = TestChat(CONFIG, llm_completions=["Hello there, how are you today?"] * completions, streaming=True)

    async def record_conversation(stage: str):
        if entered is not None and stage == "input":
            await entered()
        seen.append((stage, get_conversation_id()))
        return True

    chat.app.register_action(record_conversation, "record_conversation")
    return chat.app


class _LazyStreamRails:
    """Runs its rail while each chunk is pulled, as streaming output rails do."""

    config = CONFIG

    def __init__(self, seen: list, entered=None):
        self.seen = seen
        self.entered = entered

    def stream_async(self, messages, options):
        async def stream():
            if self.entered is not None:
                await self.entered()
            for chunk in ("Hello", " there"):
                self.seen.append(("output", get_conversation_id()))
                yield chunk

        # Both engines capture the conversation of the request that created the stream, so
        # the rails that run while a chunk is pulled see it wherever the body is drained.
        return stream_with_request_ids(stream(), conversation=get_conversation_id(), roundtrip=ensure_roundtrip_id())


def _payload(thread_id=None, stream=False):
    guardrails = {"config_id": "conversation"}
    if thread_id is not None:
        guardrails["thread_id"] = thread_id
    return {
        "model": "gpt-4o",
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
        "guardrails": guardrails,
    }


@pytest.fixture(autouse=True)
def datastore(monkeypatch):
    monkeypatch.setattr(api, "datastore", MemoryStore())


@pytest.fixture
def serve(monkeypatch):
    def serve(rails):
        monkeypatch.setattr(api, "_get_rails", AsyncMock(return_value=rails))

    return serve


async def _post(payload):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    return response


def _both_inside():
    """An awaitable that returns once two requests have reached it."""
    both_inside = asyncio.Event()
    inside = 0

    async def entered():
        nonlocal inside
        inside += 1
        if inside == 2:
            both_inside.set()
        await asyncio.wait_for(both_inside.wait(), timeout=5)

    return entered


FIRST, SECOND = "thread-aaaaaaaaaaaaaaaa", "thread-bbbbbbbbbbbbbbbb"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
async def test_rails_see_the_thread_id(serve, stream):
    seen = []
    serve(_rails(seen))

    await _post(_payload(THREAD, stream=stream))

    assert {stage for stage, _ in seen} == {"input", "output"}
    assert {conversation for _, conversation in seen} == {THREAD}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
async def test_no_conversation_id_without_a_thread_id(serve, stream):
    seen = []
    serve(_rails(seen))

    await _post(_payload(stream=stream))

    assert {stage for stage, _ in seen} == {"input", "output"}
    assert {conversation for _, conversation in seen} == {None}


@pytest.mark.asyncio
async def test_rails_run_while_streaming_see_the_thread_id(serve):
    seen = []
    serve(_LazyStreamRails(seen))

    response = await _post(_payload(THREAD, stream=True))

    assert "Hello" in response.text
    assert seen == [("output", THREAD), ("output", THREAD)]


@pytest.mark.asyncio
async def test_an_id_set_by_the_application_is_kept(serve):
    seen = []
    serve(_rails(seen))

    with conversation_id("app-set"):
        await _post(_payload(THREAD))

    assert {conversation for _, conversation in seen} == {"app-set"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
async def test_concurrent_requests_each_see_their_own_thread(serve, stream):
    seen = []
    serve(_rails(seen, completions=2, entered=_both_inside()))

    await asyncio.gather(_post(_payload(FIRST, stream=stream)), _post(_payload(SECOND, stream=stream)))

    for stage in ("input", "output"):
        assert sorted(conversation for s, conversation in seen if s == stage) == [FIRST, SECOND]


@pytest.mark.asyncio
async def test_concurrent_streams_each_see_their_own_thread(serve):
    seen = []
    serve(_LazyStreamRails(seen, entered=_both_inside()))

    await asyncio.gather(_post(_payload(FIRST, stream=True)), _post(_payload(SECOND, stream=True)))

    assert sorted(conversation for _, conversation in seen) == [FIRST, FIRST, SECOND, SECOND]
