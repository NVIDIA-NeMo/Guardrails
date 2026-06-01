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
from typing import Any, AsyncIterator, Optional, cast

import pytest

from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.rails.llm.streaming import generation_stream
from nemoguardrails.rails.llm.streaming.generation_stream import (
    generation_token_stream,
    validate_streaming_with_output_rails,
)
from nemoguardrails.streaming import END_OF_STREAM


class FakeStreamingHandler:
    instances = []

    def __init__(self, include_metadata: Optional[bool] = False):
        self.include_metadata = include_metadata
        self.queue = asyncio.Queue()
        self.pushed_chunks = []
        self.__class__.instances.append(self)

    async def push_chunk(self, chunk):
        self.pushed_chunks.append(chunk)
        await self.queue.put(chunk)

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.queue.get()
        if chunk == END_OF_STREAM:
            raise StopAsyncIteration
        return chunk


class FakeRuntime:
    def __init__(self):
        self.action_dispatcher: Any = None
        self.llm_task_manager: Any = None
        self.registered_action_params: dict[str, Any] = {}


class FakeRails:
    def __init__(self, *, streaming=None, output_flows=None, generate_error: bool = False):
        self.config = SimpleNamespace(
            rails=SimpleNamespace(
                output=SimpleNamespace(
                    flows=output_flows or [],
                    streaming=streaming,
                )
            )
        )
        self.generate_error = generate_error
        self.generate_calls = []
        self.output_rail_calls = []
        self.explain_info = None
        self.runtime = FakeRuntime()
        self.llm = None

    def _ensure_explain_info(self):
        return {"explain": True}

    async def generate_async(
        self,
        *,
        prompt=None,
        messages=None,
        streaming_handler=None,
        options=None,
        state=None,
    ):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "streaming_handler": streaming_handler,
                "options": options,
                "state": state,
            }
        )
        if self.generate_error:
            raise RuntimeError('Error code: 500 - {"error": {"message": "boom"}}')
        assert streaming_handler is not None
        await streaming_handler.push_chunk(None)
        await streaming_handler.push_chunk("hello")
        await streaming_handler.push_chunk(END_OF_STREAM)


class SlowFakeRails(FakeRails):
    def __init__(self):
        super().__init__()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_async(
        self,
        *,
        prompt=None,
        messages=None,
        streaming_handler=None,
        options=None,
        state=None,
    ):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "messages": messages,
                "streaming_handler": streaming_handler,
                "options": options,
                "state": state,
            }
        )
        assert streaming_handler is not None
        await streaming_handler.push_chunk("hello")
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        await streaming_handler.push_chunk(END_OF_STREAM)


class BlockingOutputRailsFakeRails(SlowFakeRails):
    def __init__(self):
        super().__init__()
        self.config.rails.output.streaming = SimpleNamespace(enabled=True)


async def _collect(iterator):
    return [chunk async for chunk in iterator]


async def _tokens(*chunks) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


@pytest.fixture(autouse=True)
def fake_streaming_handler_and_output_rails(monkeypatch):
    FakeStreamingHandler.instances = []
    monkeypatch.setattr(generation_stream, "StreamingHandler", FakeStreamingHandler)

    def fake_run_output_rails_in_streaming(
        rails,
        streaming_handler,
        output_rails_streaming_config,
        prompt=None,
        messages=None,
        stream_first=None,
    ):
        rails.output_rail_calls.append(
            {
                "streaming_handler": streaming_handler,
                "output_rails_streaming_config": output_rails_streaming_config,
                "messages": messages,
                "prompt": prompt,
                "stream_first": stream_first,
            }
        )

        async def _wrapped():
            async for chunk in streaming_handler:
                if chunk is None:
                    continue
                if isinstance(rails, BlockingOutputRailsFakeRails):
                    yield json.dumps(
                        {
                            "error": {
                                "message": "Blocked by self check output rails.",
                                "type": "guardrails_violation",
                                "param": "self check output",
                                "code": "content_blocked",
                            }
                        }
                    )
                    return
                yield f"wrapped:{chunk}"

        return _wrapped()

    monkeypatch.setattr(
        generation_stream,
        "run_output_rails_in_streaming",
        fake_run_output_rails_in_streaming,
    )


@pytest.mark.asyncio
async def test_generation_token_stream_runs_generation_task_and_tracks_it():
    rails = FakeRails()

    stream = generation_token_stream(
        rails,
        messages=[{"role": "user", "content": "Hi"}],
        include_metadata=True,
    )
    chunks = await _collect(stream)
    await asyncio.sleep(0)

    assert chunks == ["hello"]
    assert rails.explain_info == {"explain": True}
    assert FakeStreamingHandler.instances[0].include_metadata is True
    assert rails.generate_calls[0]["messages"] == [{"role": "user", "content": "Hi"}]
    assert rails.generate_calls[0]["streaming_handler"] is FakeStreamingHandler.instances[0]
    assert cast(set, getattr(rails, "_active_tasks")) == set()


@pytest.mark.asyncio
async def test_generation_token_stream_honors_deprecated_metadata_flag():
    rails = FakeRails()

    with pytest.warns(DeprecationWarning, match="include_generation_metadata is deprecated"):
        stream = generation_token_stream(rails, include_generation_metadata=True)

    await _collect(stream)

    assert FakeStreamingHandler.instances[0].include_metadata is True


@pytest.mark.asyncio
async def test_generation_token_stream_returns_external_generator_without_output_rails():
    rails = FakeRails()
    generator = _tokens("a", "b")

    stream = generation_token_stream(rails, generator=generator)

    assert stream is generator
    assert await _collect(stream) == ["a", "b"]


@pytest.mark.asyncio
async def test_generation_token_stream_wraps_external_generator_when_output_rail_streaming_enabled():
    streaming_config = SimpleNamespace(enabled=True)
    rails = FakeRails(streaming=streaming_config)
    generator = _tokens("a")

    stream = generation_token_stream(
        rails,
        generator=generator,
        messages=[{"role": "user", "content": "Hi"}],
        prompt=None,
    )

    assert await _collect(stream) == ["wrapped:a"]
    assert rails.output_rail_calls == [
        {
            "streaming_handler": generator,
            "output_rails_streaming_config": streaming_config,
            "messages": [{"role": "user", "content": "Hi"}],
            "prompt": None,
            "stream_first": None,
        }
    ]


@pytest.mark.asyncio
async def test_generation_token_stream_wraps_internal_handler_when_output_rail_streaming_enabled():
    rails = FakeRails(streaming=SimpleNamespace(enabled=True))

    stream = generation_token_stream(rails, prompt="Hi")

    assert await _collect(stream) == ["wrapped:hello"]
    assert rails.output_rail_calls[0]["streaming_handler"] is FakeStreamingHandler.instances[0]
    assert rails.generate_calls[0]["prompt"] == "Hi"


def test_validate_streaming_with_output_rails_rejects_non_streaming_output_rails():
    config = SimpleNamespace(
        rails=SimpleNamespace(
            output=SimpleNamespace(
                flows=["self check output"],
                streaming=SimpleNamespace(enabled=False),
            )
        )
    )

    with pytest.raises(StreamingNotSupportedError, match="stream_async\\(\\) cannot be used"):
        validate_streaming_with_output_rails(config)


@pytest.mark.asyncio
async def test_generation_token_stream_pushes_error_payload_when_generation_fails():
    rails = FakeRails(generate_error=True)

    chunks = await _collect(generation_token_stream(rails, messages=[]))

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == {"error": {"message": "boom"}}
    assert FakeStreamingHandler.instances[0].pushed_chunks == [
        chunks[0],
        END_OF_STREAM,
    ]


@pytest.mark.asyncio
async def test_generation_token_stream_cancels_generation_when_consumer_closes_early():
    rails = SlowFakeRails()
    stream = generation_token_stream(rails, messages=[{"role": "user", "content": "Hi"}])

    assert await stream.__anext__() == "hello"
    await asyncio.wait_for(cast(Any, stream).aclose(), timeout=1)
    await asyncio.wait_for(rails.cancelled.wait(), timeout=1)
    await asyncio.sleep(0)

    assert cast(set, getattr(rails, "_active_tasks")) == set()


@pytest.mark.asyncio
async def test_generation_token_stream_cancels_generation_after_output_rail_error():
    rails = BlockingOutputRailsFakeRails()
    stream = generation_token_stream(rails, messages=[{"role": "user", "content": "Hi"}])

    chunks = await asyncio.wait_for(_collect(stream), timeout=1)
    await asyncio.wait_for(rails.cancelled.wait(), timeout=1)
    await asyncio.sleep(0)

    assert len(chunks) == 1
    assert json.loads(chunks[0])["error"]["code"] == "content_blocked"
    assert cast(set, getattr(rails, "_active_tasks")) == set()
