# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import time
from unittest.mock import patch

import pytest

langchain_nvidia_ai_endpoints = pytest.importorskip("langchain_nvidia_ai_endpoints")

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402

from nemoguardrails.llm.providers._langchain_nvidia_ai_endpoints_patch import ChatNVIDIA  # noqa: E402

LIVE_TEST_MODE = os.environ.get("LIVE_TEST_MODE")


class FakeCallbackHandler:
    def __init__(self):
        self.llm_streams = 0
        self.tokens = []

    async def on_llm_new_token(self, token: str, **kwargs):
        self.llm_streams += 1
        self.tokens.append(token)


class TestAsyncStreamDecorator:
    @pytest.mark.asyncio
    async def test_decorator_with_streaming_enabled(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=True,
        )

        messages = [HumanMessage(content="Hello")]

        with patch.object(chat, "_astream") as mock_astream:
            mock_chunk = ChatGenerationChunk(message=AIMessageChunk(content="Hi there"))
            mock_astream.return_value = AsyncIteratorMock([mock_chunk])

            result = await chat._agenerate(messages)

            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hi there"
            mock_astream.assert_called_once()

    @pytest.mark.asyncio
    async def test_decorator_with_streaming_disabled(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=False,
        )

        messages = [HumanMessage(content="Hello")]

        with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA._agenerate") as mock_parent_agenerate:
            expected_result = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Response from parent"))]
            )
            mock_parent_agenerate.return_value = expected_result

            result = await chat._agenerate(messages)

            assert result == expected_result
            mock_parent_agenerate.assert_called_once()

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
        )

        assert chat._agenerate.__name__ == "_agenerate"
        assert asyncio.iscoroutinefunction(chat._agenerate)

    @pytest.mark.asyncio
    async def test_streaming_aggregates_multiple_chunks(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=True,
        )

        messages = [HumanMessage(content="Hello")]

        with patch.object(chat, "_astream") as mock_astream:
            chunks = [
                ChatGenerationChunk(message=AIMessageChunk(content="Hello ")),
                ChatGenerationChunk(message=AIMessageChunk(content="world")),
                ChatGenerationChunk(message=AIMessageChunk(content="!")),
            ]
            mock_astream.return_value = AsyncIteratorMock(chunks)

            result = await chat._agenerate(messages)

            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello world!"
            mock_astream.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=True,
        )

        messages = [HumanMessage(content="Hello")]

        with patch.object(chat, "_astream") as mock_astream:
            chunks = [
                ChatGenerationChunk(message=AIMessageChunk(content="")),
                ChatGenerationChunk(message=AIMessageChunk(content="Hello")),
                ChatGenerationChunk(message=AIMessageChunk(content="")),
            ]
            mock_astream.return_value = AsyncIteratorMock(chunks)

            result = await chat._agenerate(messages)

            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello"


class TestChatNVIDIAPatch:
    @pytest.mark.asyncio
    async def test_agenerate_calls_patched_agenerate(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=False,
        )

        messages = [[HumanMessage(content="Hello")], [HumanMessage(content="Hi")]]

        with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA._agenerate") as mock_parent:
            mock_parent.return_value = ChatResult(generations=[ChatGeneration(message=AIMessage(content="Response"))])

            result = await chat.agenerate(messages)

            assert isinstance(result.generations, list)
            assert len(result.generations) == 2
            for generation_list in result.generations:
                assert len(generation_list) == 1
                assert generation_list[0].message.content == "Response"
            assert mock_parent.call_count == 2

    @pytest.mark.asyncio
    async def test_agenerate_with_streaming_enabled(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=True,
        )

        messages = [[HumanMessage(content="Hello")]]

        with patch.object(chat, "_astream") as mock_astream:
            chunks = [
                ChatGenerationChunk(message=AIMessageChunk(content="Hello ")),
                ChatGenerationChunk(message=AIMessageChunk(content="world")),
            ]
            mock_astream.return_value = AsyncIteratorMock(chunks)

            result = await chat.agenerate(messages)

            assert isinstance(result.generations, list)
            assert len(result.generations) == 1
            assert len(result.generations[0]) == 1
            assert result.generations[0][0].message.content == "Hello world"
            mock_astream.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_field_exists(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
        )

        assert hasattr(chat, "streaming")
        assert not chat.streaming

        chat_with_streaming = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=True,
        )
        assert chat_with_streaming.streaming

    @pytest.mark.asyncio
    async def test_backward_compatibility_sync_generate(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=False,
        )

        messages = [[HumanMessage(content="Hello")]]

        with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA._generate") as mock_parent:
            mock_parent.return_value = ChatResult(generations=[ChatGeneration(message=AIMessage(content="Response"))])

            result = chat.generate(messages)

            assert isinstance(result.generations, list)
            assert len(result.generations[0]) == 1
            assert result.generations[0][0].message.content == "Response"
            mock_parent.assert_called()

    @pytest.mark.asyncio
    async def test_streaming_handles_multiple_message_batches(self):
        chat = ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            base_url="http://localhost:8000/v1",
            streaming=True,
        )

        messages = [
            [HumanMessage(content="First message")],
            [HumanMessage(content="Second message")],
        ]

        with patch.object(chat, "_astream") as mock_astream:
            mock_astream.side_effect = [
                AsyncIteratorMock(
                    [
                        ChatGenerationChunk(message=AIMessageChunk(content="First ")),
                        ChatGenerationChunk(message=AIMessageChunk(content="response")),
                    ]
                ),
                AsyncIteratorMock(
                    [
                        ChatGenerationChunk(message=AIMessageChunk(content="Second ")),
                        ChatGenerationChunk(message=AIMessageChunk(content="response")),
                    ]
                ),
            ]

            result = await chat.agenerate(messages)

            assert len(result.generations) == 2
            assert result.generations[0][0].message.content == "First response"
            assert result.generations[1][0].message.content == "Second response"
            assert mock_astream.call_count == 2


class TestIntegrationWithLLMRails:
    @pytest.mark.asyncio
    async def test_chatnvidia_with_llmrails_async(self):
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_content(
            config={
                "models": [
                    {
                        "type": "main",
                        "engine": "nim",
                        "model": "meta/llama-3.3-70b-instruct",
                    }
                ]
            }
        )

        async def mock_agenerate_func(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Test response"))])

        with patch(
            "langchain_nvidia_ai_endpoints.ChatNVIDIA._agenerate",
            new=mock_agenerate_func,
        ):
            rails = LLMRails(config)

            result = await rails.generate_async(messages=[{"role": "user", "content": "Hello"}])

            assert result is not None
            assert "content" in result
            assert result["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_chatnvidia_streaming_with_llmrails(self):
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_content(
            config={
                "models": [
                    {
                        "type": "main",
                        "engine": "nim",
                        "model": "meta/llama-3.3-70b-instruct",
                        "parameters": {"streaming": True},
                    }
                ],
                "streaming": True,
            }
        )

        rails = LLMRails(config)

        chat_model = rails.llm

        assert hasattr(chat_model, "streaming")
        assert chat_model.streaming


class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.mark.skipif(
    not LIVE_TEST_MODE,
    reason="This test requires LIVE_TEST_MODE environment variable to be set for live testing",
)
class TestChatNVIDIAStreamingE2E:
    @pytest.mark.asyncio
    async def test_stream_async_ttft_with_nim(self):
        from nemoguardrails import LLMRails, RailsConfig

        yaml_content = """
models:
  - type: main
    engine: nim
    model: meta/llama-3.3-70b-instruct

streaming: True
"""
        config = RailsConfig.from_content(yaml_content=yaml_content)
        rails = LLMRails(config)

        chunk_times = [time.time()]
        chunks = []

        async for chunk in rails.stream_async(
            messages=[{"role": "user", "content": "Count to 20 by 2s, e.g. 2 4 6 8 ..."}]
        ):
            chunks.append(chunk)
            chunk_times.append(time.time())

        ttft = chunk_times[1] - chunk_times[0]
        total_time = chunk_times[-1] - chunk_times[0]

        assert len(chunks) > 0, "Should receive at least one chunk"
        assert ttft < (total_time / 2), f"TTFT ({ttft:.3f}s) should be less than half of total time ({total_time:.3f}s)"
        assert len(chunk_times) > 2, "Should receive multiple chunks for streaming"

        full_response = "".join(chunks)
        assert len(full_response) > 0, "Full response should not be empty"
