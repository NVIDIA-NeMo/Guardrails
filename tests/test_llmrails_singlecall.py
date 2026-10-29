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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallChunk, ToolCallDeltaChunk
from tests.utils import TestChat


def test_1():
    """Test for single line single LLM call configuration."""
    config: RailsConfig = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting
        """
    )
    # Set the single call flag.
    config.rails.dialog.single_call.enabled = True

    chat = TestChat(
        config,
        llm_completions=[
            '  express greeting\nbot express greeting\n  "Hello, there!"',
        ],
    )

    chat >> "hello there!"
    (chat << "Hello, there!")


def test_2():
    """Test for single line single LLM call configuration when generated bot intent
    is different from the one in the flow."""
    config: RailsConfig = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define bot express greeting
            "Hello, there!"

        define flow
            user express greeting
            bot express greeting
        """
    )
    # Set the single call flag.
    config.rails.dialog.single_call.enabled = True

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting\n"
            "bot express greeting and present yourself\n"
            '  "Hello, there! I am a bot and I can help you with a lot of tasks!"',
        ],
    )

    chat >> "hello there!"
    (chat << "Hello, there!")


async def _collect_single_call_stream(rails):
    return [chunk async for chunk in rails.stream_async(messages=[{"role": "user", "content": "hello"}])]


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental", [False, True])
async def test_single_call_tool_stream_returns_error_instead_of_hanging(incremental, httpx_mock):
    config = RailsConfig.from_content(
        config={"models": [], "streaming": True, "rails": {"dialog": {"single_call": {"enabled": True}}}},
        colang_content='define user express greeting\n  "hello"\ndefine flow\n  user express greeting\n  bot express greeting\n',
    )
    if incremental:
        from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
        from nemoguardrails.llm.models.openai_chat import OpenAIChatModel

        payload = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": "{"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        httpx_mock.add_response(
            url="https://provider.test/v1/chat/completions",
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n",
        )
        model = OpenAIChatModel(
            OpenAICompatibleClient(base_url="https://provider.test/v1", api_key="test", max_retries=0),
            model="test-model",
        )
    else:
        model = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=[ToolCall(id="call_1")])])
    rails = LLMRails(config, llm=model)
    consumer = asyncio.create_task(_collect_single_call_stream(rails))
    try:
        done, _ = await asyncio.wait([consumer], timeout=3)
        assert consumer in done, "single-call generation did not propagate the producer failure"
        chunks = consumer.result()
    finally:
        tasks = [consumer, *rails._active_tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    errors = [
        json.loads(chunk)["error"] for chunk in chunks if isinstance(chunk, str) and chunk.startswith('{"error":')
    ]
    assert len(errors) == 1
    assert "Tool calls are not supported during buffered dialog generation." in errors[0]["message"]
    assert not any(isinstance(chunk, (ToolCallChunk, ToolCallDeltaChunk)) for chunk in chunks)


@pytest.mark.asyncio
async def test_single_call_text_still_streams_successfully():
    config = RailsConfig.from_content(
        config={"models": [], "streaming": True, "rails": {"dialog": {"single_call": {"enabled": True}}}},
        colang_content='define user express greeting\n  "hello"\ndefine flow\n  user express greeting\n  bot express greeting\n',
    )
    rails = LLMRails(
        config, llm=FakeLLMModel(responses=['  express greeting\nbot express greeting\n  "Hello, there!"'])
    )
    chunks = await asyncio.wait_for(_collect_single_call_stream(rails), timeout=3)
    assert "".join(chunks) == "Hello, there!"


@pytest.mark.asyncio
@pytest.mark.parametrize("single_call", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "body", ["She said “hello”\nand then waved.", "The phrase is “helpful bot”\nbot software can assist."]
)
async def test_straight_quoted_multiline_compatibility(single_call, streaming, body):
    config = RailsConfig.from_content(
        config={"models": [], "rails": {"dialog": {"single_call": {"enabled": single_call}}}},
        colang_content='define user express greeting\n  "hello"\ndefine flow\n  user express greeting\n  bot express greeting\n',
    )
    message = f'  "{body}"'
    responses = (
        [f"  express greeting\nbot express greeting\n{message}"] if single_call else ["  express greeting", message]
    )
    rails = LLMRails(config, llm=FakeLLMModel(responses=responses))
    if streaming:
        chunks = await asyncio.wait_for(_collect_single_call_stream(rails), timeout=3)
        assert "".join(chunks) == body
    else:
        response = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])
        assert response["content"] == body


@pytest.mark.asyncio
@pytest.mark.parametrize("single_call", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("opening,closing", [('"', '"'), ("“", "”")])
async def test_generated_message_quote_variants(single_call, streaming, verbose, opening, closing):
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"dialog": {"single_call": {"enabled": single_call}}},
            "prompts": [
                {
                    "task": "generate_intent_steps_message" if single_call else "generate_bot_message",
                    "content": "Generate a greeting.",
                    "models": ["unknown"],
                    "output_parser": "verbose_v1" if verbose else None,
                }
            ],
        },
        colang_content='define user express greeting\n  "hello"\ndefine flow\n  user express greeting\n  bot express greeting\n',
    )
    body = 'Hello! Say “hello” and "goodbye".'
    message = f"{'Bot message: ' if verbose else '  '}{opening}{body}{closing}"
    if single_call:
        intents = (
            "User intent: express greeting\nBot intent: express greeting\n"
            if verbose
            else "  express greeting\nbot express greeting\n"
        )
        responses = [intents + message]
    else:
        responses = ["  express greeting", message]
    model = FakeLLMModel(responses=responses)
    rails = LLMRails(config, llm=model)
    if streaming:
        chunks = await asyncio.wait_for(_collect_single_call_stream(rails), timeout=3)
        assert "".join(chunks) == body
    else:
        response = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])
        assert response["content"] == body
    assert model.inference_count == (1 if single_call else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_before_handoff", [False, True])
@pytest.mark.parametrize(
    "body,expected",
    [
        ("Hello!", "Hello!"),
        ("Hello, there!", "Hello, there!"),
        ('  "Hello!"', "Hello!"),
        ("  “Hello!”", "Hello!"),
    ],
)
async def test_single_call_stream_flushes_on_producer_end(monkeypatch, body, expected, finish_before_handoff):
    if finish_before_handoff:
        from nemoguardrails.streaming import StreamingHandler

        original_wait = StreamingHandler.wait_top_k_nonempty_lines

        async def wait_for_completed_producer(self, k):
            await self._producer_task
            return await original_wait(self, k)

        monkeypatch.setattr(StreamingHandler, "wait_top_k_nonempty_lines", wait_for_completed_producer)

    config = RailsConfig.from_content(
        config={"models": [], "rails": {"dialog": {"single_call": {"enabled": True}}}},
        colang_content='define user express greeting\n  "hello"\ndefine flow\n  user express greeting\n  bot express greeting\n',
    )
    model = FakeLLMModel(responses=[f"  express greeting\nbot express greeting\n{body}"])
    rails = LLMRails(config, llm=model)
    chunks = await asyncio.wait_for(_collect_single_call_stream(rails), timeout=3)
    assert "".join(chunks) == expected
    assert model.inference_count == 1
