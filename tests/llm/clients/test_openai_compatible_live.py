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

import asyncio
import json
import os
from pathlib import Path

import pytest

from nemoguardrails.exceptions import (
    LLMAuthenticationError,
    LLMClientError,
    LLMContextWindowError,
)
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.models.openai_chat import OpenAIChatModel
from nemoguardrails.types import ChatMessage, LLMResponse, Role
from tests.llm.clients._helpers import (
    LIVE_TEST_MODE,
    NIM_BASE_URL,
    OPENAI_BASE_URL,
    live_mode_enabled,
    live_nim_model,
    live_openai_model,
    simulated_model,
    simulated_model_sequenced,
)


def _make_model(model, base_url, api_key=None, **kwargs):
    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(client=client, model=model, **kwargs)


FIXTURES_DIR = Path(__file__).parent / "fixtures"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def _load_fixture(name):
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _to_http_response(data):
    from nemoguardrails.llm.clients.base import HTTPResponse

    if isinstance(data, dict):
        headers = data.get("_response_headers", {})
        body = {key: value for key, value in data.items() if key != "_response_headers"}
    else:
        headers, body = {}, data
    return HTTPResponse(body=body, headers=headers, status_code=200)


def _load_response_fixture(name):
    return _to_http_response(_load_fixture(name))


def _load_stream_fixture(name):
    return [_to_http_response(c) for c in _load_fixture(name)]


def _fixture_exists(name):
    return (FIXTURES_DIR / name).exists()


def _replay_stream(model, fixture_name):
    chunks_data = _load_stream_fixture(fixture_name)

    async def replay(*args, **kwargs):
        for chunk_data in chunks_data:
            yield chunk_data

    async def run():
        model._client.stream_chat_completion = replay
        results = []
        async for chunk in model.stream_async("test"):
            results.append(chunk)
        return results

    return asyncio.run(run())


class TestRecordedOpenAI:
    """Parse real OpenAI API responses captured as JSON fixtures.

    These tests feed actual API responses (recorded via record_fixtures.py) through
    our _parse_response and _parse_chunk methods. They prove our parser handles the
    real response shape - including fields like refusal, annotations, logprobs,
    system_fingerprint that hand-written mocks would miss.

    If OpenAI changes their response format, re-record fixtures with:
        poetry run python tests/llm/clients/record_fixtures.py
    """

    def test_parse_text_response(self):
        data = _load_response_fixture("openai_generate_text.json")
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        result = client._parse_response(data)

        assert isinstance(result, LLMResponse)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.model == "gpt-4o-mini-2024-07-18"
        assert result.finish_reason == "stop"
        assert result.tool_calls is None
        assert result.request_id is not None
        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens
        assert result.provider_metadata is not None
        assert "system_fingerprint" in result.provider_metadata

    def test_parse_tool_call_response(self):
        data = _load_response_fixture("openai_generate_tool_call.json")
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        result = client._parse_response(data)

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == {"city": "Paris"}
        assert isinstance(result.tool_calls[0].id, str)
        assert result.tool_calls[1].function.name == "get_time"
        assert result.tool_calls[1].function.arguments == {"city": "Paris"}
        assert result.usage.cached_tokens is not None

    def test_parse_stream_text_chunks(self):
        chunks_data = _load_stream_fixture("openai_stream_text.json")
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")

        content_parts = []
        last_chunk = None
        for chunk_data in chunks_data:
            chunk = client._parse_chunk(chunk_data)
            if chunk is None:
                continue
            last_chunk = chunk
            if chunk.delta_content:
                content_parts.append(chunk.delta_content)

        assert len(content_parts) > 0
        assert last_chunk.usage is not None
        assert last_chunk.usage.total_tokens > 0

    def test_parse_stream_tool_call_chunks(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        results = _replay_stream(client, "openai_stream_tool_calls.json")

        tool_call_chunk = [r for r in results if r.delta_tool_calls]
        assert len(tool_call_chunk) == 1
        tcs = tool_call_chunk[0].delta_tool_calls
        assert len(tcs) == 2
        assert tcs[0].function.name == "get_weather"
        assert tcs[0].function.arguments == {"city": "Paris"}
        assert tcs[1].function.name == "get_time"
        assert tcs[1].function.arguments == {"city": "Paris"}
        assert tool_call_chunk[0].finish_reason == "tool_calls"

        usage_chunk = [r for r in results if r.usage]
        assert len(usage_chunk) == 1
        assert usage_chunk[0].usage.total_tokens > 0

    def test_stream_text_request_id_consistent(self):
        chunks_data = _load_stream_fixture("openai_stream_text.json")
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")

        request_ids = set()
        for chunk_data in chunks_data:
            chunk = client._parse_chunk(chunk_data)
            if chunk and chunk.request_id:
                request_ids.add(chunk.request_id)

        assert len(request_ids) == 1


class TestRecordedNIM:
    """Parse real NIM API responses captured as JSON fixtures.

    NIM responses differ from OpenAI in key ways:
    - prompt_tokens_details is null (not a dict) - crashed us before the `or {}` fix
    - reasoning_content appears alongside tool_calls (mixed reasoning + tool calls)
    - Tool call IDs use chatcmpl-tool-* format (vs OpenAI's call_*)
    - Extra fields: reasoning, stop_reason, token_ids, kv_transfer_params

    These tests prove our parser handles NIM's specific response shape correctly.
    """

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "nim_generate_text.json").exists(),
        reason="NIM fixtures not recorded",
    )
    def test_parse_text_response(self):
        data = _load_response_fixture("nim_generate_text.json")
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        result = client._parse_response(data)

        assert isinstance(result, LLMResponse)
        assert len(result.content) > 0
        assert result.model == "nvidia/nemotron-3-nano-30b-a3b"
        assert result.finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.cached_tokens is None

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "nim_generate_tool_call.json").exists(),
        reason="NIM fixtures not recorded",
    )
    def test_parse_tool_call_response(self):
        data = _load_response_fixture("nim_generate_tool_call.json")
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        result = client._parse_response(data)

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[0].function.arguments == {"city": "Paris"}

    @pytest.mark.skipif(
        not (FIXTURES_DIR / "nim_generate_reasoning.json").exists(),
        reason="NIM fixtures not recorded",
    )
    def test_parse_reasoning_response(self):
        data = _load_response_fixture("nim_generate_reasoning.json")
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        result = client._parse_response(data)

        assert result.content is not None
        assert result.reasoning is not None
        assert len(result.reasoning) > 0
        assert result.finish_reason == "stop"

    @pytest.mark.skipif(not _fixture_exists("nim_stream_tool_calls.json"), reason="NIM fixtures not recorded")
    def test_parse_stream_tool_call_chunks(self):
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        results = _replay_stream(client, "nim_stream_tool_calls.json")

        tool_call_chunk = [r for r in results if r.delta_tool_calls]
        assert len(tool_call_chunk) == 1
        tcs = tool_call_chunk[0].delta_tool_calls
        assert len(tcs) >= 1
        assert tcs[0].function.name == "get_weather"

    @pytest.mark.skipif(not _fixture_exists("nim_stream_reasoning.json"), reason="NIM fixtures not recorded")
    def test_parse_stream_reasoning_chunks(self):
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        results = _replay_stream(client, "nim_stream_reasoning.json")

        reasoning_parts = [r.delta_reasoning for r in results if r.delta_reasoning]
        content_parts = [r.delta_content for r in results if r.delta_content]
        assert len(reasoning_parts) > 0
        assert len(content_parts) > 0
        assert any(r.finish_reason == "stop" for r in results)

    @pytest.mark.skipif(not _fixture_exists("nim_stream_text.json"), reason="NIM fixtures not recorded")
    def test_parse_stream_text_chunks(self):
        chunks_data = _load_stream_fixture("nim_stream_text.json")
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")

        content_parts = []
        for chunk_data in chunks_data:
            chunk = client._parse_chunk(chunk_data)
            if chunk and chunk.delta_content:
                content_parts.append(chunk.delta_content)

        assert len(content_parts) > 0

    @pytest.mark.skipif(not _fixture_exists("nim_generate_tool_call.json"), reason="NIM fixtures not recorded")
    def test_nim_null_fields_dont_crash(self):
        data = _load_response_fixture("nim_generate_tool_call.json")
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        result = client._parse_response(data)

        assert result.usage is not None
        assert result.usage.cached_tokens is None
        assert result.reasoning is not None

    @pytest.mark.skipif(not _fixture_exists("nim_generate_tool_call.json"), reason="NIM fixtures not recorded")
    def test_reasoning_alongside_tool_calls(self):
        data = _load_response_fixture("nim_generate_tool_call.json")
        client = _make_model(model="nvidia/nemotron-3-nano-30b-a3b", base_url="https://integrate.api.nvidia.com/v1")
        result = client._parse_response(data)

        assert result.tool_calls is not None
        assert result.reasoning is not None
        assert len(result.reasoning) > 0
        assert result.tool_calls[0].function.name == "get_weather"


class TestRecordedEdgeCases:
    """Edge cases that real APIs may produce but are hard to trigger on demand.

    These use hand-crafted responses (not recorded fixtures) to test specific
    parsing edge cases: zero-argument tool calls, content alongside tool calls.
    """

    def test_empty_tool_call_arguments(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        body = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_status", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = client._parse_response(_to_http_response(body))

        assert result.tool_calls[0].function.name == "get_status"
        assert result.tool_calls[0].function.arguments == {}

    def test_content_alongside_tool_calls(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        body = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that for you.",
                        "tool_calls": [
                            {
                                "id": "call_456",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = client._parse_response(_to_http_response(body))

        assert result.content == "Let me check that for you."
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.arguments == {"city": "Paris"}


class TestFixtureSanity:
    """Guards to ensure recorded fixtures retain the coverage they were recorded for.

    If someone re-records fixtures and the model's behavior shifts (e.g. single
    tool call instead of parallel), these tests catch the silent coverage loss.
    """

    def test_openai_generate_tool_call_fixture_is_parallel(self):
        data = _load_response_fixture("openai_generate_tool_call.json")
        tool_calls = data.body["choices"][0]["message"].get("tool_calls") or []
        assert len(tool_calls) >= 2, (
            "fixture should contain >=2 parallel tool calls; re-record with a multi-tool prompt"
        )

    def test_openai_stream_tool_calls_fixture_is_parallel(self):
        chunks = _load_stream_fixture("openai_stream_tool_calls.json")
        indexes = set()
        for chunk in chunks:
            choices = chunk.body.get("choices") or []
            if not choices:
                continue
            for tc in choices[0].get("delta", {}).get("tool_calls") or []:
                indexes.add(tc.get("index"))
        assert len(indexes) >= 2, "stream fixture should contain >=2 tool_call indexes"


class TestRecordedErrorBodies:
    """Verify our error parser handles real recorded error bodies from providers."""

    @pytest.mark.skipif(not _fixture_exists("openai_error_401.json"), reason="OpenAI error fixture not recorded")
    def test_openai_401_classifies_as_auth_error(self):
        from nemoguardrails.llm.clients._errors import raise_for_status

        envelope = _load_fixture("openai_error_401.json")
        body = json.dumps(envelope["body"]) if envelope["body"] is not None else ""
        with pytest.raises(LLMAuthenticationError) as exc_info:
            raise_for_status(envelope["status_code"], body, envelope.get("response_headers") or {})
        assert exc_info.value.status_code == 401

    @pytest.mark.skipif(
        not _fixture_exists("openai_error_400_context_length.json"), reason="OpenAI error fixture not recorded"
    )
    def test_openai_400_context_length_classifies(self):
        from nemoguardrails.exceptions import LLMContextWindowError
        from nemoguardrails.llm.clients._errors import raise_for_status

        envelope = _load_fixture("openai_error_400_context_length.json")
        body = json.dumps(envelope["body"]) if envelope["body"] is not None else ""
        with pytest.raises(LLMContextWindowError):
            raise_for_status(envelope["status_code"], body, envelope.get("response_headers") or {})


class TestRecordedFinishLength:
    @pytest.mark.skipif(
        not _fixture_exists("openai_generate_finish_length.json"),
        reason="finish_length fixture not recorded",
    )
    def test_finish_reason_length_parsed(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        data = _load_response_fixture("openai_generate_finish_length.json")
        result = client._parse_response(data)
        assert result.finish_reason == "length"


class TestRecordedRefusal:
    @pytest.mark.skipif(
        not _fixture_exists("openai_generate_refusal.json"),
        reason="refusal fixture not recorded",
    )
    def test_refusal_produces_response(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        data = _load_response_fixture("openai_generate_refusal.json")
        result = client._parse_response(data)
        assert result.finish_reason in ("stop", "content_filter", "other")


class TestRecordedMultimodal:
    @pytest.mark.skipif(
        not _fixture_exists("openai_generate_multimodal.json"),
        reason="multimodal fixture not recorded",
    )
    def test_multimodal_generate_parses(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        data = _load_response_fixture("openai_generate_multimodal.json")
        result = client._parse_response(data)
        assert result.content
        assert result.finish_reason == "stop"

    @pytest.mark.skipif(
        not _fixture_exists("openai_stream_multimodal.json"),
        reason="multimodal stream fixture not recorded",
    )
    def test_multimodal_stream_parses(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        results = _replay_stream(client, "openai_stream_multimodal.json")
        content_parts = [r.delta_content for r in results if r.delta_content]
        assert len(content_parts) > 0


class TestRecordedMultiTurn:
    @pytest.mark.skipif(
        not _fixture_exists("openai_multiturn_tool_roundtrip.json"),
        reason="multi-turn fixture not recorded",
    )
    def test_first_turn_has_tool_call(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        data = _load_fixture("openai_multiturn_tool_roundtrip.json")
        first = client._parse_response(_to_http_response(data["first_response"]))
        assert first.tool_calls is not None
        assert len(first.tool_calls) >= 1
        assert first.finish_reason == "tool_calls"

    @pytest.mark.skipif(
        not _fixture_exists("openai_multiturn_tool_roundtrip.json"),
        reason="multi-turn fixture not recorded",
    )
    def test_second_turn_has_text_content(self):
        client = _make_model(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
        data = _load_fixture("openai_multiturn_tool_roundtrip.json")
        second = client._parse_response(_to_http_response(data["second_response"]))
        assert second.content
        assert second.finish_reason == "stop"


_RED_SQUARE_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAMElEQVR4nO3NMQE"
    "AIAzAsIF/zyCh175EQHve7LrL/TFIBskgGSSDZJAMkkEySAZTPqPEAT9XMVNHAAAAAElFTkSuQmCC"
)


class TestSimulatedOpenAIChat:
    """Parser regression tests for OpenAI chat. Always runs. Assertions are
    strict equality against pinned fixture content - the point of recorded
    fixtures is to catch parser/serialization regressions byte-for-byte.
    """

    @pytest.mark.asyncio
    async def test_generate_text(self):
        async with simulated_model("openai_generate_text.json", base_url=OPENAI_BASE_URL) as model:
            result = await model.generate_async("Say hello in one word")
            assert result.content == "Hello!"
            assert result.finish_reason == "stop"
            assert result.request_id is not None
            assert result.usage is not None
            assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_generate_tool_call(self):
        async with simulated_model("openai_generate_tool_call.json", base_url=OPENAI_BASE_URL) as model:
            result = await model.generate_async("What's the weather in Paris?", tools=TOOLS)
            assert result.finish_reason == "tool_calls"
            assert result.tool_calls is not None
            names = [tc.function.name for tc in result.tool_calls]
            assert "get_weather" in names
            weather_call = next(tc for tc in result.tool_calls if tc.function.name == "get_weather")
            assert weather_call.function.arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_stream_text(self):
        async with simulated_model("openai_stream_text.json", base_url=OPENAI_BASE_URL) as model:
            chunks = []
            async for chunk in model.stream_async("Say hello in one word"):
                chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in chunks)
            assert content == "Hello!"
            assert any(c.finish_reason == "stop" for c in chunks)
            assert any(c.usage for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_tool_calls(self):
        async with simulated_model("openai_stream_tool_calls.json", base_url=OPENAI_BASE_URL) as model:
            chunks = []
            async for chunk in model.stream_async("What's the weather and time in Paris?", tools=TOOLS):
                chunks.append(chunk)
            tool_chunks = [c for c in chunks if c.delta_tool_calls]
            assert len(tool_chunks) >= 1
            assert all(isinstance(tc.function.arguments, dict) for tc in tool_chunks[0].delta_tool_calls)

    @pytest.mark.asyncio
    async def test_generate_multimodal(self):
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[
                    {"type": "text", "text": "Describe the color of this image in one word."},
                    {"type": "image_url", "image_url": {"url": _RED_SQUARE_PNG}},
                ],
            )
        ]
        async with simulated_model("openai_generate_multimodal.json", base_url=OPENAI_BASE_URL) as model:
            result = await model.generate_async(messages)
            assert result.content == "Red."
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_multimodal(self):
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[
                    {"type": "text", "text": "Describe the color in one word."},
                    {"type": "image_url", "image_url": {"url": _RED_SQUARE_PNG}},
                ],
            )
        ]
        async with simulated_model("openai_stream_multimodal.json", base_url=OPENAI_BASE_URL) as model:
            chunks = []
            async for chunk in model.stream_async(messages):
                chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in chunks)
            assert content == "Red."

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens_truncates(self):
        async with simulated_model("openai_generate_finish_length.json", base_url=OPENAI_BASE_URL) as model:
            result = await model.generate_async("Count from 1 to 100, one number per line.", max_tokens=5)
            assert result.finish_reason == "length"
            assert result.usage.output_tokens <= 10

    @pytest.mark.asyncio
    async def test_generate_refusal(self):
        async with simulated_model("openai_generate_refusal.json", base_url=OPENAI_BASE_URL) as model:
            result = await model.generate_async("Something the model refuses")
            assert result.content == "I'm sorry, but I can't assist with that."
            assert result.finish_reason == "stop"


@pytest.mark.skipif(not live_mode_enabled("openai"), reason="LIVE_TEST_MODE=1 and OPENAI_API_KEY required")
class TestLiveOpenAIChat:
    """Provider contract tests against the real OpenAI API. Opt-in
    (LIVE_TEST_MODE=1 + OPENAI_API_KEY). Assertions are shape-only - response
    content varies; we only verify protocol invariants hold against live.
    """

    @pytest.mark.asyncio
    async def test_generate_text(self):
        async with live_openai_model() as model:
            result = await model.generate_async("Say hello in one word")
            assert isinstance(result.content, str)
            assert len(result.content) > 0
            assert result.finish_reason == "stop"
            assert result.request_id is not None
            assert result.usage is not None
            assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_generate_tool_call(self):
        async with live_openai_model() as model:
            result = await model.generate_async("What's the weather in Paris?", tools=TOOLS)
            assert result.finish_reason == "tool_calls"
            assert result.tool_calls is not None
            assert any(tc.function.name == "get_weather" for tc in result.tool_calls)

    @pytest.mark.asyncio
    async def test_stream_text(self):
        async with live_openai_model() as model:
            chunks = []
            async for chunk in model.stream_async("Say hello in one word"):
                chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in chunks)
            assert len(content) > 0
            assert any(c.finish_reason for c in chunks)
            assert any(c.usage for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_tool_calls(self):
        async with live_openai_model() as model:
            chunks = []
            async for chunk in model.stream_async("What's the weather and time in Paris?", tools=TOOLS):
                chunks.append(chunk)
            tool_chunks = [c for c in chunks if c.delta_tool_calls]
            assert len(tool_chunks) >= 1
            assert all(isinstance(tc.function.arguments, dict) for tc in tool_chunks[0].delta_tool_calls)

    @pytest.mark.asyncio
    async def test_generate_multimodal(self):
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[
                    {"type": "text", "text": "Describe the color of this image in one word."},
                    {"type": "image_url", "image_url": {"url": _RED_SQUARE_PNG}},
                ],
            )
        ]
        async with live_openai_model() as model:
            result = await model.generate_async(messages)
            assert isinstance(result.content, str)
            assert len(result.content) > 0
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_multimodal(self):
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[
                    {"type": "text", "text": "Describe the color in one word."},
                    {"type": "image_url", "image_url": {"url": _RED_SQUARE_PNG}},
                ],
            )
        ]
        async with live_openai_model() as model:
            chunks = []
            async for chunk in model.stream_async(messages):
                chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in chunks)
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens_truncates(self):
        async with live_openai_model() as model:
            result = await model.generate_async("Count from 1 to 100, one number per line.", max_tokens=5)
            assert result.finish_reason == "length"
            assert result.usage.output_tokens <= 10

    @pytest.mark.asyncio
    async def test_generate_refusal(self):
        async with live_openai_model() as model:
            result = await model.generate_async("Something the model refuses")
            assert isinstance(result.content, str)
            assert result.finish_reason in ("stop", "content_filter", "other")


class TestSimulatedNIMChat:
    """Parser regression tests for NIM chat. Always runs. Strict-equality
    assertions against pinned fixture content.
    """

    @pytest.mark.asyncio
    async def test_generate_text(self):
        async with simulated_model("nim_generate_text.json", base_url=NIM_BASE_URL) as model:
            result = await model.generate_async(
                "Say hello in one word", chat_template_kwargs={"enable_thinking": False}
            )
            assert result.content == "Hey"
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_tool_call(self):
        async with simulated_model("nim_generate_tool_call.json", base_url=NIM_BASE_URL) as model:
            result = await model.generate_async("What's the weather in Paris?", tools=TOOLS[:1])
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].function.name == "get_weather"
            assert result.tool_calls[0].function.arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_generate_reasoning(self):
        async with simulated_model("nim_generate_reasoning.json", base_url=NIM_BASE_URL) as model:
            result = await model.generate_async("What is 2+2?", chat_template_kwargs={"enable_thinking": True})
            assert result.reasoning is not None
            assert len(result.reasoning) > 0
            assert result.content == "4"

    @pytest.mark.asyncio
    async def test_stream_text(self):
        async with simulated_model("nim_stream_text.json", base_url=NIM_BASE_URL) as model:
            chunks = []
            async for chunk in model.stream_async("Say hello", chat_template_kwargs={"enable_thinking": False}):
                chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in chunks)
            assert content == "Hey"

    @pytest.mark.asyncio
    async def test_stream_tool_calls(self):
        async with simulated_model("nim_stream_tool_calls.json", base_url=NIM_BASE_URL) as model:
            chunks = []
            async for chunk in model.stream_async("weather in Paris", tools=TOOLS[:1]):
                chunks.append(chunk)
            tool_chunks = [c for c in chunks if c.delta_tool_calls]
            assert len(tool_chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_reasoning(self):
        async with simulated_model("nim_stream_reasoning.json", base_url=NIM_BASE_URL) as model:
            chunks = []
            async for chunk in model.stream_async("What is 2+2?", chat_template_kwargs={"enable_thinking": True}):
                chunks.append(chunk)
            reasoning = "".join(c.delta_reasoning or "" for c in chunks)
            assert len(reasoning) > 0


@pytest.mark.skipif(not live_mode_enabled("nim"), reason="LIVE_TEST_MODE=1 and NVIDIA_API_KEY required")
class TestLiveNIMChat:
    """Provider contract tests against the real NIM API. Opt-in. Shape-only
    assertions - content varies between runs.
    """

    @pytest.mark.asyncio
    async def test_generate_text(self):
        async with live_nim_model() as model:
            result = await model.generate_async(
                "Say hello in one word", chat_template_kwargs={"enable_thinking": False}
            )
            assert isinstance(result.content, str)
            assert len(result.content) > 0
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_tool_call(self):
        async with live_nim_model() as model:
            result = await model.generate_async("What's the weather in Paris?", tools=TOOLS[:1])
            assert result.tool_calls is not None
            assert len(result.tool_calls) >= 1

    @pytest.mark.asyncio
    async def test_generate_reasoning(self):
        async with live_nim_model() as model:
            result = await model.generate_async("What is 2+2?", chat_template_kwargs={"enable_thinking": True})
            assert result.reasoning is not None

    @pytest.mark.asyncio
    async def test_stream_text(self):
        async with live_nim_model() as model:
            chunks = []
            async for chunk in model.stream_async("Say hello", chat_template_kwargs={"enable_thinking": False}):
                chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in chunks)
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_stream_tool_calls(self):
        async with live_nim_model() as model:
            chunks = []
            async for chunk in model.stream_async("weather in Paris", tools=TOOLS[:1]):
                chunks.append(chunk)
            tool_chunks = [c for c in chunks if c.delta_tool_calls]
            assert len(tool_chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_reasoning(self):
        async with live_nim_model() as model:
            chunks = []
            async for chunk in model.stream_async("What is 2+2?", chat_template_kwargs={"enable_thinking": True}):
                chunks.append(chunk)
            reasoning = "".join(c.delta_reasoning or "" for c in chunks)
            assert len(reasoning) > 0


class TestSimulatedOpenAIRoundTrip:
    """Parser regression tests for OpenAI multi-turn flows. Always runs."""

    @pytest.mark.skipif(not _fixture_exists("openai_multiturn_tool_roundtrip.json"), reason="fixture not recorded")
    @pytest.mark.asyncio
    async def test_tool_call_two_turn_roundtrip(self):
        multiturn = _load_fixture("openai_multiturn_tool_roundtrip.json")
        fixtures = [multiturn["first_response"], multiturn["second_response"]]
        async with simulated_model_sequenced(fixtures, base_url=OPENAI_BASE_URL) as model:
            user_content = multiturn["user_message"]["content"]
            first = await model.generate_async(user_content, tools=TOOLS[:1])
            assert first.finish_reason == "tool_calls"
            assert first.tool_calls

            followup = [
                ChatMessage(role=Role.USER, content=user_content),
                ChatMessage(role=Role.ASSISTANT, tool_calls=first.tool_calls),
            ]
            tool_result_content = json.dumps({"temperature": 18, "conditions": "cloudy"})
            for tc in first.tool_calls:
                followup.append(ChatMessage(role=Role.TOOL, tool_call_id=tc.id, content=tool_result_content))
            second = await model.generate_async(followup, tools=TOOLS[:1])
            assert second.content
            assert second.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_multi_turn_text_conversation(self):
        async with simulated_model_sequenced(
            ["openai_generate_text.json", "openai_generate_text.json"], base_url=OPENAI_BASE_URL
        ) as model:
            first = await model.generate_async("Say hello")
            assert first.content == "Hello!"

            followup = [
                ChatMessage(role=Role.USER, content="Say hello"),
                ChatMessage(role=Role.ASSISTANT, content=first.content),
                ChatMessage(role=Role.USER, content="Say it again"),
            ]
            second = await model.generate_async(followup)
            assert second.content == "Hello!"
            assert second.finish_reason == "stop"


@pytest.mark.skipif(not live_mode_enabled("openai"), reason="LIVE_TEST_MODE=1 and OPENAI_API_KEY required")
class TestLiveOpenAIRoundTrip:
    """Provider contract tests for OpenAI multi-turn. Opt-in."""

    @pytest.mark.asyncio
    async def test_tool_call_two_turn_roundtrip(self):
        async with live_openai_model() as model:
            first = await model.generate_async("What's the weather in Paris?", tools=TOOLS[:1])
            assert first.finish_reason == "tool_calls"
            assert first.tool_calls

            followup = [
                ChatMessage(role=Role.USER, content="What's the weather in Paris?"),
                ChatMessage(role=Role.ASSISTANT, tool_calls=first.tool_calls),
            ]
            tool_result_content = json.dumps({"temperature": 18, "conditions": "cloudy"})
            for tc in first.tool_calls:
                followup.append(ChatMessage(role=Role.TOOL, tool_call_id=tc.id, content=tool_result_content))
            second = await model.generate_async(followup, tools=TOOLS[:1])
            assert second.content
            assert second.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_multi_turn_text_conversation(self):
        async with live_openai_model() as model:
            first = await model.generate_async("Say hello")
            assert first.content

            followup = [
                ChatMessage(role=Role.USER, content="Say hello"),
                ChatMessage(role=Role.ASSISTANT, content=first.content),
                ChatMessage(role=Role.USER, content="Say it again"),
            ]
            second = await model.generate_async(followup)
            assert second.content
            assert second.finish_reason == "stop"


class TestSimulatedNIMRoundTrip:
    """Parser regression for NIM multi-turn tool-call. NIM's wire format diverges
    from OpenAI (null prompt_tokens_details, different tool-calling quirks),
    so OpenAI coverage doesn't transfer. Skipped until the fixture is recorded.
    """

    @pytest.mark.skipif(
        not _fixture_exists("nim_multiturn_tool_roundtrip.json"),
        reason="fixture not recorded - run record_fixtures.py with NVIDIA_API_KEY set",
    )
    @pytest.mark.asyncio
    async def test_tool_call_two_turn_roundtrip(self):
        multiturn = _load_fixture("nim_multiturn_tool_roundtrip.json")
        fixtures = [multiturn["first_response"], multiturn["second_response"]]
        async with simulated_model_sequenced(fixtures, base_url=NIM_BASE_URL) as model:
            user_content = multiturn["user_message"]["content"]
            first = await model.generate_async(user_content, tools=TOOLS[:1])
            assert first.finish_reason == "tool_calls"
            assert first.tool_calls

            followup = [
                ChatMessage(role=Role.USER, content=user_content),
                ChatMessage(role=Role.ASSISTANT, tool_calls=first.tool_calls),
            ]
            tool_result_content = json.dumps({"temperature": 18, "conditions": "cloudy"})
            for tc in first.tool_calls:
                followup.append(ChatMessage(role=Role.TOOL, tool_call_id=tc.id, content=tool_result_content))
            second = await model.generate_async(followup, tools=TOOLS[:1])
            assert second.content
            assert second.finish_reason == "stop"


@pytest.mark.skipif(not live_mode_enabled("nim"), reason="LIVE_TEST_MODE=1 and NVIDIA_API_KEY required")
class TestLiveNIMRoundTrip:
    """Provider contract test for NIM multi-turn tool-call. Opt-in."""

    @pytest.mark.asyncio
    async def test_tool_call_two_turn_roundtrip(self):
        async with live_nim_model() as model:
            first = await model.generate_async("What's the weather in Paris?", tools=TOOLS[:1])
            assert first.finish_reason == "tool_calls"
            assert first.tool_calls

            followup = [
                ChatMessage(role=Role.USER, content="What's the weather in Paris?"),
                ChatMessage(role=Role.ASSISTANT, tool_calls=first.tool_calls),
            ]
            tool_result_content = json.dumps({"temperature": 18, "conditions": "cloudy"})
            for tc in first.tool_calls:
                followup.append(ChatMessage(role=Role.TOOL, tool_call_id=tc.id, content=tool_result_content))
            second = await model.generate_async(followup, tools=TOOLS[:1])
            assert second.content
            assert second.finish_reason == "stop"


class TestOpenAIErrorHandling:
    """Error classification via recorded error envelopes. Fixture-only -
    live error tests require engineered failures (bad keys, huge prompts,
    invalid schemas) which are flaky and expensive to run online.
    """

    @pytest.mark.skipif(not _fixture_exists("openai_error_401.json"), reason="fixture not recorded")
    @pytest.mark.asyncio
    async def test_error_401_bad_key_redacts(self):
        async with simulated_model("openai_error_401.json", api_key="sk-proj-realkey123456789") as model:
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await model.generate_async("test")
        assert exc_info.value.status_code == 401
        assert "sk-proj-realkey123456789" not in exc_info.value.error_message
        assert "sk-inval" not in exc_info.value.error_message
        assert "sk-***" in exc_info.value.error_message

    @pytest.mark.skipif(not _fixture_exists("openai_error_400_context_length.json"), reason="fixture not recorded")
    @pytest.mark.asyncio
    async def test_error_400_context_length(self):
        async with simulated_model("openai_error_400_context_length.json") as model:
            with pytest.raises(LLMContextWindowError):
                await model.generate_async("test")


class TestOpenAIPayloadSerialization:
    """Assert on outgoing request payloads. Fixture-only - the check is on
    what we SEND, not what we receive, so LIVE mode would cost an API call
    per test without extra coverage.
    """

    @pytest.mark.asyncio
    async def test_stream_options_include_usage_is_sent(self):
        captured = []
        async with simulated_model("openai_stream_text.json", on_request=captured.append) as model:
            async for _ in model.stream_async("hi"):
                pass
        payload = json.loads(captured[0].content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_non_stream_does_not_include_stream_keys(self):
        captured = []
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async("hi")
        payload = json.loads(captured[0].content)
        assert "stream" not in payload
        assert "stream_options" not in payload

    @pytest.mark.asyncio
    async def test_tools_are_serialized_in_payload(self):
        captured = []
        async with simulated_model("openai_generate_tool_call.json", on_request=captured.append) as model:
            await model.generate_async("weather", tools=TOOLS)
        payload = json.loads(captured[0].content)
        assert payload["tools"] == TOOLS

    @pytest.mark.asyncio
    async def test_temperature_is_serialized(self):
        captured = []
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async("hi", temperature=0.3)
        payload = json.loads(captured[0].content)
        assert payload["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_stop_sequence_is_serialized(self):
        captured = []
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async("hi", stop=["END"])
        payload = json.loads(captured[0].content)
        assert payload["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_max_tokens_is_serialized(self):
        captured = []
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async("hi", max_tokens=50)
        payload = json.loads(captured[0].content)
        assert payload["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_authorization_header_sent(self):
        captured = []
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async("hi")
        assert captured[0].headers["authorization"] == "Bearer sk-test-simulated"

    @pytest.mark.asyncio
    async def test_content_type_header_sent(self):
        captured = []
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async("hi")
        assert "application/json" in captured[0].headers["content-type"]

    @pytest.mark.asyncio
    async def test_multimodal_content_blocks_serialized(self):
        captured = []
        messages = [
            ChatMessage(
                role=Role.USER,
                content=[
                    {"type": "text", "text": "what color?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            )
        ]
        async with simulated_model("openai_generate_text.json", on_request=captured.append) as model:
            await model.generate_async(messages)
        payload = json.loads(captured[0].content)
        user_msg = payload["messages"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "what color?"}
        assert user_msg["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_reasoning_model_strips_stop_and_temperature(self):
        captured = []
        async with simulated_model(
            "openai_generate_text.json", on_request=captured.append, model_name="o3-mini"
        ) as model:
            await model.generate_async("hi", stop=["END"], temperature=0.5)
        payload = json.loads(captured[0].content)
        assert "stop" not in payload
        assert "temperature" not in payload

    @pytest.mark.asyncio
    async def test_model_name_in_payload(self):
        captured = []
        async with simulated_model(
            "openai_generate_text.json", on_request=captured.append, model_name="gpt-4o"
        ) as model:
            await model.generate_async("hi")
        payload = json.loads(captured[0].content)
        assert payload["model"] == "gpt-4o"


@pytest.mark.skipif(
    not LIVE_TEST_MODE or not os.getenv("OPENAI_API_KEY"),
    reason="requires LIVE_TEST_MODE=1 and OPENAI_API_KEY",
)
class TestOpenAIProviderContract:
    """Thin live-only layer that catches provider-side format drift.

    Fixtures used by the unified classes are frozen snapshots - they won't
    flag a response-shape change until they're re-recorded. This class
    runs against the real OpenAI API and verifies the fields and formats
    our parser depends on still hold. Run periodically (e.g., nightly
    LIVE CI) to detect drift early.
    """

    @pytest.mark.asyncio
    async def test_response_shape_is_still_compatible(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) as client:
            response = await client.chat_completion(
                "gpt-4o-mini",
                [{"role": "user", "content": "Say hello in one word"}],
            )
        body = response.body
        assert "id" in body
        assert "model" in body
        choices = body.get("choices")
        assert isinstance(choices, list) and len(choices) > 0
        message = choices[0].get("message")
        assert isinstance(message, dict) and "content" in message
        assert "finish_reason" in choices[0]
        usage = body.get("usage")
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage and "completion_tokens" in usage

    @pytest.mark.asyncio
    async def test_401_body_is_still_compatible(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-invalid-provider-contract-test",
            max_retries=0,
        ) as client:
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await client.chat_completion("gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 401
        assert exc_info.value.error_message
        assert "sk-invalid-provider-contract-test" not in exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_wrong_model_error_is_still_compatible(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_retries=0,
        ) as client:
            with pytest.raises(LLMClientError) as exc_info:
                await client.chat_completion(
                    "nonexistent-model-xyz-123",
                    [{"role": "user", "content": "hi"}],
                )
        assert exc_info.value.status_code in (400, 404)
        assert exc_info.value.error_message

    @pytest.mark.asyncio
    async def test_sse_framing_is_still_compatible(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) as client:
            chunks = []
            async for chunk in client.stream_chat_completion(
                "gpt-4o-mini",
                [{"role": "user", "content": "Say hello"}],
            ):
                chunks.append(chunk)
        assert len(chunks) > 0
        assert any(chunk.body.get("choices") for chunk in chunks)
        assert any(chunk.body.get("usage") for chunk in chunks)

    @pytest.mark.asyncio
    async def test_401_body_does_not_echo_bad_key(self):
        bad_key = "sk-proj-live-contract-should-not-echo-this-12345"
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=bad_key,
            max_retries=0,
        ) as client:
            with pytest.raises(LLMAuthenticationError) as exc_info:
                await client.chat_completion("gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert bad_key not in exc_info.value.error_message
        assert bad_key not in str(exc_info.value.body or "")

    @pytest.mark.asyncio
    async def test_bad_temperature_still_classifies_as_bad_request(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_retries=0,
        ) as client:
            with pytest.raises(LLMClientError) as exc_info:
                await client.chat_completion(
                    "gpt-4o-mini",
                    [{"role": "user", "content": "hi"}],
                    temperature=99,
                )
        assert exc_info.value.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_invalid_tool_schema_still_classifies_as_bad_request(self):
        bad_tools = [{"type": "function", "function": {"name": "x", "parameters": "not_an_object"}}]
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_retries=0,
        ) as client:
            with pytest.raises(LLMClientError) as exc_info:
                await client.chat_completion(
                    "gpt-4o-mini",
                    [{"role": "user", "content": "hi"}],
                    tools=bad_tools,
                )
        assert exc_info.value.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_streaming_tool_call_round_trip_is_still_compatible(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) as client:
            model = OpenAIChatModel(client=client, model="gpt-4o-mini")

            first_chunks = []
            async for chunk in model.stream_async("What's the weather in Paris?", tools=TOOLS):
                first_chunks.append(chunk)

            tool_call_chunks = [c for c in first_chunks if c.delta_tool_calls]
            assert tool_call_chunks, "first streaming turn should yield tool_calls"
            tool_calls = tool_call_chunks[0].delta_tool_calls

            followup = [
                ChatMessage(role=Role.USER, content="What's the weather in Paris?"),
                ChatMessage(role=Role.ASSISTANT, tool_calls=tool_calls),
            ]
            tool_result = json.dumps({"temperature": 18, "conditions": "cloudy"})
            for tc in tool_calls:
                followup.append(ChatMessage(role=Role.TOOL, tool_call_id=tc.id, content=tool_result))

            second_chunks = []
            async for chunk in model.stream_async(followup, tools=TOOLS):
                second_chunks.append(chunk)
            content = "".join(c.delta_content or "" for c in second_chunks)
            assert len(content) > 0
