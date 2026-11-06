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

from __future__ import annotations

import json
from typing import cast

import pytest
import yaml

from nemoguardrails import LLMRails
from nemoguardrails.types import ToolCall, ToolCallChunk, ToolCallDeltaChunk, ToolCallFunction
from tests.recorded.assertions import assert_llm_call_usage, assert_no_stream_error, assert_request_payload
from tests.recorded.cassette import recorded_chat_response, stream_payloads_from_body
from tests.recorded.normalization import normalize_stream_chunks
from tests.recorded.rails.public_api.configs import (
    NIM_TOOL_CALL_CONFIG,
    NIM_TOOL_CALL_MODEL,
    OPENAI_BASELINE_CONFIG,
    OPENAI_INVALID_MODEL_CONFIG,
    OPENAI_MODEL,
    WEATHER_TOOL,
    WEATHER_TOOL_CHOICE,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def single_upstream_exchange(vcr):
    yield
    assert len(vcr.requests) == 1
    if vcr.record_mode == "none":
        assert vcr.play_count == 1
        assert vcr.all_played


def _assemble_calls(events):
    parts = {}
    for event in events:
        for delta in event.tool_calls:
            entry = parts.setdefault(delta.index, {"id": "", "name": "", "arguments": "", "type": "function"})
            entry["id"] += delta.id or ""
            entry["type"] = delta.type or entry["type"]
            if delta.function is not None:
                entry["name"] += delta.function.name or ""
                entry["arguments"] += delta.function.arguments or ""
    return [
        ToolCall(
            id=part["id"],
            type=part["type"],
            function=ToolCallFunction(name=part["name"], arguments=json.loads(part["arguments"])),
        )
        for _, part in sorted(parts.items())
    ]


def _assert_recorded_calls(cassette_path, calls, events):
    interactions = yaml.safe_load(cassette_path.read_text())["interactions"]
    assert len(interactions) == 1
    payloads = stream_payloads_from_body(interactions[0]["response"]["body"])
    expected_deltas = [
        choice["delta"]["tool_calls"]
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice.get("delta", {}).get("tool_calls")
    ]
    assert [[delta.to_dict() for delta in event.tool_calls] for event in events if event.tool_calls] == expected_deltas
    fragments = {}
    for payload in payloads:
        for choice in payload.get("choices", []):
            for delta in choice.get("delta", {}).get("tool_calls", []):
                fragments.setdefault(delta["index"], []).append(delta)
    assert len(fragments) == len(calls)
    expected = []
    for index in sorted(fragments):
        parts = fragments[index]
        expected.append(
            ToolCall(
                id="".join(part.get("id") or "" for part in parts),
                type=next(part["type"] for part in parts if part.get("type")),
                function=ToolCallFunction(
                    name="".join(part.get("function", {}).get("name") or "" for part in parts),
                    arguments=json.loads("".join(part.get("function", {}).get("arguments") or "" for part in parts)),
                ),
            )
        )
    assert calls == expected


async def _run_tool_stream(
    source, model, *, include_metadata: bool, record_mode, cassette_path, parallel=False, api_key=None
):
    config = load_config(source)
    if api_key is not None:
        config.models[0].parameters["api_key"] = api_key
    rails = LLMRails(config)
    params = {
        "tools": [WEATHER_TOOL],
        "tool_choice": WEATHER_TOOL_CHOICE,
        "stream_options": {"include_usage": True},
        "max_tokens": 1024,
    }
    if source == NIM_TOOL_CALL_CONFIG:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        params["stop"] = ["Paris"]
    if parallel:
        params["parallel_tool_calls"] = True
        params["tool_choice"] = "required"
    prompt = (
        "Call get_weather once for Paris and once for Boston. Do not answer in prose."
        if parallel
        else "Call get_weather for Paris. Do not answer in prose."
    )
    chunks: list[str | dict | ToolCallChunk | ToolCallDeltaChunk] = [
        chunk
        async for chunk in rails.stream_async(
            messages=[{"role": "user", "content": prompt}],
            options={"llm_params": params},
            include_metadata=include_metadata,
        )
    ]
    assert_no_stream_error(chunks)
    assert not any(isinstance(chunk, ToolCallChunk) for chunk in chunks)
    events = [chunk for chunk in chunks if isinstance(chunk, ToolCallDeltaChunk)]
    assert len(events) > 2
    assert events[-1].tool_calls == []
    assert events[-1].finish_reason == ("stop" if source == NIM_TOOL_CALL_CONFIG else "tool_calls")
    assert all(event.finish_reason is None for event in events[:-1])
    calls = _assemble_calls(events)
    assert len(calls) == (2 if parallel else 1)
    assert all(isinstance(call, ToolCall) and isinstance(call.function, ToolCallFunction) for call in calls)
    assert all(call.id and call.type == "function" and call.function.name == "get_weather" for call in calls)
    assert len({call.id for call in calls}) == len(calls)
    assert sorted(call.function.arguments["city"] for call in calls) == (["Boston", "Paris"] if parallel else ["Paris"])
    text_chunks = [chunk for chunk in chunks if not isinstance(chunk, (ToolCallChunk, ToolCallDeltaChunk))]
    assert all(isinstance(chunk, dict if include_metadata else str) for chunk in text_chunks)
    if include_metadata:
        assert isinstance(chunks[-1], dict)
        final_chunk = cast(dict, chunks[-1])
        assert final_chunk["text"] == ""
        assert "response_metadata" in final_chunk["metadata"]
        assert "usage_metadata" in final_chunk["metadata"]
    normalized = normalize_stream_chunks(chunks)
    llm_calls = rails.explain().llm_calls
    assert len(llm_calls) == 1
    assert llm_calls[0].completion == normalized["content"]
    assert isinstance(llm_calls[0].completion, str)
    assert "tool_calls" not in llm_calls[0].completion
    if record_mode == "none":
        _assert_recorded_calls(cassette_path, calls, events)
        assert_request_payload(
            cassette_path,
            model=model,
            stream=True,
            expected_params={
                "tools": [WEATHER_TOOL],
                "tool_choice": params["tool_choice"],
                "stream_options": {"include_usage": True},
            },
        )
        expected = recorded_chat_response(cassette_path, request_model=model, stream=True)
        assert expected.finish_reason == ("stop" if source == NIM_TOOL_CALL_CONFIG else "tool_calls")
        assert expected.content == normalized["content"]
        assert_llm_call_usage(llm_calls[0], expected)
        if include_metadata:
            usage = [
                chunk["metadata"]["usage"]
                for chunk in text_chunks
                if isinstance(chunk, dict) and chunk.get("metadata", {}).get("usage")
            ]
            assert usage == [{key: expected.usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")}]
    return normalized


async def test_openai_streamed_tool_calls(openai_api_key, record_mode, recorded_cassette_path):
    assert await _run_tool_stream(
        OPENAI_BASELINE_CONFIG,
        OPENAI_MODEL,
        include_metadata=False,
        record_mode=record_mode,
        cassette_path=recorded_cassette_path,
    ) == snapshot(
        {
            "content": "",
            "chunks": [
                {
                    "tool_call_deltas": [
                        {
                            "index": 0,
                            "id": "[TOOL_CALL_0]",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                    "finish_reason": None,
                },
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '{"'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": "city"}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '":"'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": "Paris"}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '"}'}}], "finish_reason": None},
                {"tool_call_deltas": [], "finish_reason": "tool_calls"},
            ],
            "errors": [],
        }
    )


async def test_openai_streamed_tool_calls_with_metadata(openai_api_key, record_mode, recorded_cassette_path):
    assert await _run_tool_stream(
        OPENAI_BASELINE_CONFIG,
        OPENAI_MODEL,
        include_metadata=True,
        record_mode=record_mode,
        cassette_path=recorded_cassette_path,
    ) == snapshot(
        {
            "content": "",
            "chunks": [
                {
                    "tool_call_deltas": [
                        {
                            "index": 0,
                            "id": "[TOOL_CALL_0]",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                    "finish_reason": None,
                },
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '{"'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": "city"}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '":"'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": "Paris"}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '"}'}}], "finish_reason": None},
                {"tool_call_deltas": [], "finish_reason": "tool_calls"},
                {"text": "", "usage": {"input_tokens": 138, "output_tokens": 17, "total_tokens": 155}},
            ],
            "errors": [],
        }
    )


async def test_nim_gateway_streamed_tool_calls(nvidia_inference_key, record_mode, recorded_cassette_path):
    assert await _run_tool_stream(
        NIM_TOOL_CALL_CONFIG,
        NIM_TOOL_CALL_MODEL,
        include_metadata=False,
        record_mode=record_mode,
        cassette_path=recorded_cassette_path,
        api_key=nvidia_inference_key,
    ) == snapshot(
        {
            "content": "",
            "chunks": [
                "",
                "",
                {
                    "tool_call_deltas": [
                        {
                            "index": 0,
                            "id": "[TOOL_CALL_0]",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                    "finish_reason": None,
                },
                "",
                {
                    "tool_call_deltas": [{"index": 0, "type": "function", "function": {"arguments": '{"city": "'}}],
                    "finish_reason": None,
                },
                "",
                {
                    "tool_call_deltas": [{"index": 0, "type": "function", "function": {"arguments": "Paris"}}],
                    "finish_reason": None,
                },
                "",
                {
                    "tool_call_deltas": [{"index": 0, "type": "function", "function": {"arguments": '"}'}}],
                    "finish_reason": None,
                },
                "",
                {"tool_call_deltas": [], "finish_reason": "stop"},
                "",
            ],
            "errors": [],
        }
    )


async def test_nim_gateway_streamed_tool_calls_with_metadata(nvidia_inference_key, record_mode, recorded_cassette_path):
    assert await _run_tool_stream(
        NIM_TOOL_CALL_CONFIG,
        NIM_TOOL_CALL_MODEL,
        include_metadata=True,
        record_mode=record_mode,
        cassette_path=recorded_cassette_path,
        api_key=nvidia_inference_key,
    ) == snapshot(
        {
            "content": "",
            "chunks": [
                {"text": ""},
                {"text": ""},
                {
                    "tool_call_deltas": [
                        {
                            "index": 0,
                            "id": "[TOOL_CALL_0]",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                    "finish_reason": None,
                },
                {"text": ""},
                {
                    "tool_call_deltas": [{"index": 0, "type": "function", "function": {"arguments": '{"city": "'}}],
                    "finish_reason": None,
                },
                {"text": ""},
                {
                    "tool_call_deltas": [{"index": 0, "type": "function", "function": {"arguments": "Paris"}}],
                    "finish_reason": None,
                },
                {"text": ""},
                {
                    "tool_call_deltas": [{"index": 0, "type": "function", "function": {"arguments": '"}'}}],
                    "finish_reason": None,
                },
                {"text": ""},
                {"tool_call_deltas": [], "finish_reason": "stop"},
                {"text": "", "usage": {"input_tokens": 291, "output_tokens": 23, "total_tokens": 314}},
                {"text": ""},
            ],
            "errors": [],
        }
    )


async def test_openai_parallel_streamed_tool_calls(openai_api_key, record_mode, recorded_cassette_path):
    assert await _run_tool_stream(
        OPENAI_BASELINE_CONFIG,
        OPENAI_MODEL,
        include_metadata=True,
        parallel=True,
        record_mode=record_mode,
        cassette_path=recorded_cassette_path,
    ) == snapshot(
        {
            "content": "",
            "chunks": [
                {
                    "tool_call_deltas": [
                        {
                            "index": 0,
                            "id": "[TOOL_CALL_0]",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                    "finish_reason": None,
                },
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '{"ci'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": 'ty": '}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '"Paris'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 0, "function": {"arguments": '"}'}}], "finish_reason": None},
                {
                    "tool_call_deltas": [
                        {
                            "index": 1,
                            "id": "[TOOL_CALL_1]",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ],
                    "finish_reason": None,
                },
                {"tool_call_deltas": [{"index": 1, "function": {"arguments": '{"ci'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 1, "function": {"arguments": 'ty": '}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 1, "function": {"arguments": '"Bosto'}}], "finish_reason": None},
                {"tool_call_deltas": [{"index": 1, "function": {"arguments": 'n"}'}}], "finish_reason": None},
                {"tool_call_deltas": [], "finish_reason": "tool_calls"},
                {"text": "", "usage": {"input_tokens": 143, "output_tokens": 47, "total_tokens": 190}},
            ],
            "errors": [],
        }
    )


async def test_tool_stream_invalid_model_fails_closed(openai_api_key):
    rails = LLMRails(load_config(OPENAI_INVALID_MODEL_CONFIG))
    chunks = [
        chunk
        async for chunk in rails.stream_async(
            messages=[{"role": "user", "content": "Call get_weather for Paris."}],
            options={"llm_params": {"tools": [WEATHER_TOOL], "tool_choice": WEATHER_TOOL_CHOICE}},
        )
    ]
    assert not any(isinstance(chunk, (ToolCallChunk, ToolCallDeltaChunk)) for chunk in chunks)
    errors = normalize_stream_chunks(chunks)["errors"]
    assert len(errors) == 1
    assert errors[0]["error"]["type"] == "downstream_error"
    assert errors[0]["error"]["code"] == 404
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "",
            "chunks": [
                '{"error": {"message": "The model `gpt-nonexistent-test-model` does not exist or you do not have access to it.", "type": "downstream_error", "param": null, "code": 404}}'
            ],
            "errors": [
                {
                    "error": {
                        "message": "The model `gpt-nonexistent-test-model` does not exist or you do not have access to it.",
                        "type": "downstream_error",
                        "param": None,
                        "code": 404,
                    }
                }
            ],
        }
    )
