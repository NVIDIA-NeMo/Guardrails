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
from pathlib import Path
from typing import Any

from nemoguardrails.rails.llm.options import GenerationResponse, RailsResult, RailStatus
from tests.recorded.cassette import RecordedChatResponse, cassette_request_jsons


def assert_rails_result(
    result: RailsResult,
    *,
    status: RailStatus,
    rail: str | None = None,
    content: str | None = None,
) -> None:
    assert result.status == status
    assert isinstance(result.content, str)
    if rail is not None:
        assert result.rail == rail
    if content is not None:
        assert result.content == content


def assert_generation_response(result: Any) -> GenerationResponse:
    assert isinstance(result, GenerationResponse)
    assert result.response
    if isinstance(result.response, list):
        assert result.response[-1]["role"] == "assistant"
        assert result.response[-1]["content"].strip()
    else:
        assert result.response.strip()
    return result


def assert_blocked_generation(result: Any, *, refusal: str | None = None) -> GenerationResponse:
    """Assert a generate-surface result was blocked by a rail.

    Unlike ``assert_generation_response`` (which only checks the assistant message is
    non-empty), this asserts the block semantics: a rail decided to ``stop``, and the
    assistant message carries the refusal text (matched exactly when ``refusal`` is given).
    """
    result = assert_generation_response(result)
    assert result.log is not None
    assert any(rail.stop for rail in result.log.activated_rails)
    message = result.response[-1]["content"] if isinstance(result.response, list) else result.response
    if refusal is not None:
        assert message == refusal
    return result


def assert_activated_rails(result: GenerationResponse, expected: set[str]) -> None:
    assert result.log is not None
    activated = {rail.name for rail in result.log.activated_rails}
    assert expected <= activated


def assert_llm_tasks(result: GenerationResponse, expected: set[str]) -> None:
    assert result.log is not None
    assert result.log.llm_calls is not None
    tasks = {call.task for call in result.log.llm_calls}
    assert expected <= tasks


def assert_generated_text(result: Any) -> None:
    assert isinstance(result, str)
    assert result.strip()


def assert_generated_message(result: Any) -> None:
    assert isinstance(result, dict)
    assert result.get("role") == "assistant"
    content = result.get("content")
    assert isinstance(content, str)
    assert content.strip()


def assert_stream_contract(chunks: list[Any], *, expect_multiple: bool = True) -> str:
    assert chunks
    if expect_multiple:
        assert len(chunks) > 1

    content_parts = []
    for chunk in chunks:
        assert isinstance(chunk, (str, dict))
        if isinstance(chunk, str):
            content_parts.append(chunk)
        elif "text" in chunk and isinstance(chunk["text"], str):
            content_parts.append(chunk["text"])
        elif "content" in chunk and isinstance(chunk["content"], str):
            content_parts.append(chunk["content"])

    content = "".join(content_parts)
    assert content.strip()
    return content


def assert_no_stream_error(chunks: list[Any]) -> None:
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith('{"error":'):
            raise AssertionError(chunk)


def assert_blocked_stream_error(chunks: list[Any]) -> None:
    errors = [json.loads(chunk) for chunk in chunks if isinstance(chunk, str) and chunk.startswith('{"error":')]
    assert errors
    assert errors[-1]["error"]["type"] == "guardrails_violation"
    assert errors[-1]["error"]["code"] == "content_blocked"


def assert_llm_call_usage(llm_call: Any, expected: RecordedChatResponse) -> None:
    assert expected.usage
    assert llm_call.prompt_tokens == expected.usage["input_tokens"]
    assert llm_call.completion_tokens == expected.usage["output_tokens"]
    assert llm_call.total_tokens == expected.usage["total_tokens"]


def assert_runtime_model_matches(llm_call: Any, *, configured_model: str, recorded_model: str | None) -> None:
    assert llm_call.llm_model_name
    assert llm_call.llm_model_name in {configured_model, recorded_model}
    assert llm_call.task


def assert_request_payload(
    cassette_path: Path,
    *,
    model: str,
    stream: bool | None = None,
    expected_params: dict[str, Any] | None = None,
    absent_params: set[str] | None = None,
) -> None:
    """Assert that the last recorded request for ``model`` in the cassette matches.

    Optionally checks the ``stream`` flag, that specific ``expected_params`` were
    sent, and that none of ``absent_params`` appear in the payload.
    """
    payloads = cassette_request_jsons(cassette_path)
    matches = [payload for payload in payloads if payload.get("model") == model]
    assert matches, f"{cassette_path} does not contain a request for {model}"
    payload = matches[-1]
    if stream is not None:
        assert payload.get("stream") is stream
    if expected_params:
        for key, value in expected_params.items():
            assert payload.get(key) == value
    if absent_params:
        for key in absent_params:
            assert key not in payload
