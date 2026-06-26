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

from collections.abc import AsyncIterator

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.exceptions import StreamingNotSupportedError
from tests.recorded.assertions import (
    assert_blocked_stream_error,
    assert_llm_call_usage,
    assert_no_stream_error,
    assert_runtime_model_matches,
    assert_stream_contract,
)
from tests.recorded.cassette import recorded_chat_response
from tests.recorded.normalization import normalize_stream_chunks
from tests.recorded.rails.public_api.configs import (
    NIM_BASELINE_CONFIG,
    OPENAI_BASELINE_CONFIG,
    OPENAI_MODEL,
    STREAMING_DISABLED_CONFIG,
    STREAMING_OUTPUT_RAILS_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


async def _chunks(values: list[str]) -> AsyncIterator[str]:
    for value in values:
        yield value


@pytest.mark.vcr
async def test_openai_stream_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(prompt="Say hello in a few words."):
        chunks.append(chunk)

    assert_stream_contract(chunks, expect_multiple=False)
    assert_no_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {"content": "Hello there! 👋", "chunks": ["", "Hello", " there", "!", " 👋", "", ""], "errors": []}
    )


@pytest.mark.vcr
async def test_nim_stream_async_public_contract(nvidia_api_key):
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(messages=[{"role": "user", "content": "Say hello in a few words."}]):
        chunks.append(chunk)

    assert_stream_contract(chunks, expect_multiple=False)
    assert_no_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "Hello! 😊",
            "chunks": [
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Hello! 😊",
                "",
            ],
            "errors": [],
        }
    )


@pytest.mark.vcr
async def test_stream_async_matches_recorded_chat_completion_metadata(
    openai_api_key, record_mode, recorded_cassette_path
):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(prompt="Say hello in a few words.", include_metadata=True):
        chunks.append(chunk)

    assert chunks
    assert all(isinstance(chunk, dict) for chunk in chunks)
    content = "".join(chunk["text"] for chunk in chunks if isinstance(chunk.get("text"), str))
    assert content.strip()

    if record_mode == "none":
        expected = recorded_chat_response(
            recorded_cassette_path,
            request_model=OPENAI_MODEL,
            stream=True,
        )
        assert expected.raw_usage is not None
        assert expected.finish_reason == "stop"
        assert expected.request_id
        assert content == expected.content

        usage_chunks = [chunk for chunk in chunks if chunk.get("metadata", {}).get("usage")]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["metadata"]["usage"] == {
            "input_tokens": expected.usage["input_tokens"],
            "output_tokens": expected.usage["output_tokens"],
            "total_tokens": expected.usage["total_tokens"],
        }

        llm_calls = rails.explain().llm_calls
        assert len(llm_calls) == 1
        llm_call = llm_calls[0]
        assert llm_call.completion == expected.content
        assert llm_call.llm_provider_name == "openai"
        assert_llm_call_usage(llm_call, expected)
        assert_runtime_model_matches(llm_call, configured_model=OPENAI_MODEL, recorded_model=expected.model)

    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "Hello there! 👋",
            "chunks": [
                {"text": ""},
                {"text": "Hello"},
                {"text": " there"},
                {"text": "!"},
                {"text": " 👋"},
                {"text": ""},
                {"text": "", "usage": {"input_tokens": 13, "output_tokens": 8, "total_tokens": 21}},
                {"text": ""},
            ],
            "errors": [],
        }
    )


async def test_streaming_output_rails_allowed():
    rails = LLMRails(load_config(STREAMING_OUTPUT_RAILS_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "stream"}],
        generator=_chunks(["Hello", " ", "there"]),
    ):
        chunks.append(chunk)

    assert_stream_contract(chunks)
    assert_no_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {"content": "Hello there", "chunks": ["Hello", " ", "there"], "errors": []}
    )


async def test_streaming_output_rails_blocked():
    rails = LLMRails(load_config(STREAMING_OUTPUT_RAILS_CONFIG), verbose=False)

    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "stream"}],
        generator=_chunks(["BLOCK"]),
    ):
        chunks.append(chunk)

    assert_blocked_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "",
            "chunks": [
                '{"error": {"message": "Blocked by streaming output rail rails.", "type": "guardrails_violation", "param": "streaming output rail", "code": "content_blocked"}}'
            ],
            "errors": [
                {
                    "error": {
                        "message": "Blocked by streaming output rail rails.",
                        "type": "guardrails_violation",
                        "param": "streaming output rail",
                        "code": "content_blocked",
                    }
                }
            ],
        }
    )


async def test_streaming_output_rails_disabled_validation():
    rails = LLMRails(load_config(STREAMING_DISABLED_CONFIG), verbose=False)

    with pytest.raises(StreamingNotSupportedError):
        async for _ in rails.stream_async(
            messages=[{"role": "user", "content": "stream"}],
            generator=_chunks(["Hello"]),
        ):
            pass
