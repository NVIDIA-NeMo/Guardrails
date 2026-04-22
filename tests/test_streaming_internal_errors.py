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

import json
import os
from json.decoder import JSONDecodeError

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.streaming import decode_stream_error_chunk
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
            parsed = decode_stream_error_chunk(chunk)
            if parsed and parsed.get("code") == "rail_execution_failure":
                error_chunks.append(parsed)
        except (JSONDecodeError, TypeError, ValueError):
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
    assert error["type"] == "internal_error"
    assert error["code"] == "rail_execution_failure"
    assert "Internal error" in error["message"]
    assert "failing safety check" in error["message"]
    assert "Action failing_rail_action failed with status: failed" in error["message"]
    assert error["param"] == "failing safety check"


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

    assert "type" in error
    assert error["type"] == "internal_error"

    assert "code" in error
    assert error["code"] == "rail_execution_failure"

    assert "message" in error
    assert "Internal error in test rail rail:" in error["message"]

    assert "param" in error
    assert error["param"] == "test rail"
