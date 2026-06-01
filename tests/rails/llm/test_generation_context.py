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

from typing import Any, cast

import pytest

from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_stats_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.rails.llm.generation import generation_context
from nemoguardrails.rails.llm.generation.generation_context import (
    bind_generation_request_context,
    close_current_streaming_handler,
    ensure_explain_info,
    explain_info_for_current_context,
    set_generation_options_context,
    start_generation_request_context,
    start_generation_stats,
)
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import END_OF_STREAM


def test_generation_context_exports():
    assert generation_context.__all__ == [
        "GenerationRequestContext",
        "bind_generation_request_context",
        "close_current_streaming_handler",
        "ensure_explain_info",
        "explain_info_for_current_context",
        "set_generation_options_context",
        "start_generation_request_context",
        "start_generation_stats",
    ]


@pytest.fixture(autouse=True)
def reset_generation_context_vars():
    tokens = [
        explain_info_var.set(None),
        generation_options_var.set(None),
        llm_stats_var.set(None),
        raw_llm_request.set(None),
        streaming_handler_var.set(None),
    ]
    try:
        yield
    finally:
        for var, token in reversed(
            [
                (explain_info_var, tokens[0]),
                (generation_options_var, tokens[1]),
                (llm_stats_var, tokens[2]),
                (raw_llm_request, tokens[3]),
                (streaming_handler_var, tokens[4]),
            ]
        ):
            var.reset(token)


def test_ensure_explain_info_creates_and_reuses_context_value():
    explain_info = ensure_explain_info()

    assert isinstance(explain_info, ExplainInfo)
    assert explain_info_var.get() is explain_info
    assert ensure_explain_info() is explain_info


def test_explain_info_for_current_context_prefers_context_then_fallback():
    fallback = ExplainInfo()
    request_explain_info = ExplainInfo()

    assert explain_info_for_current_context(fallback) is fallback

    explain_info_var.set(request_explain_info)

    assert explain_info_for_current_context(fallback) is request_explain_info


def test_explain_info_for_current_context_creates_context_when_needed():
    explain_info = explain_info_for_current_context(None)

    assert isinstance(explain_info, ExplainInfo)
    assert explain_info_var.get() is explain_info


def test_set_generation_options_context():
    options = GenerationOptions()

    set_generation_options_context(options)

    assert generation_options_var.get() is options


def test_bind_generation_request_context_sets_request_vars():
    handler = cast(Any, object())
    messages = [{"role": "user", "content": "Hi"}]

    explain_info = bind_generation_request_context(
        messages=messages,
        streaming_handler=handler,
    )

    assert explain_info_var.get() is explain_info
    assert streaming_handler_var.get() is handler
    assert raw_llm_request.get() == messages


def test_bind_generation_request_context_preserves_existing_streaming_handler_when_omitted():
    handler = cast(Any, object())
    streaming_handler_var.set(handler)

    bind_generation_request_context(
        messages=[{"role": "user", "content": "Hi"}],
        streaming_handler=None,
    )

    assert streaming_handler_var.get() is handler


def test_start_generation_stats_sets_stats_and_returns_processing_log():
    llm_stats, processing_log = start_generation_stats()

    assert isinstance(llm_stats, LLMStats)
    assert llm_stats_var.get() is llm_stats
    assert processing_log == []


@pytest.mark.asyncio
async def test_generation_request_context_restores_previous_bindings_and_closes_stream_once():
    class StreamingHandler:
        def __init__(self):
            self.chunks = []

        async def push_chunk(self, chunk):
            self.chunks.append(chunk)

    outer_options = GenerationOptions(llm_params={"request_id": "outer"})
    inner_options = GenerationOptions(llm_params={"request_id": "inner"})
    outer_messages = [{"role": "user", "content": "outer"}]
    inner_messages = [{"role": "user", "content": "inner"}]
    outer_explain_info = ExplainInfo()
    outer_stats = LLMStats()
    outer_streaming_handler = cast(Any, object())
    inner_streaming_handler = StreamingHandler()
    tokens = [
        generation_options_var.set(outer_options),
        raw_llm_request.set(outer_messages),
        explain_info_var.set(outer_explain_info),
        llm_stats_var.set(outer_stats),
        streaming_handler_var.set(outer_streaming_handler),
    ]
    try:
        request_context = start_generation_request_context(
            gen_options=inner_options,
            messages=inner_messages,
            streaming_handler=cast(Any, inner_streaming_handler),
        )

        assert request_context.explain_info is outer_explain_info
        assert generation_options_var.get() is inner_options
        assert raw_llm_request.get() is inner_messages
        assert streaming_handler_var.get() is inner_streaming_handler

        start_generation_stats()
        await request_context.close_streaming_handler()
        await request_context.close()
        await request_context.close()

        assert inner_streaming_handler.chunks == [END_OF_STREAM]
        assert generation_options_var.get() is outer_options
        assert raw_llm_request.get() is outer_messages
        assert explain_info_var.get() is outer_explain_info
        assert llm_stats_var.get() is outer_stats
        assert streaming_handler_var.get() is outer_streaming_handler
    finally:
        for var, token in reversed(
            [
                (generation_options_var, tokens[0]),
                (raw_llm_request, tokens[1]),
                (explain_info_var, tokens[2]),
                (llm_stats_var, tokens[3]),
                (streaming_handler_var, tokens[4]),
            ]
        ):
            var.reset(token)


@pytest.mark.asyncio
async def test_generation_request_context_restores_previous_bindings_when_stream_close_fails():
    class FailingStreamingHandler:
        async def push_chunk(self, chunk):
            assert chunk is END_OF_STREAM
            raise RuntimeError("stream close failed")

    outer_options = GenerationOptions(llm_params={"request_id": "outer"})
    inner_options = GenerationOptions(llm_params={"request_id": "inner"})
    outer_messages = [{"role": "user", "content": "outer"}]
    inner_messages = [{"role": "user", "content": "inner"}]
    outer_explain_info = ExplainInfo()
    outer_stats = LLMStats()
    outer_streaming_handler = cast(Any, object())
    tokens = [
        generation_options_var.set(outer_options),
        raw_llm_request.set(outer_messages),
        explain_info_var.set(outer_explain_info),
        llm_stats_var.set(outer_stats),
        streaming_handler_var.set(outer_streaming_handler),
    ]
    try:
        request_context = start_generation_request_context(
            gen_options=inner_options,
            messages=inner_messages,
            streaming_handler=cast(Any, FailingStreamingHandler()),
        )

        start_generation_stats()
        with pytest.raises(RuntimeError, match="stream close failed"):
            await request_context.close()

        assert generation_options_var.get() is outer_options
        assert raw_llm_request.get() is outer_messages
        assert explain_info_var.get() is outer_explain_info
        assert llm_stats_var.get() is outer_stats
        assert streaming_handler_var.get() is outer_streaming_handler
    finally:
        for var, token in reversed(
            [
                (generation_options_var, tokens[0]),
                (raw_llm_request, tokens[1]),
                (explain_info_var, tokens[2]),
                (llm_stats_var, tokens[3]),
                (streaming_handler_var, tokens[4]),
            ]
        ):
            var.reset(token)


@pytest.mark.asyncio
async def test_close_current_streaming_handler_pushes_end_of_stream():
    class StreamingHandler:
        def __init__(self):
            self.chunks = []

        async def push_chunk(self, chunk):
            self.chunks.append(chunk)

    handler = StreamingHandler()
    streaming_handler_var.set(cast(Any, handler))

    await close_current_streaming_handler()

    assert handler.chunks == [END_OF_STREAM]


@pytest.mark.asyncio
async def test_close_current_streaming_handler_ignores_empty_context():
    await close_current_streaming_handler()
