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
import logging
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_stats_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.streaming import END_OF_STREAM
from tests.utils import FakeLLMModel

COLANG = """
define user express greeting
  "hi"

define flow
  user express greeting
  $user_greeted = True
  bot express greeting

define bot express greeting
  "Hello there!"
"""


def _config(yaml_content: str | None = None) -> RailsConfig:
    return RailsConfig.from_content(colang_content=COLANG, yaml_content=yaml_content)


def _rails(config: RailsConfig | None = None) -> LLMRails:
    return LLMRails(config=config or _config(), llm=FakeLLMModel(responses=["  express greeting"]))


@pytest.mark.asyncio
async def test_generate_standard_info_logs_keep_llmrails_logger_name(caplog):
    with caplog.at_level(logging.INFO):
        await _rails().generate_async(messages=[{"role": "user", "content": "hi"}])

    total_processing_logs = [record for record in caplog.records if "--- :: Total processing took" in record.message]

    assert [record.name for record in total_processing_logs] == ["nemoguardrails.rails.llm.llmrails"]


@pytest.mark.asyncio
async def test_generate_prompt_and_messages_return_shapes_without_options():
    prompt_result = await _rails().generate_async(prompt="hi")
    assert prompt_result == "Hello there!"

    messages_result = await _rails().generate_async(messages=[{"role": "user", "content": "hi"}])
    assert messages_result == {"role": "assistant", "content": "Hello there!"}


@pytest.mark.asyncio
async def test_generate_options_dict_and_object_return_generation_response():
    dict_result = await _rails().generate_async(
        prompt="hi",
        options={"output_vars": ["user_greeted"]},
    )

    assert isinstance(dict_result, GenerationResponse)
    assert dict_result.response == "Hello there!"
    assert dict_result.output_data == {"user_greeted": True}

    options = GenerationOptions(output_vars=["user_greeted"])
    object_result = await _rails().generate_async(
        messages=[{"role": "user", "content": "hi"}],
        options=options,
    )

    assert isinstance(object_result, GenerationResponse)
    assert object_result.response == [{"role": "assistant", "content": "Hello there!"}]
    assert object_result.output_data == {"user_greeted": True}


@pytest.mark.asyncio
async def test_generate_state_forces_generation_response_without_options():
    state = {"events": []}

    result = await _rails().generate_async(
        messages=[{"role": "user", "content": "hi"}],
        state=state,
    )

    assert isinstance(result, GenerationResponse)
    assert result.response == [{"role": "assistant", "content": "Hello there!"}]
    assert result.state is not None
    assert "events" in result.state
    assert state == {"events": []}


@pytest.mark.asyncio
async def test_generate_tracing_enabled_returns_generation_response_without_requested_log():
    config = _config(
        yaml_content="""
tracing:
  enabled: true
  adapters: []
"""
    )

    result = await _rails(config).generate_async(prompt="hi")

    assert isinstance(result, GenerationResponse)
    assert result.response == "Hello there!"
    assert result.log is None


@pytest.mark.asyncio
async def test_generate_tracing_does_not_mutate_generation_options_object():
    config = _config(
        yaml_content="""
tracing:
  enabled: true
  adapters: []
"""
    )
    options = GenerationOptions()

    result = await _rails(config).generate_async(prompt="hi", options=options)

    assert isinstance(result, GenerationResponse)
    assert result.response == "Hello there!"
    assert result.log is None
    assert options.log.activated_rails is False
    assert options.log.llm_calls is False
    assert options.log.internal_events is False


@pytest.mark.asyncio
async def test_generate_does_not_mutate_input_messages_or_options_dict():
    messages = [{"role": "user", "content": "hi"}]
    options = {
        "rails": ["input", "output", "retrieval", "dialog", "tool_input", "tool_output"],
        "output_vars": ["user_greeted"],
    }
    original_messages = deepcopy(messages)
    original_options = deepcopy(options)

    result = await _rails().generate_async(messages=messages, options=options)

    assert isinstance(result, GenerationResponse)
    assert result.response == [{"role": "assistant", "content": "Hello there!"}]
    assert messages == original_messages
    assert options == original_options


@pytest.mark.asyncio
async def test_concurrent_generate_requests_keep_context_isolated():
    token_options = generation_options_var.set(None)
    token_request = raw_llm_request.set(None)
    snapshots = {}
    both_requests_started = asyncio.Event()

    async def fake_run_colang_turn(rails, events, state, processing_log):
        request = cast(list[dict[str, Any]], deepcopy(raw_llm_request.get()))
        options = cast(GenerationOptions, generation_options_var.get())
        prompt = request[0]["content"]
        snapshots[prompt] = {
            "request": request,
            "llm_params": deepcopy(options.llm_params),
        }

        if len(snapshots) == 2:
            both_requests_started.set()

        await both_requests_started.wait()
        processing_log.extend(
            [
                {"type": "noop", "timestamp": 0.0},
                {"type": "noop", "timestamp": 0.1},
            ]
        )
        return [
            {
                "type": "StartUtteranceBotAction",
                "script": f"reply {prompt}",
            }
        ]

    try:
        rails = _rails()

        with patch(
            "nemoguardrails.rails.llm.generation.generation_workflow.run_colang_turn",
            fake_run_colang_turn,
        ):
            first, second = await asyncio.gather(
                rails.generate_async(
                    prompt="first",
                    options={"llm_params": {"request_id": "first"}},
                ),
                rails.generate_async(
                    prompt="second",
                    options={"llm_params": {"request_id": "second"}},
                ),
            )

        first = cast(GenerationResponse, first)
        second = cast(GenerationResponse, second)
        assert first.response == "reply first"
        assert second.response == "reply second"
        assert snapshots == {
            "first": {
                "request": [{"role": "user", "content": "first"}],
                "llm_params": {"request_id": "first"},
            },
            "second": {
                "request": [{"role": "user", "content": "second"}],
                "llm_params": {"request_id": "second"},
            },
        }
        assert generation_options_var.get() is None
        assert raw_llm_request.get() is None

    finally:
        generation_options_var.reset(token_options)
        raw_llm_request.reset(token_request)


@pytest.mark.asyncio
async def test_generate_resets_request_context_after_success_in_same_task():
    outer_options = GenerationOptions(llm_params={"request_id": "outer"})
    outer_request = [{"role": "user", "content": "outer"}]
    outer_explain_info = ExplainInfo()
    outer_stats = LLMStats()
    token_options = generation_options_var.set(outer_options)
    token_request = raw_llm_request.set(outer_request)
    token_explain_info = explain_info_var.set(outer_explain_info)
    token_stats = llm_stats_var.set(outer_stats)
    token_streaming_handler = streaming_handler_var.set(None)

    try:
        result = await _rails().generate_async(
            prompt="hi",
            options={"llm_params": {"request_id": "inner"}},
        )

        assert isinstance(result, GenerationResponse)
        assert result.response == "Hello there!"
        assert generation_options_var.get() is outer_options
        assert raw_llm_request.get() is outer_request
        assert explain_info_var.get() is outer_explain_info
        request_stats = llm_stats_var.get()
        assert isinstance(request_stats, LLMStats)
        assert request_stats is not outer_stats
        assert streaming_handler_var.get() is None
    finally:
        streaming_handler_var.reset(token_streaming_handler)
        llm_stats_var.reset(token_stats)
        explain_info_var.reset(token_explain_info)
        raw_llm_request.reset(token_request)
        generation_options_var.reset(token_options)


@pytest.mark.asyncio
async def test_generate_resets_request_context_after_failure_in_same_task():
    outer_options = GenerationOptions(llm_params={"request_id": "outer"})
    outer_request = [{"role": "user", "content": "outer"}]
    outer_explain_info = ExplainInfo()
    outer_stats = LLMStats()
    token_options = generation_options_var.set(outer_options)
    token_request = raw_llm_request.set(outer_request)
    token_explain_info = explain_info_var.set(outer_explain_info)
    token_stats = llm_stats_var.set(outer_stats)
    token_streaming_handler = streaming_handler_var.set(None)

    async def failing_run_colang_turn(rails, events, state, processing_log):
        raise RuntimeError("generation failed")

    try:
        rails = _rails()
        with patch(
            "nemoguardrails.rails.llm.generation.generation_workflow.run_colang_turn",
            failing_run_colang_turn,
        ):
            with pytest.raises(RuntimeError, match="generation failed"):
                await rails.generate_async(
                    prompt="hi",
                    options={"llm_params": {"request_id": "inner"}},
                )

        assert generation_options_var.get() is outer_options
        assert raw_llm_request.get() is outer_request
        assert explain_info_var.get() is outer_explain_info
        request_stats = llm_stats_var.get()
        assert isinstance(request_stats, LLMStats)
        assert request_stats is not outer_stats
        assert streaming_handler_var.get() is None
    finally:
        streaming_handler_var.reset(token_streaming_handler)
        llm_stats_var.reset(token_stats)
        explain_info_var.reset(token_explain_info)
        raw_llm_request.reset(token_request)
        generation_options_var.reset(token_options)


@pytest.mark.asyncio
async def test_generate_closes_request_streaming_handler_once_on_success():
    class StreamingHandler:
        def __init__(self):
            self.chunks = []

        async def push_chunk(self, chunk):
            self.chunks.append(chunk)

    streaming_handler = StreamingHandler()

    result = await _rails().generate_async(
        prompt="hi",
        streaming_handler=cast(Any, streaming_handler),
    )

    assert result == "Hello there!"
    assert streaming_handler.chunks.count(END_OF_STREAM) == 1
    assert streaming_handler.chunks[-1] is END_OF_STREAM


@pytest.mark.asyncio
async def test_generate_closes_request_streaming_handler_once_on_failure():
    class StreamingHandler:
        def __init__(self):
            self.chunks = []

        async def push_chunk(self, chunk):
            self.chunks.append(chunk)

    async def failing_run_colang_turn(rails, events, state, processing_log):
        raise RuntimeError("generation failed")

    streaming_handler = StreamingHandler()

    with patch(
        "nemoguardrails.rails.llm.generation.generation_workflow.run_colang_turn",
        failing_run_colang_turn,
    ):
        with pytest.raises(RuntimeError, match="generation failed"):
            await _rails().generate_async(
                prompt="hi",
                streaming_handler=cast(Any, streaming_handler),
            )

    assert streaming_handler.chunks == [END_OF_STREAM]


@pytest.mark.asyncio
async def test_generate_closes_request_streaming_handler_before_trace_export():
    order = []

    class StreamingHandler:
        async def push_chunk(self, chunk):
            if chunk is END_OF_STREAM:
                order.append("stream-closed")

    async def export_trace(**kwargs):
        order.append("trace-exported")

    config = _config(
        yaml_content="""
tracing:
  enabled: true
  adapters: []
"""
    )

    with patch(
        "nemoguardrails.rails.llm.generation.generation_workflow.export_generation_trace",
        export_trace,
    ):
        result = await _rails(config).generate_async(
            prompt="hi",
            streaming_handler=cast(Any, StreamingHandler()),
        )

    assert isinstance(result, GenerationResponse)
    assert result.response == "Hello there!"
    assert order == ["stream-closed", "trace-exported"]


@pytest.mark.asyncio
async def test_generate_colang_1_writes_implicit_history_cache_when_state_is_none():
    rails = _rails()

    result = await rails.generate_async(messages=[{"role": "user", "content": "hi"}])

    assert result == {"role": "assistant", "content": "Hello there!"}
    assert len(rails.events_history_cache) == 1


@pytest.mark.asyncio
async def test_generate_colang_1_does_not_write_implicit_history_cache_with_explicit_state():
    rails = _rails()

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "hi"}],
        state={"events": []},
    )

    assert isinstance(result, GenerationResponse)
    assert result.response == [{"role": "assistant", "content": "Hello there!"}]
    assert rails.events_history_cache == {}


def test_env_api_key_model_kwargs_do_not_write_back_to_config(monkeypatch):
    monkeypatch.setenv("NGR_TEST_API_KEY", "env-secret")
    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
                api_key_env_var="NGR_TEST_API_KEY",
                parameters={"base_url": "https://example.test"},
            )
        ]
    )

    with patch("nemoguardrails.rails.llm.llmrails.init_llm_model") as init_llm:
        init_llm.return_value = FakeLLMModel(responses=[])
        LLMRails(config=config)

    init_kwargs = init_llm.call_args.kwargs["kwargs"]
    assert init_kwargs["api_key"] == "env-secret"
    assert init_kwargs["base_url"] == "https://example.test"
    assert config.models[0].parameters == {"base_url": "https://example.test"}
