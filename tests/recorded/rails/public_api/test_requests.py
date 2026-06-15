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

from dataclasses import dataclass
from typing import Any

import pytest

from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.assertions import (
    assert_generated_message,
    assert_generation_response,
    assert_llm_tasks,
    assert_request_payload,
    assert_stream_contract,
)
from tests.recorded.conftest import provider_key
from tests.recorded.normalization import normalize_generation_response, normalize_stream_chunks
from tests.recorded.rails.helpers import build_rails
from tests.recorded.rails.public_api.configs import (
    NIM_BASELINE_CONFIG,
    NIM_MODEL,
    OPENAI_BASELINE_CONFIG,
    OPENAI_MODEL,
    TASK_MODELS_CONFIG,
)
from tests.recorded.rails_config import RailsConfigSource
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


@dataclass(frozen=True)
class LLMParamScenario:
    id: str
    provider: str
    config: RailsConfigSource
    model: str
    llm_params: dict[str, Any]
    expected_params: dict[str, Any]
    absent_params: set[str]


SCENARIOS = [
    LLMParamScenario(
        id="openai",
        provider="openai",
        config=OPENAI_BASELINE_CONFIG,
        model=OPENAI_MODEL,
        llm_params={"temperature": 0.0, "max_tokens": 8},
        expected_params={"max_completion_tokens": 8},
        absent_params={"temperature", "max_tokens"},
    ),
    LLMParamScenario(
        id="nim",
        provider="nim",
        config=NIM_BASELINE_CONFIG,
        model=NIM_MODEL,
        llm_params={
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        expected_params={
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        absent_params=set(),
    ),
]


OPENAI_SCENARIO = SCENARIOS[0]
NIM_SCENARIO = SCENARIOS[1]


async def _run_generate_request(request, record_mode, recorded_cassette_path, scenario):
    provider_key(request, scenario.provider)
    rails = build_rails(scenario.config)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "Reply with one short greeting."}],
        options={"llm_params": scenario.llm_params},
    )

    assert isinstance(result, GenerationResponse)
    assert result.response
    assert_generated_message(result.response[-1])
    if record_mode == "none":
        assert_request_payload(
            recorded_cassette_path,
            model=scenario.model,
            expected_params=scenario.expected_params,
            absent_params=scenario.absent_params,
        )
    return normalize_generation_response(result)


async def test_openai_llm_params_generate_async_request(request, record_mode, recorded_cassette_path):
    assert await _run_generate_request(request, record_mode, recorded_cassette_path, OPENAI_SCENARIO) == snapshot(
        {"response": [{"role": "assistant", "content": "Hello!"}], "activated_rails": [], "llm_calls": []}
    )


async def test_nim_llm_params_generate_async_request(request, record_mode, recorded_cassette_path):
    assert await _run_generate_request(request, record_mode, recorded_cassette_path, NIM_SCENARIO) == snapshot(
        {"response": [{"role": "assistant", "content": "Hello!"}], "activated_rails": [], "llm_calls": []}
    )


async def _run_stream_request(request, record_mode, recorded_cassette_path, scenario):
    provider_key(request, scenario.provider)
    rails = build_rails(scenario.config)

    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "Reply with one short greeting."}],
        options={"llm_params": scenario.llm_params},
    ):
        chunks.append(chunk)

    assert_stream_contract(chunks, expect_multiple=False)
    if record_mode == "none":
        assert_request_payload(
            recorded_cassette_path,
            model=scenario.model,
            stream=True,
            expected_params=scenario.expected_params,
            absent_params=scenario.absent_params,
        )
    return normalize_stream_chunks(chunks)


async def test_openai_llm_params_stream_async_request(request, record_mode, recorded_cassette_path):
    assert await _run_stream_request(request, record_mode, recorded_cassette_path, OPENAI_SCENARIO) == snapshot(
        {"content": "Hello!", "chunks": ["", "Hello", "!", "", ""], "errors": []}
    )


async def test_nim_llm_params_stream_async_request(request, record_mode, recorded_cassette_path):
    assert await _run_stream_request(request, record_mode, recorded_cassette_path, NIM_SCENARIO) == snapshot(
        {"content": "Hello!", "chunks": ["", "Hello", "!", ""], "errors": []}
    )


async def test_task_specific_models_generate_async(openai_api_key, record_mode, recorded_cassette_path):
    rails = build_rails(TASK_MODELS_CONFIG)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "hello"}],
        options={"log": {"llm_calls": True}},
    )

    assert_generation_response(result)
    assert_llm_tasks(result, {"generate_user_intent"})
    if record_mode == "none":
        assert_request_payload(recorded_cassette_path, model=OPENAI_MODEL)
    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "Hi! 👋 How can I help you today?"}],
            "activated_rails": [],
            "llm_calls": [
                {
                    "task": "generate_user_intent",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": "Hello! How can I assist you today?",
                    "prompt_tokens": 568,
                    "completion_tokens": 12,
                    "total_tokens": 580,
                },
                {
                    "task": "generate_next_steps",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": "Hello! 👋 How can I assist you today?",
                    "prompt_tokens": 228,
                    "completion_tokens": 14,
                    "total_tokens": 242,
                },
                {
                    "task": "generate_bot_message",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": "Hi! 👋 How can I help you today?",
                    "prompt_tokens": 678,
                    "completion_tokens": 14,
                    "total_tokens": 692,
                },
            ],
        }
    )
