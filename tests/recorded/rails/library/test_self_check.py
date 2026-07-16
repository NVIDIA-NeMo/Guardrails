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

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.rails.llm.options import GenerationResponse, RailStatus, RailType
from tests.recorded.assertions import (
    assert_blocked_stream_error,
    assert_rails_result,
)
from tests.recorded.normalization import normalize_rails_result, normalize_stream_chunks
from tests.recorded.rails.helpers import async_chunks, build_rails
from tests.recorded.rails.library.configs import (
    OPENAI_MULTI_SELF_CHECK_CONFIG,
    OPENAI_MULTI_SELF_CHECK_INVALID_MODEL_CONFIG,
    OPENAI_SELF_CHECK_CONFIG,
)
from tests.recorded.rails.library.helpers import check_rails
from tests.recorded.rails_config import enable_streaming, load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


def _llm_routes(rails, start):
    """Return recorded task, provider, and model routes from the given call index."""
    return [(call.task, call.llm_provider_name, call.llm_model_name) for call in rails.explain().llm_calls[start:]]


def _self_check_routes(rails, start):
    """Return self-check routes from the given call index."""
    return [route for route in _llm_routes(rails, start) if route[0] and route[0].startswith("self_check_")]


async def test_self_check_input_blocks_user_message(openai_api_key):
    result = await check_rails(
        OPENAI_SELF_CHECK_CONFIG,
        [{"role": "user", "content": "blocked_self_check_input"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check input")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "self check input", "content": "I'm sorry, I can't respond to that."}
    )


async def test_self_check_output_blocks_assistant_message(openai_api_key):
    result = await check_rails(
        OPENAI_SELF_CHECK_CONFIG,
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "blocked_self_check_output"}],
        rail_types=(RailType.OUTPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check output")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "self check output", "content": "I'm sorry, I can't respond to that."}
    )


async def test_multiple_self_check_input_second_task_blocks(openai_api_key):
    """Verify the second sequential input task blocks and records its route."""
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG)
    start = len(rails.explain().llm_calls)

    result = await rails.check_async(
        [{"role": "user", "content": "blocked_off_topic"}],
        rail_types=[RailType.INPUT],
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check input $variant=check_off_topic")
    assert _llm_routes(rails, start) == [
        ("self_check_input $variant=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $variant=check_off_topic", "openai", "gpt-5.4-nano"),
    ]
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "self check input $variant=check_off_topic",
            "content": "I'm sorry, I can't respond to that.",
        }
    )


async def test_multiple_self_check_input_tasks_allow(openai_api_key):
    """Verify all sequential input tasks run when they allow the message."""
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG)
    start = len(rails.explain().llm_calls)

    result = await rails.check_async(
        [{"role": "user", "content": "allowed_multi_self_check_input"}],
        rail_types=[RailType.INPUT],
    )

    assert_rails_result(result, status=RailStatus.PASSED)
    assert _llm_routes(rails, start) == [
        ("self_check_input $variant=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $variant=check_off_topic", "openai", "gpt-5.4-nano"),
    ]
    assert normalize_rails_result(result) == snapshot(
        {"status": "passed", "rail": None, "content": "allowed_multi_self_check_input"}
    )


async def test_multiple_self_check_output_second_task_blocks(openai_api_key):
    """Verify the second sequential output task blocks and records its route."""
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG)
    start = len(rails.explain().llm_calls)

    result = await rails.check_async(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "blocked_data_leakage"},
        ],
        rail_types=[RailType.OUTPUT],
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check output $variant=check_data_leakage")
    assert _llm_routes(rails, start) == [
        ("self_check_output $variant=check_inappropriate", "openai", "gpt-5.4-nano"),
        ("self_check_output $variant=check_data_leakage", "openai", "gpt-4.1-mini"),
    ]
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "self check output $variant=check_data_leakage",
            "content": "I'm sorry, I can't respond to that.",
        }
    )


async def test_multiple_self_check_generate_async_sequential_blocks_output(openai_api_key):
    """Verify sequential self-check tasks block generated output."""
    rails = build_rails(
        OPENAI_MULTI_SELF_CHECK_CONFIG,
        llm=FakeLLMModel(responses=["blocked_data_leakage"]),
    )
    start = len(rails.explain().llm_calls)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "allowed_multi_self_check_input"}],
        options={"log": {"activated_rails": True, "llm_calls": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert result.response == snapshot([{"role": "assistant", "content": "I'm sorry, I can't respond to that."}])
    assert _self_check_routes(rails, start) == [
        ("self_check_input $variant=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $variant=check_off_topic", "openai", "gpt-5.4-nano"),
        ("self_check_output $variant=check_inappropriate", "openai", "gpt-5.4-nano"),
        ("self_check_output $variant=check_data_leakage", "openai", "gpt-4.1-mini"),
    ]


async def test_multiple_self_check_generate_async_parallel_runs_all_tasks(openai_api_key):
    """Verify parallel generation runs every configured self-check task."""
    config = load_config(OPENAI_MULTI_SELF_CHECK_CONFIG)
    config.rails.input.parallel = True
    config.rails.output.parallel = True
    rails = LLMRails(config, llm=FakeLLMModel(responses=["allowed_parallel_response"]), verbose=False)
    start = len(rails.explain().llm_calls)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "allowed_multi_self_check_input"}],
        options={"log": {"activated_rails": True, "llm_calls": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert result.response == snapshot([{"role": "assistant", "content": "allowed_parallel_response"}])
    assert sorted(_self_check_routes(rails, start)) == [
        ("self_check_input $variant=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $variant=check_off_topic", "openai", "gpt-5.4-nano"),
        ("self_check_output $variant=check_data_leakage", "openai", "gpt-4.1-mini"),
        ("self_check_output $variant=check_inappropriate", "openai", "gpt-5.4-nano"),
    ]


async def test_multiple_self_check_stream_async_runs_input_tasks(openai_api_key):
    """Verify streaming generation runs every configured input self-check task."""
    config = enable_streaming(load_config(OPENAI_MULTI_SELF_CHECK_CONFIG))
    config.rails.output.flows = []
    rails = LLMRails(config, llm=FakeLLMModel(responses=["allowed_streaming_response"]), verbose=False)
    start = len(rails.explain().llm_calls)

    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "allowed_multi_self_check_input"}],
    ):
        chunks.append(chunk)

    assert _self_check_routes(rails, start) == [
        ("self_check_input $variant=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $variant=check_off_topic", "openai", "gpt-5.4-nano"),
    ]
    assert normalize_stream_chunks(chunks) == snapshot(
        {"content": "allowed_streaming_response", "chunks": ["allowed_streaming_response"], "errors": []}
    )


async def test_multiple_self_check_output_second_task_blocks_fake_main_stream(openai_api_key):
    """Verify the second output task blocks content from a supplied stream."""
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG, streaming=True)
    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "hello"}],
        generator=async_chunks(["blocked_data_leakage"]),
        options={"rails": ["output"]},
    ):
        chunks.append(chunk)

    assert_blocked_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "blocked_data_leakage",
            "chunks": [
                "blocked_data_leakage",
                '{"error": {"message": "Blocked by self check output $variant=check_data_leakage rails.", "type": "guardrails_violation", "param": "self check output $variant=check_data_leakage", "code": "content_blocked"}}',
            ],
            "errors": [
                {
                    "error": {
                        "message": "Blocked by self check output $variant=check_data_leakage rails.",
                        "type": "guardrails_violation",
                        "param": "self check output $variant=check_data_leakage",
                        "code": "content_blocked",
                    }
                }
            ],
        }
    )


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential-blocks", "parallel-allows"])
async def test_multiple_self_check_output_rails_streaming(openai_api_key, parallel):
    """Verify sequential and parallel output tasks use their configured routes."""
    config = enable_streaming(load_config(OPENAI_MULTI_SELF_CHECK_CONFIG))
    config.rails.input.flows = []
    config.rails.output.parallel = parallel
    config.rails.output.streaming.enabled = True
    config.rails.output.streaming.chunk_size = 1
    config.rails.output.streaming.context_size = 0
    config.rails.output.streaming.stream_first = False
    rails = LLMRails(config, verbose=False)
    start = len(rails.explain().llm_calls)

    output = "allowed_output_streaming_response" if parallel else "blocked_data_leakage"
    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "hello"}],
        generator=async_chunks([output]),
        options={"rails": ["output"]},
    ):
        chunks.append(chunk)

    if not parallel:
        assert_blocked_stream_error(chunks)
    expected_routes = [
        ("self_check_output $variant=check_inappropriate", "openai", "gpt-5.4-nano"),
        ("self_check_output $variant=check_data_leakage", "openai", "gpt-4.1-mini"),
    ]
    routes = _self_check_routes(rails, start)
    if parallel:
        assert sorted(routes) == sorted(expected_routes * 2)
        assert normalize_stream_chunks(chunks) == snapshot(
            {
                "content": "allowed_output_streaming_response",
                "chunks": ["allowed_output_streaming_response"],
                "errors": [],
            }
        )
    else:
        assert routes == expected_routes
        assert normalize_stream_chunks(chunks) == snapshot(
            {
                "content": "",
                "chunks": [
                    '{"error": {"message": "Blocked by self check output $variant=check_data_leakage rails.", "type": "guardrails_violation", "param": "self check output $variant=check_data_leakage", "code": "content_blocked"}}'
                ],
                "errors": [
                    {
                        "error": {
                            "message": "Blocked by self check output $variant=check_data_leakage rails.",
                            "type": "guardrails_violation",
                            "param": "self check output $variant=check_data_leakage",
                            "code": "content_blocked",
                        }
                    }
                ],
            }
        )


async def test_multiple_self_check_input_provider_error_raises(openai_api_key):
    """Verify provider errors from a custom input task are propagated."""
    with pytest.raises(LLMCallException) as exc_info:
        await check_rails(
            OPENAI_MULTI_SELF_CHECK_INVALID_MODEL_CONFIG,
            [{"role": "user", "content": "hello"}],
            rail_types=(RailType.INPUT,),
        )

    assert getattr(exc_info.value.inner_exception, "status_code", None) == 404


async def test_self_check_facts_blocks_unsupported_response(openai_api_key):
    result = await check_rails(
        OPENAI_SELF_CHECK_CONFIG,
        [
            {"role": "context", "content": {"check_facts": True, "relevant_chunks": "Paris is in France."}},
            {"role": "user", "content": "Where is Paris?"},
            {"role": "assistant", "content": "Paris is in Germany."},
        ],
        rail_types=(RailType.OUTPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check facts")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "self check facts", "content": "I'm sorry, I can't respond to that."}
    )


async def test_self_check_output_blocks_fake_main_stream(openai_api_key):
    rails = build_rails(OPENAI_SELF_CHECK_CONFIG, streaming=True)
    chunks = []
    async for chunk in rails.stream_async(
        messages=[{"role": "user", "content": "hello"}],
        generator=async_chunks(["blocked_self_check_output"]),
        options={"rails": ["output"]},
    ):
        chunks.append(chunk)

    assert_blocked_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "blocked_self_check_output",
            "chunks": [
                "blocked_self_check_output",
                '{"error": {"message": "Blocked by self check output rails.", "type": "guardrails_violation", "param": "self check output", "code": "content_blocked"}}',
            ],
            "errors": [
                {
                    "error": {
                        "message": "Blocked by self check output rails.",
                        "type": "guardrails_violation",
                        "param": "self check output",
                        "code": "content_blocked",
                    }
                }
            ],
        }
    )
