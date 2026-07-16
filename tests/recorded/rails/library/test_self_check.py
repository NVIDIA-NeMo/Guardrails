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

from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.assertions import (
    assert_blocked_stream_error,
    assert_rails_result,
)
from tests.recorded.normalization import normalize_rails_result, normalize_stream_chunks
from tests.recorded.rails.helpers import build_rails
from tests.recorded.rails.library.configs import (
    OPENAI_MULTI_SELF_CHECK_CONFIG,
    OPENAI_MULTI_SELF_CHECK_INVALID_MODEL_CONFIG,
    OPENAI_SELF_CHECK_CONFIG,
)
from tests.recorded.rails.library.helpers import check_rails, stream_with_fake_main
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


def _llm_routes(rails, start):
    return [(call.task, call.llm_provider_name, call.llm_model_name) for call in rails.explain().llm_calls[start:]]


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
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG)
    start = len(rails.explain().llm_calls)

    result = await rails.check_async(
        [{"role": "user", "content": "blocked_off_topic"}],
        rail_types=[RailType.INPUT],
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check input $task=check_off_topic")
    assert _llm_routes(rails, start) == [
        ("self_check_input $task=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $task=check_off_topic", "openai", "gpt-5.4-nano"),
    ]
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "self check input $task=check_off_topic",
            "content": "I'm sorry, I can't respond to that.",
        }
    )


async def test_multiple_self_check_input_tasks_allow(openai_api_key):
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG)
    start = len(rails.explain().llm_calls)

    result = await rails.check_async(
        [{"role": "user", "content": "allowed_multi_self_check_input"}],
        rail_types=[RailType.INPUT],
    )

    assert_rails_result(result, status=RailStatus.PASSED)
    assert _llm_routes(rails, start) == [
        ("self_check_input $task=check_harmful", "openai", "gpt-4.1-mini"),
        ("self_check_input $task=check_off_topic", "openai", "gpt-5.4-nano"),
    ]
    assert normalize_rails_result(result) == snapshot(
        {"status": "passed", "rail": None, "content": "allowed_multi_self_check_input"}
    )


async def test_multiple_self_check_output_second_task_blocks(openai_api_key):
    rails = build_rails(OPENAI_MULTI_SELF_CHECK_CONFIG)
    start = len(rails.explain().llm_calls)

    result = await rails.check_async(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "blocked_data_leakage"},
        ],
        rail_types=[RailType.OUTPUT],
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="self check output $task=check_data_leakage")
    assert _llm_routes(rails, start) == [
        ("self_check_output $task=check_inappropriate", "openai", "gpt-5.4-nano"),
        ("self_check_output $task=check_data_leakage", "openai", "gpt-4.1-mini"),
    ]
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "self check output $task=check_data_leakage",
            "content": "I'm sorry, I can't respond to that.",
        }
    )


async def test_multiple_self_check_output_second_task_blocks_fake_main_stream(openai_api_key):
    chunks = await stream_with_fake_main(
        OPENAI_MULTI_SELF_CHECK_CONFIG,
        "blocked_data_leakage",
        [{"role": "user", "content": "hello"}],
    )

    assert_blocked_stream_error(chunks)
    assert normalize_stream_chunks(chunks) == snapshot(
        {
            "content": "blocked_data_leakage",
            "chunks": [
                "blocked_data_leakage",
                '{"error": {"message": "Blocked by self check output $task=check_data_leakage rails.", "type": "guardrails_violation", "param": "self check output $task=check_data_leakage", "code": "content_blocked"}}',
            ],
            "errors": [
                {
                    "error": {
                        "message": "Blocked by self check output $task=check_data_leakage rails.",
                        "type": "guardrails_violation",
                        "param": "self check output $task=check_data_leakage",
                        "code": "content_blocked",
                    }
                }
            ],
        }
    )


async def test_multiple_self_check_input_provider_error_raises(openai_api_key):
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
    chunks = await stream_with_fake_main(
        OPENAI_SELF_CHECK_CONFIG,
        "blocked_self_check_output",
        [{"role": "user", "content": "hello"}],
    )

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
