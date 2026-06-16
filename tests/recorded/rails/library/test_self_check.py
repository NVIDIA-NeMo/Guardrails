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

from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.assertions import (
    assert_blocked_stream_error,
    assert_rails_result,
)
from tests.recorded.normalization import normalize_rails_result, normalize_stream_chunks
from tests.recorded.rails.library.configs import OPENAI_SELF_CHECK_CONFIG
from tests.recorded.rails.library.helpers import check_rails, stream_with_fake_main
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


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
