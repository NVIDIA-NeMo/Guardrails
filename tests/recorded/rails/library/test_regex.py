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
from tests.recorded.assertions import assert_generation_response, assert_rails_result
from tests.recorded.normalization import normalize_generation_response, normalize_rails_result
from tests.recorded.rails.library.configs import REGEX_CONFIG
from tests.recorded.rails.library.helpers import check_rails, generate_with_fake_main
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


async def test_regex_input_blocks_secret():
    result = await check_rails(
        REGEX_CONFIG,
        [{"role": "user", "content": "my token is SECRET-1234"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="regex check input")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "regex check input", "content": "I'm sorry, I can't respond to that."}
    )


async def test_regex_output_blocks_secret():
    result = await check_rails(
        REGEX_CONFIG,
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "INTERNAL-SECRET"}],
        rail_types=(RailType.OUTPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="regex check output")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "regex check output", "content": "I'm sorry, I can't respond to that."}
    )


async def test_regex_output_blocks_fake_main_generation():
    result = await generate_with_fake_main(
        REGEX_CONFIG,
        "INTERNAL-SECRET",
        [{"role": "user", "content": "hello"}],
    )

    result = assert_generation_response(result)

    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}],
            "activated_rails": [
                {
                    "type": "input",
                    "name": "regex check input",
                    "decisions": ["execute detect_regex_pattern"],
                    "stop": False,
                },
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {
                    "type": "output",
                    "name": "regex check output",
                    "decisions": [
                        "execute detect_regex_pattern",
                        "refuse to respond",
                        "execute retrieve_relevant_chunks",
                        "execute generate_bot_message",
                        "stop",
                    ],
                    "stop": True,
                },
            ],
            "llm_calls": [
                {
                    "task": "general",
                    "provider": "test",
                    "model": "fake",
                    "completion": "INTERNAL-SECRET",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ],
        }
    )
