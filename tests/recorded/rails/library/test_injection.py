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
from nemoguardrails.rails.llm.options import GenerationResponse, RailStatus, RailType
from tests.recorded.assertions import assert_generation_response, assert_rails_result
from tests.recorded.normalization import normalize_generation_response, normalize_rails_result
from tests.recorded.rails.library.configs import INJECTION_CONFIG, INJECTION_OMIT_CONFIG
from tests.recorded.rails.library.helpers import check_rails, generate_with_fake_main
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


async def test_injection_detection_rejects_xss_output():
    result = await check_rails(
        INJECTION_CONFIG,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello <script>alert('xss')</script> world"},
        ],
        rail_types=(RailType.OUTPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="injection detection")
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "injection detection",
            "content": "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of markdown_xss.",
        }
    )


async def test_injection_detection_omits_sql_output():
    result = await check_rails(
        INJECTION_OMIT_CONFIG,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "This is a SELECT * FROM users; -- malicious comment in text"},
        ],
        rail_types=(RailType.OUTPUT,),
    )

    assert_rails_result(
        result,
        status=RailStatus.MODIFIED,
        content="This is a  * FROM usersmalicious comment in text",
    )
    assert normalize_rails_result(result) == snapshot(
        {"status": "modified", "rail": None, "content": "This is a  * FROM usersmalicious comment in text"}
    )


async def test_injection_detection_omits_fake_main_generation():
    result = await generate_with_fake_main(
        INJECTION_OMIT_CONFIG,
        "This is a SELECT * FROM users; -- malicious comment in text",
        [{"role": "user", "content": "hello"}],
    )

    result = assert_generation_response(result)

    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "This is a  * FROM usersmalicious comment in text"}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {
                    "type": "output",
                    "name": "injection detection",
                    "decisions": ["execute injection_detection"],
                    "stop": False,
                },
            ],
            "llm_calls": [
                {
                    "task": "general",
                    "provider": "test",
                    "model": "fake",
                    "completion": "This is a SELECT * FROM users; -- malicious comment in text",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ],
        }
    )


async def test_injection_output_returns_exception_when_enabled():
    config = load_config(INJECTION_CONFIG)
    config.enable_rails_exceptions = True
    rails = LLMRails(config, verbose=False)

    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello <script>alert('xss')</script> world"},
        ],
        options={"rails": ["output"]},
    )

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    exception = result.response[0]
    assert exception["role"] == "exception"
    assert exception["content"]["type"] == "InjectionDetectionRailException"
    assert (
        exception["content"]["message"]
        == "Output not allowed. The output was blocked by the 'injection detection' flow."
    )
