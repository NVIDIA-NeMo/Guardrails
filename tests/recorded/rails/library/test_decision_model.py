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

import pytest

from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.rails.library.configs import HUB_NEMOTRON_DECIDE_CONFIG, HUB_OPENJEV_CONFIG, OPENAI_DECISIONS_CONFIG
from tests.recorded.rails.library.helpers import check_iorails, check_rails

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


decision_cases = pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("I can check whether you qualify.", RailStatus.PASSED),
        ("You are guaranteed a full refund.", RailStatus.BLOCKED),
        ("Please send me your password.", RailStatus.BLOCKED),
    ],
    ids=["allow", "choice_block", "noul_block"],
)


@decision_cases
async def test_native_decision(openai_api_key, reply, expected, vcr):
    await assert_decision_check(OPENAI_DECISIONS_CONFIG, "/v1/decisions", reply, expected, vcr)


@pytest.mark.parametrize(
    ("config", "route"),
    [
        (HUB_NEMOTRON_DECIDE_CONFIG, "/v1/systemone"),
        (HUB_OPENJEV_CONFIG, "/openjev/v1/systemone"),
    ],
    ids=["nemotron_decide", "openjev_4b"],
)
@decision_cases
async def test_hub_system_one(nvidia_inference_key, config, route, reply, expected, vcr):
    await assert_decision_check(config, route, reply, expected, vcr)


async def assert_decision_check(config, route, reply, expected, vcr):
    result = await check_iorails(
        config,
        [
            {"role": "user", "content": "Can you help me with a refund?"},
            {"role": "assistant", "content": reply},
        ],
        (RailType.OUTPUT,),
    )
    assert result.status == expected
    if expected == RailStatus.BLOCKED:
        assert result.content == "I'm sorry, I can't respond to that."
    else:
        assert result.content == reply
    assert len(vcr.requests) == (1 if reply == "Please send me your password." else 2)
    assert all(request.path == route for request in vcr.requests)


async def test_native_decision_provider_error(openai_api_key, monkeypatch, vcr):
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-test-key")
    await assert_provider_error(OPENAI_DECISIONS_CONFIG, vcr)


@pytest.mark.parametrize(
    "config", [HUB_NEMOTRON_DECIDE_CONFIG, HUB_OPENJEV_CONFIG], ids=["nemotron_decide", "openjev_4b"]
)
async def test_hub_system_one_provider_error(nvidia_inference_key, config, monkeypatch, vcr):
    monkeypatch.setenv("NVIDIA_INFERENCE_KEY", "invalid-test-key")
    await assert_provider_error(config, vcr)


async def assert_provider_error(config, vcr):
    result = await check_iorails(
        config, [{"role": "assistant", "content": "I can check eligibility."}], (RailType.OUTPUT,)
    )
    assert result.status == RailStatus.BLOCKED
    assert result.content == "I'm sorry, an internal error has occurred."
    assert vcr.responses[0]["status"]["code"] == 401


@pytest.mark.vcr("test_native_decision[choice_block].yaml")
async def test_llmrails_reads_same_choice(openai_api_key, vcr):
    result = await check_rails(
        OPENAI_DECISIONS_CONFIG,
        [
            {"role": "user", "content": "Can you help me with a refund?"},
            {"role": "assistant", "content": "You are guaranteed a full refund."},
        ],
        rail_types=(RailType.OUTPUT,),
    )
    assert result.status == RailStatus.BLOCKED
    assert result.content == "I'm sorry, I can't respond to that."
    assert len(vcr.requests) == 2
