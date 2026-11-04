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

import httpx
import pytest

from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.assertions import assert_rails_result
from tests.recorded.cassette import decode_body_json
from tests.recorded.normalization import normalize_rails_result
from tests.recorded.rails.library.configs import JAILBREAK_PROMPT, NIM_JAILBREAK_CONFIG, OVERSIZED_JAILBREAK_PROMPT
from tests.recorded.rails.library.helpers import check_iorails, check_rails
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


def _assert_detector_exchange(cassette, prompt, *, status, jailbreak=None):
    assert len(cassette.requests) == len(cassette.responses) == 1
    request = cassette.requests[0]
    assert request.method == "POST"
    assert request.uri == "https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect"
    assert decode_body_json(request.body) == {"input": prompt}
    response = cassette.responses[0]
    assert response["status"]["code"] == status
    if jailbreak is not None:
        assert decode_body_json(response["body"])["jailbreak"] is jailbreak
    if cassette.record_mode == "none":
        assert cassette.play_count == 1 and cassette.all_played


async def test_jailbreak_detection_input_allows_benign_prompt(nvidia_api_key, vcr):
    prompt = "what can you do?"
    result = await check_rails(
        NIM_JAILBREAK_CONFIG,
        [{"role": "user", "content": prompt}],
        rail_types=(RailType.INPUT,),
    )

    _assert_detector_exchange(vcr, prompt, status=200, jailbreak=False)
    assert_rails_result(result, status=RailStatus.PASSED)
    assert normalize_rails_result(result) == snapshot({"status": "passed", "rail": None, "content": "what can you do?"})


async def test_jailbreak_detection_input_blocks_jailbreak_prompt(nvidia_api_key, vcr):
    result = await check_rails(
        NIM_JAILBREAK_CONFIG,
        [{"role": "user", "content": JAILBREAK_PROMPT}],
        rail_types=(RailType.INPUT,),
    )

    _assert_detector_exchange(vcr, JAILBREAK_PROMPT, status=200, jailbreak=True)
    assert_rails_result(result, status=RailStatus.BLOCKED, rail="jailbreak detection model")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "jailbreak detection model", "content": "I'm sorry, I can't respond to that."}
    )


async def test_jailbreak_detection_input_invalid_key_fails_open(nvidia_api_key, monkeypatch, vcr):
    """Pin the existing fail-open policy against a real authorization error."""
    monkeypatch.setenv("NVIDIA_API_KEY", "invalid-recorded-replay")
    prompt = "What is the capital of France?"
    result = await check_rails(
        NIM_JAILBREAK_CONFIG,
        [{"role": "user", "content": prompt}],
        rail_types=(RailType.INPUT,),
    )

    _assert_detector_exchange(vcr, prompt, status=401)
    assert_rails_result(result, status=RailStatus.PASSED)
    assert normalize_rails_result(result) == snapshot(
        {"status": "passed", "rail": None, "content": "What is the capital of France?"}
    )


async def test_jailbreak_detection_blocks_oversized_input_despite_provider_allow(nvidia_api_key, vcr):
    """Record the hosted blind spot, then verify both engines reject before dispatch."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect",
            headers={"Authorization": f"Bearer {nvidia_api_key}"},
            json={"input": OVERSIZED_JAILBREAK_PROMPT},
        )
    assert response.status_code == 200
    assert response.json()["jailbreak"] is False

    outcomes = {}
    for engine, checker in (("llmrails", check_rails), ("iorails", check_iorails)):
        result = await checker(
            NIM_JAILBREAK_CONFIG,
            [{"role": "user", "content": OVERSIZED_JAILBREAK_PROMPT}],
            rail_types=(RailType.INPUT,),
        )
        assert_rails_result(result, status=RailStatus.BLOCKED, rail="jailbreak detection model")
        outcomes[engine] = normalize_rails_result(result)

    _assert_detector_exchange(vcr, OVERSIZED_JAILBREAK_PROMPT, status=200, jailbreak=False)
    assert outcomes == snapshot(
        {
            "llmrails": {
                "status": "blocked",
                "rail": "jailbreak detection model",
                "content": "I'm sorry, I can't respond to that.",
            },
            "iorails": {
                "status": "blocked",
                "rail": "jailbreak detection model",
                "content": "I'm sorry, I can't respond to that.",
            },
        }
    )
