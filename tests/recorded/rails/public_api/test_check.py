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

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.assertions import assert_rails_result
from tests.recorded.normalization import normalize_rails_result
from tests.recorded.rails.public_api.configs import (
    INPUT_OUTPUT_RAILS_CONFIG,
    INPUT_RAILS_CONFIG,
    NEMOGUARDS_FULL_CONFIG,
    OUTPUT_RAILS_CONFIG,
    PARALLEL_RAILS_CONFIG,
)
from tests.recorded.rails_config import RailsConfigSource, load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded]


@dataclass(frozen=True)
class CheckScenario:
    id: str
    config: RailsConfigSource
    messages: list[dict[str, Any]]
    status: RailStatus
    rail: str | None = None
    content: str | None = None
    rail_types: tuple[RailType, ...] | None = None


SCENARIOS = [
    CheckScenario(
        id="input-allowed",
        config=INPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "allowed input"}],
        status=RailStatus.PASSED,
    ),
    CheckScenario(
        id="input-blocked",
        config=INPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "block input"}],
        status=RailStatus.BLOCKED,
        rail="input rail",
    ),
    CheckScenario(
        id="input-modified",
        config=INPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "modify input"}],
        status=RailStatus.MODIFIED,
        content="modified input",
    ),
    CheckScenario(
        id="output-allowed",
        config=OUTPUT_RAILS_CONFIG,
        messages=[{"role": "assistant", "content": "allowed output"}],
        status=RailStatus.PASSED,
    ),
    CheckScenario(
        id="output-blocked",
        config=OUTPUT_RAILS_CONFIG,
        messages=[{"role": "assistant", "content": "block output"}],
        status=RailStatus.BLOCKED,
        rail="output rail",
    ),
    CheckScenario(
        id="output-modified",
        config=OUTPUT_RAILS_CONFIG,
        messages=[{"role": "assistant", "content": "modify output"}],
        status=RailStatus.MODIFIED,
        content="modified output",
    ),
    CheckScenario(
        id="auto-input",
        config=INPUT_OUTPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "block input"}],
        status=RailStatus.BLOCKED,
        rail="input rail",
    ),
    CheckScenario(
        id="auto-output",
        config=INPUT_OUTPUT_RAILS_CONFIG,
        messages=[{"role": "assistant", "content": "block output"}],
        status=RailStatus.BLOCKED,
        rail="output rail",
    ),
    CheckScenario(
        id="auto-both",
        config=INPUT_OUTPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "allowed input"}, {"role": "assistant", "content": "block output"}],
        status=RailStatus.BLOCKED,
        rail="output rail",
    ),
    CheckScenario(
        id="explicit-input",
        config=INPUT_OUTPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "allowed input"}, {"role": "assistant", "content": "block output"}],
        status=RailStatus.PASSED,
        rail_types=(RailType.INPUT,),
    ),
    CheckScenario(
        id="explicit-output",
        config=INPUT_OUTPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "block input"}, {"role": "assistant", "content": "allowed output"}],
        status=RailStatus.PASSED,
        rail_types=(RailType.OUTPUT,),
    ),
    CheckScenario(
        id="explicit-both",
        config=INPUT_OUTPUT_RAILS_CONFIG,
        messages=[{"role": "user", "content": "block input"}, {"role": "assistant", "content": "allowed output"}],
        status=RailStatus.BLOCKED,
        rail="input rail",
        rail_types=(RailType.INPUT, RailType.OUTPUT),
    ),
    CheckScenario(
        id="parallel-pass",
        config=PARALLEL_RAILS_CONFIG,
        messages=[{"role": "user", "content": "allowed parallel"}],
        status=RailStatus.PASSED,
    ),
    CheckScenario(
        id="parallel-block",
        config=PARALLEL_RAILS_CONFIG,
        messages=[{"role": "user", "content": "block parallel"}],
        status=RailStatus.BLOCKED,
        rail="parallel input rail",
    ),
]


def _rail_types(scenario: CheckScenario) -> list[RailType] | None:
    return list(scenario.rail_types) if scenario.rail_types is not None else None


def _assert_check_contract_results(results: dict[str, dict[str, Any]]) -> None:
    assert results == snapshot(
        {
            "input-allowed": {"status": "passed", "rail": None, "content": "allowed input"},
            "input-blocked": {
                "status": "blocked",
                "rail": "input rail",
                "content": "I'm sorry, I can't respond to that.",
            },
            "input-modified": {"status": "modified", "rail": None, "content": "modified input"},
            "output-allowed": {"status": "passed", "rail": None, "content": "allowed output"},
            "output-blocked": {
                "status": "blocked",
                "rail": "output rail",
                "content": "I'm sorry, I can't respond to that.",
            },
            "output-modified": {"status": "modified", "rail": None, "content": "modified output"},
            "auto-input": {"status": "blocked", "rail": "input rail", "content": "I'm sorry, I can't respond to that."},
            "auto-output": {
                "status": "blocked",
                "rail": "output rail",
                "content": "I'm sorry, I can't respond to that.",
            },
            "auto-both": {"status": "blocked", "rail": "output rail", "content": "I'm sorry, I can't respond to that."},
            "explicit-input": {"status": "passed", "rail": None, "content": "allowed input"},
            "explicit-output": {"status": "passed", "rail": None, "content": "allowed output"},
            "explicit-both": {
                "status": "blocked",
                "rail": "input rail",
                "content": "I'm sorry, I can't respond to that.",
            },
            "parallel-pass": {"status": "passed", "rail": None, "content": "allowed parallel"},
            "parallel-block": {
                "status": "blocked",
                "rail": "parallel input rail",
                "content": "Parallel input blocked.",
            },
        }
    )


def test_check_sync_public_contracts():
    results = {}
    for scenario in SCENARIOS:
        rails = LLMRails(load_config(scenario.config), verbose=False)

        result = rails.check(scenario.messages, rail_types=_rail_types(scenario))

        assert_rails_result(result, status=scenario.status, content=scenario.content, rail=scenario.rail)
        results[scenario.id] = normalize_rails_result(result)

    _assert_check_contract_results(results)


@pytest.mark.asyncio
async def test_check_async_public_contracts():
    results = {}
    for scenario in SCENARIOS:
        rails = LLMRails(load_config(scenario.config), verbose=False)

        result = await rails.check_async(scenario.messages, rail_types=_rail_types(scenario))

        assert_rails_result(result, status=scenario.status, content=scenario.content, rail=scenario.rail)
        results[scenario.id] = normalize_rails_result(result)

    _assert_check_contract_results(results)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_nemoguards_full_check_async(nvidia_api_key):
    rails = LLMRails(load_config(NEMOGUARDS_FULL_CONFIG), verbose=False)

    result = await rails.check_async([{"role": "user", "content": "what can you do?"}])

    assert_rails_result(result, status=RailStatus.PASSED)
    assert normalize_rails_result(result) == snapshot({"status": "passed", "rail": None, "content": "what can you do?"})


@pytest.mark.asyncio
async def test_check_async_empty_messages_passes():
    rails = LLMRails(load_config(INPUT_RAILS_CONFIG), verbose=False)

    result = await rails.check_async([])

    assert_rails_result(result, status=RailStatus.PASSED, content="")
    assert normalize_rails_result(result) == snapshot({"status": "passed", "rail": None, "content": ""})
