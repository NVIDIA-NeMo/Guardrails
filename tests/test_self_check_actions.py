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

from types import SimpleNamespace
from typing import Any, cast

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.factchecking.align_score import actions as alignscore_actions
from nemoguardrails.library.factchecking.align_score.actions import alignscore_check_facts
from nemoguardrails.library.self_check.facts.actions import _fact_check_outcome, self_check_facts
from nemoguardrails.library.self_check.input_check.actions import self_check_input
from nemoguardrails.library.self_check.output_check.actions import self_check_output
from nemoguardrails.llm.taskmanager import LLMTaskManager
from tests.utils import FakeLLMModel


class _SelfCheckTaskManager:
    def __init__(self, parsed: list[bool] | None = None):
        self.parsed = parsed or [True]
        self.config = SimpleNamespace(prompts=[])

    def render_task_prompt(self, task: Any, context: dict[str, Any]) -> str:
        return "prompt"

    def get_stop_tokens(self, task: Any) -> list[str]:
        return []

    def get_max_tokens(self, task: Any) -> None:
        return None

    def has_output_parser(self, task: Any) -> bool:
        return True

    def parse_task_output(self, task: Any, output: str, forced_output_parser: str | None = None) -> list[bool]:
        return self.parsed


def _config() -> RailsConfig:
    return cast(RailsConfig, SimpleNamespace(lowest_temperature=0.0))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        ([True], RailOutcome.allow()),
        ([False], RailOutcome.block()),
    ],
)
async def test_self_check_output_returns_rail_outcome(parsed, expected):
    task_manager = cast(LLMTaskManager, _SelfCheckTaskManager(parsed))

    outcome = await self_check_output(
        llms={},
        llm_task_manager=task_manager,
        context={"user_message": "hello", "bot_message": "answer"},
        llm=FakeLLMModel(responses=["parsed by test manager"]),
        config=_config(),
    )

    assert outcome == expected


@pytest.mark.asyncio
async def test_self_check_input_block_preserves_mask_event():
    task_manager = cast(LLMTaskManager, _SelfCheckTaskManager([False]))

    result = await self_check_input(
        llms={},
        llm_task_manager=task_manager,
        context={"user_message": "blocked"},
        llm=FakeLLMModel(responses=["parsed by test manager"]),
        config=_config(),
    )

    assert isinstance(result, ActionResult)
    assert result.return_value == RailOutcome.block()
    assert result.events is not None
    assert result.events[0]["type"] == "mask_prev_user_message"


@pytest.mark.parametrize(
    ("accuracy", "expected"),
    [
        (0.49, RailOutcome.block(accuracy=0.49)),
        (0.5, RailOutcome.allow(accuracy=0.5)),
        (0.51, RailOutcome.allow(accuracy=0.51)),
    ],
)
def test_fact_check_outcome_pins_threshold(accuracy, expected):
    assert _fact_check_outcome(accuracy) == expected


@pytest.mark.asyncio
async def test_self_check_facts_without_evidence_allows():
    task_manager = cast(LLMTaskManager, _SelfCheckTaskManager())

    outcome = await self_check_facts(
        llm_task_manager=task_manager,
        context={"relevant_chunks": [], "bot_message": "answer"},
        llm=FakeLLMModel(responses=[]),
        config=_config(),
    )

    assert outcome == RailOutcome.allow(accuracy=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.49, RailOutcome.block(accuracy=0.49)),
        (0.5, RailOutcome.allow(accuracy=0.5)),
    ],
)
async def test_alignscore_check_facts_returns_rail_outcome(monkeypatch, score, expected):
    async def fake_alignscore_request(url, evidence, response):
        return score

    task_manager = cast(
        LLMTaskManager,
        SimpleNamespace(
            config=SimpleNamespace(
                rails=SimpleNamespace(
                    config=SimpleNamespace(
                        fact_checking=SimpleNamespace(
                            fallback_to_self_check=False,
                            parameters={"endpoint": "http://localhost:5000/alignscore_base"},
                        )
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(alignscore_actions, "alignscore_request", fake_alignscore_request)

    outcome = await alignscore_check_facts(
        llm_task_manager=task_manager,
        context={"relevant_chunks": ["evidence"], "bot_message": "answer"},
        llm=FakeLLMModel(responses=[]),
        config=_config(),
    )

    assert outcome == expected
