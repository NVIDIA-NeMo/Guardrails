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

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome
from tests.utils import TestChat

NORMAL_OUTPUT = "NORMAL OUTPUT"
REFUSAL = "I'm sorry, I can't respond to that."


class ObservableOutcome(Enum):
    ALLOW = "allow"
    REFUSAL = "refusal"
    EXCEPTION = "exception"
    TRANSFORM = "transform"


class FlowDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


@dataclass(frozen=True)
class RailSpec:
    name: str
    flow: str
    direction: str
    action: str
    model_type: str | None = None
    task: str | None = None
    output_parser: str | None = None
    rails_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class FlowEquivalenceCase:
    case_id: str
    spec: RailSpec
    raw_return: Any
    expected_observable: ObservableOutcome
    expected_decision: FlowDecision
    enable_rails_exceptions: bool = False


JAILBREAK_RAILS_CONFIG = {"jailbreak_detection": {"server_endpoint": "http://localhost:9999"}}

SELF_CHECK_OUTPUT = RailSpec(
    name="self_check_output",
    flow="self check output",
    direction="output",
    action="self_check_output",
    task="self_check_output",
)

CONTENT_SAFETY_OUTPUT = RailSpec(
    name="content_safety_output",
    flow="content safety check output $model=content_safety",
    direction="output",
    action="content_safety_check_output",
    model_type="content_safety",
    task="content_safety_check_output $model=content_safety",
    output_parser="nemoguard_parse_prompt_safety",
)

CONTENT_SAFETY_INPUT = RailSpec(
    name="content_safety_input",
    flow="content safety check input $model=content_safety",
    direction="input",
    action="content_safety_check_input",
    model_type="content_safety",
    task="content_safety_check_input $model=content_safety",
    output_parser="nemoguard_parse_prompt_safety",
)

TOPIC_SAFETY_INPUT = RailSpec(
    name="topic_safety_input",
    flow="topic safety check input $model=topic_control",
    direction="input",
    action="topic_safety_check_input",
    model_type="topic_control",
    task="topic_safety_check_input $model=topic_control",
)

JAILBREAK_HEURISTICS_INPUT = RailSpec(
    name="jailbreak_heuristics_input",
    flow="jailbreak detection heuristics",
    direction="input",
    action="jailbreak_detection_heuristics",
    rails_config=JAILBREAK_RAILS_CONFIG,
)

JAILBREAK_MODEL_INPUT = RailSpec(
    name="jailbreak_model_input",
    flow="jailbreak detection model",
    direction="input",
    action="jailbreak_detection_model",
    rails_config=JAILBREAK_RAILS_CONFIG,
)


def _case(
    case_id: str,
    spec: RailSpec,
    raw_return: Any,
    expected_observable: ObservableOutcome,
    expected_decision: FlowDecision,
    *,
    enable_rails_exceptions: bool = False,
) -> FlowEquivalenceCase:
    return FlowEquivalenceCase(
        case_id=case_id,
        spec=spec,
        raw_return=raw_return,
        expected_observable=expected_observable,
        expected_decision=expected_decision,
        enable_rails_exceptions=enable_rails_exceptions,
    )


def _rail_outcome_cases(
    spec: RailSpec,
    *,
    allow_return: RailOutcome | None = None,
    block_return: RailOutcome | None = None,
    include_exception_case: bool = False,
) -> list[FlowEquivalenceCase]:
    allow_return = allow_return or RailOutcome.allow()
    block_return = block_return or RailOutcome.block()
    cases = [
        _case(
            f"{spec.name}_allows_outcome_allow",
            spec,
            allow_return,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
        ),
        _case(
            f"{spec.name}_blocks_outcome_block",
            spec,
            block_return,
            ObservableOutcome.REFUSAL,
            FlowDecision.BLOCK,
        ),
    ]
    if include_exception_case:
        cases.append(
            _case(
                f"{spec.name}_blocks_outcome_block_exception",
                spec,
                block_return,
                ObservableOutcome.EXCEPTION,
                FlowDecision.BLOCK,
                enable_rails_exceptions=True,
            )
        )
    return cases


FIXTURES = [
    _case(
        "self_check_output_allows_true",
        SELF_CHECK_OUTPUT,
        True,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "self_check_output_blocks_false",
        SELF_CHECK_OUTPUT,
        False,
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    *_rail_outcome_cases(
        CONTENT_SAFETY_INPUT,
        allow_return=RailOutcome.allow(policy_violations=[]),
        block_return=RailOutcome.block(policy_violations=["violence"]),
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        CONTENT_SAFETY_OUTPUT,
        allow_return=RailOutcome.allow(policy_violations=[]),
        block_return=RailOutcome.block(policy_violations=["violence"]),
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        TOPIC_SAFETY_INPUT,
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        JAILBREAK_HEURISTICS_INPUT,
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        JAILBREAK_MODEL_INPUT,
        include_exception_case=True,
    ),
]


def _build_config(spec: RailSpec, *, enable_rails_exceptions: bool) -> dict[str, Any]:
    models = []
    if spec.model_type:
        models.append({"type": spec.model_type, "engine": "openai", "model": "placeholder"})

    rails: dict[str, Any] = {spec.direction: {"flows": [spec.flow]}}
    if spec.rails_config:
        rails["config"] = spec.rails_config

    config: dict[str, Any] = {
        "models": models,
        "rails": rails,
        "enable_rails_exceptions": enable_rails_exceptions,
    }

    if spec.task:
        prompt = {"task": spec.task, "content": "..."}
        if spec.output_parser:
            prompt["output_parser"] = spec.output_parser
        config["prompts"] = [prompt]

    return config


def _run_flow(case: FlowEquivalenceCase) -> dict[str, Any]:
    config = RailsConfig.from_content(
        config=_build_config(
            case.spec,
            enable_rails_exceptions=case.enable_rails_exceptions,
        )
    )
    chat = TestChat(config, llm_completions=[NORMAL_OUTPUT])

    async def stub_action(**kwargs):
        return case.raw_return

    chat.app.register_action(stub_action, case.spec.action)
    response = chat.app.generate(messages=[{"role": "user", "content": "hello"}])
    if not isinstance(response, dict):
        raise AssertionError(f"Unexpected runtime response type: {response!r}")
    return response


def _classify_response(response: dict[str, Any]) -> ObservableOutcome:
    if response == {"role": "assistant", "content": NORMAL_OUTPUT}:
        return ObservableOutcome.ALLOW
    if response == {"role": "assistant", "content": REFUSAL}:
        return ObservableOutcome.REFUSAL
    if response.get("role") == "exception":
        return ObservableOutcome.EXCEPTION
    if response.get("role") == "assistant":
        return ObservableOutcome.TRANSFORM

    raise AssertionError(f"Unexpected runtime response: {response!r}")


def _decision_from_observable(observable: ObservableOutcome) -> FlowDecision:
    if observable is ObservableOutcome.ALLOW:
        return FlowDecision.ALLOW
    if observable in (ObservableOutcome.REFUSAL, ObservableOutcome.EXCEPTION):
        return FlowDecision.BLOCK
    return FlowDecision.TRANSFORM


def _outcome_decision(raw_return: Any, action_name: str) -> FlowDecision:
    if isinstance(raw_return, RailOutcome):
        blocked = raw_return.is_blocked
    elif isinstance(raw_return, bool):
        blocked = not raw_return
    elif isinstance(raw_return, (int, float)):
        blocked = raw_return < 0.5
    else:
        blocked = False

    return FlowDecision.BLOCK if blocked else FlowDecision.ALLOW


@pytest.mark.parametrize("case", FIXTURES, ids=[case.case_id for case in FIXTURES])
def test_runtime_flow_gate_matches_rail_outcome(case: FlowEquivalenceCase):
    observable = _classify_response(_run_flow(case))

    assert observable is case.expected_observable
    assert _decision_from_observable(observable) is case.expected_decision
    assert _outcome_decision(case.raw_return, case.spec.action) is case.expected_decision
