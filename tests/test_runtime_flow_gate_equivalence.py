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

import textwrap
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
class FlowEquivalenceCase:
    case_id: str
    yaml_content: str
    action_name: str
    raw_return: Any
    expected_observable: ObservableOutcome
    expected_decision: FlowDecision
    enable_rails_exceptions: bool = False


SELF_CHECK_OUTPUT_YAML = textwrap.dedent(
    """
    models: []
    rails:
      output:
        flows:
          - self check output
    prompts:
      - task: self_check_output
        content: ...
    """
)


CONTENT_SAFETY_OUTPUT_YAML = textwrap.dedent(
    """
    models:
      - type: content_safety
        engine: openai
        model: placeholder
    rails:
      output:
        flows:
          - content safety check output $model=content_safety
    prompts:
      - task: content_safety_check_output $model=content_safety
        content: ...
        output_parser: nemoguard_parse_prompt_safety
    """
)


CONTENT_SAFETY_INPUT_YAML = textwrap.dedent(
    """
    models:
      - type: content_safety
        engine: openai
        model: placeholder
    rails:
      input:
        flows:
          - content safety check input $model=content_safety
    prompts:
      - task: content_safety_check_input $model=content_safety
        content: ...
        output_parser: nemoguard_parse_prompt_safety
    """
)


TOPIC_SAFETY_INPUT_YAML = textwrap.dedent(
    """
    models:
      - type: topic_control
        engine: openai
        model: placeholder
    rails:
      input:
        flows:
          - topic safety check input $model=topic_control
    prompts:
      - task: topic_safety_check_input $model=topic_control
        content: ...
    """
)


JAILBREAK_HEURISTICS_INPUT_YAML = textwrap.dedent(
    """
    models: []
    rails:
      config:
        jailbreak_detection:
          server_endpoint: http://localhost:9999
      input:
        flows:
          - jailbreak detection heuristics
    """
)


JAILBREAK_MODEL_INPUT_YAML = textwrap.dedent(
    """
    models: []
    rails:
      config:
        jailbreak_detection:
          server_endpoint: http://localhost:9999
      input:
        flows:
          - jailbreak detection model
    """
)


def _case(
    case_id: str,
    yaml_content: str,
    action_name: str,
    raw_return: Any,
    expected_observable: ObservableOutcome,
    expected_decision: FlowDecision,
    *,
    enable_rails_exceptions: bool = False,
) -> FlowEquivalenceCase:
    return FlowEquivalenceCase(
        case_id=case_id,
        yaml_content=yaml_content,
        action_name=action_name,
        raw_return=raw_return,
        expected_observable=expected_observable,
        expected_decision=expected_decision,
        enable_rails_exceptions=enable_rails_exceptions,
    )


def _rail_outcome_cases(
    prefix: str,
    yaml_content: str,
    action_name: str,
    *,
    allow_return: RailOutcome | None = None,
    block_return: RailOutcome | None = None,
    include_exception_case: bool = False,
) -> list[FlowEquivalenceCase]:
    allow_return = allow_return or RailOutcome.allow()
    block_return = block_return or RailOutcome.block()
    cases = [
        _case(
            f"{prefix}_allows_outcome_allow",
            yaml_content,
            action_name,
            allow_return,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
        ),
        _case(
            f"{prefix}_blocks_outcome_block",
            yaml_content,
            action_name,
            block_return,
            ObservableOutcome.REFUSAL,
            FlowDecision.BLOCK,
        ),
    ]
    if include_exception_case:
        cases.append(
            _case(
                f"{prefix}_blocks_outcome_block_exception",
                yaml_content,
                action_name,
                block_return,
                ObservableOutcome.EXCEPTION,
                FlowDecision.BLOCK,
                enable_rails_exceptions=True,
            )
        )
    return cases


def _legacy_cases(
    prefix: str,
    yaml_content: str,
    action_name: str,
    *,
    allow_return: Any,
    block_return: Any,
    include_exception_case: bool = False,
) -> list[FlowEquivalenceCase]:
    cases = [
        _case(
            f"{prefix}_allows_legacy_return",
            yaml_content,
            action_name,
            allow_return,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
        ),
        _case(
            f"{prefix}_blocks_legacy_return",
            yaml_content,
            action_name,
            block_return,
            ObservableOutcome.REFUSAL,
            FlowDecision.BLOCK,
        ),
    ]
    if include_exception_case:
        cases.append(
            _case(
                f"{prefix}_blocks_legacy_return_exception",
                yaml_content,
                action_name,
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
        SELF_CHECK_OUTPUT_YAML,
        "self_check_output",
        True,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "self_check_output_blocks_false",
        SELF_CHECK_OUTPUT_YAML,
        "self_check_output",
        False,
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    *_legacy_cases(
        "content_safety_input",
        CONTENT_SAFETY_INPUT_YAML,
        "content_safety_check_input",
        allow_return={"allowed": True, "policy_violations": []},
        block_return={"allowed": False, "policy_violations": ["violence"]},
        include_exception_case=True,
    ),
    *_legacy_cases(
        "content_safety_output",
        CONTENT_SAFETY_OUTPUT_YAML,
        "content_safety_check_output",
        allow_return={"allowed": True, "policy_violations": []},
        block_return={"allowed": False, "policy_violations": ["violence"]},
        include_exception_case=True,
    ),
    *_legacy_cases(
        "topic_safety_input",
        TOPIC_SAFETY_INPUT_YAML,
        "topic_safety_check_input",
        allow_return={"on_topic": True},
        block_return={"on_topic": False},
        include_exception_case=True,
    ),
    *_legacy_cases(
        "jailbreak_heuristics_input",
        JAILBREAK_HEURISTICS_INPUT_YAML,
        "jailbreak_detection_heuristics",
        allow_return=False,
        block_return=True,
        include_exception_case=True,
    ),
    *_legacy_cases(
        "jailbreak_model_input",
        JAILBREAK_MODEL_INPUT_YAML,
        "jailbreak_detection_model",
        allow_return=False,
        block_return=True,
        include_exception_case=True,
    ),
]


def _yaml_with_exception_mode(case: FlowEquivalenceCase) -> str:
    return case.yaml_content + f"\nenable_rails_exceptions: {str(case.enable_rails_exceptions).lower()}\n"


def _run_flow(case: FlowEquivalenceCase) -> dict[str, Any]:
    config = RailsConfig.from_content(yaml_content=_yaml_with_exception_mode(case))
    chat = TestChat(config, llm_completions=[NORMAL_OUTPUT])

    async def stub_action(**kwargs):
        return case.raw_return

    chat.app.register_action(stub_action, case.action_name)
    return chat.app.generate(messages=[{"role": "user", "content": "hello"}])


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
    elif action_name in {"content_safety_check_input", "content_safety_check_output"}:
        blocked = not raw_return["allowed"]
    elif action_name == "topic_safety_check_input":
        blocked = not raw_return["on_topic"]
    elif action_name in {"jailbreak_detection_heuristics", "jailbreak_detection_model"}:
        blocked = raw_return
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
    assert _outcome_decision(case.raw_return, case.action_name) is case.expected_decision
