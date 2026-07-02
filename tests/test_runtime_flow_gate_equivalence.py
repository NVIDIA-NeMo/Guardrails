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
from typing import Any, Callable

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.content_safety.actions import content_safety_check_output
from nemoguardrails.library.self_check.output_check.actions import self_check_output
from tests.utils import TestChat

NORMAL_OUTPUT = "NORMAL OUTPUT"
REFUSAL = "I'm sorry, I can't respond to that."


class RuntimeVerdict(Enum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True)
class RuntimeFlowFixture:
    name: str
    yaml_content: str
    action_name: str
    action_func: Callable[..., Any]
    raw_return: Any
    expected: RuntimeVerdict


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


FIXTURES = [
    RuntimeFlowFixture(
        name="self_check_output_allows_true",
        yaml_content=SELF_CHECK_OUTPUT_YAML,
        action_name="self_check_output",
        action_func=self_check_output,
        raw_return=True,
        expected=RuntimeVerdict.ALLOW,
    ),
    RuntimeFlowFixture(
        name="self_check_output_blocks_false",
        yaml_content=SELF_CHECK_OUTPUT_YAML,
        action_name="self_check_output",
        action_func=self_check_output,
        raw_return=False,
        expected=RuntimeVerdict.BLOCK,
    ),
    RuntimeFlowFixture(
        name="content_safety_output_allows_allowed_true",
        yaml_content=CONTENT_SAFETY_OUTPUT_YAML,
        action_name="content_safety_check_output",
        action_func=content_safety_check_output,
        raw_return={"allowed": True, "policy_violations": []},
        expected=RuntimeVerdict.ALLOW,
    ),
    RuntimeFlowFixture(
        name="content_safety_output_blocks_allowed_false",
        yaml_content=CONTENT_SAFETY_OUTPUT_YAML,
        action_name="content_safety_check_output",
        action_func=content_safety_check_output,
        raw_return={"allowed": False, "policy_violations": ["violence"]},
        expected=RuntimeVerdict.BLOCK,
    ),
]


def _runtime_verdict(fixture: RuntimeFlowFixture) -> RuntimeVerdict:
    config = RailsConfig.from_content(yaml_content=fixture.yaml_content)
    chat = TestChat(config, llm_completions=[NORMAL_OUTPUT])

    async def stub_action(**kwargs):
        return fixture.raw_return

    chat.app.register_action(stub_action, fixture.action_name)
    response = chat.app.generate(messages=[{"role": "user", "content": "hello"}])

    if response == {"role": "assistant", "content": NORMAL_OUTPUT}:
        return RuntimeVerdict.ALLOW
    if response == {"role": "assistant", "content": REFUSAL}:
        return RuntimeVerdict.BLOCK

    raise AssertionError(f"Unexpected runtime response: {response!r}")


def _outcome_verdict(raw_return: Any, action_name: str) -> RuntimeVerdict:
    if isinstance(raw_return, RailOutcome):
        blocked = raw_return.is_blocked
    elif action_name == "content_safety_check_output":
        blocked = not raw_return["allowed"]
    elif isinstance(raw_return, bool):
        blocked = not raw_return
    elif isinstance(raw_return, (int, float)):
        blocked = raw_return < 0.5
    else:
        blocked = False

    return RuntimeVerdict.BLOCK if blocked else RuntimeVerdict.ALLOW


def _mapping_verdict(raw_return: Any, action_func: Callable[..., Any]) -> RuntimeVerdict:
    return RuntimeVerdict.BLOCK if is_output_blocked(raw_return, action_func) else RuntimeVerdict.ALLOW


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture.name for fixture in FIXTURES])
def test_runtime_flow_gate_matches_output_mapping_and_rail_outcome(fixture: RuntimeFlowFixture):
    assert _runtime_verdict(fixture) is fixture.expected
    assert _mapping_verdict(fixture.raw_return, fixture.action_func) is fixture.expected
    assert _outcome_verdict(fixture.raw_return, fixture.action_name) is fixture.expected
