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

import dataclasses
from typing import Any, cast

import pytest

from nemoguardrails.actions.rail_outcome import (
    RailDecision,
    RailOutcome,
    TransformSpec,
    TransformTarget,
    require_rail_outcome,
)


def test_allow_is_not_blocked():
    outcome = RailOutcome.allow()
    assert outcome.decision is RailDecision.ALLOW
    assert outcome.is_blocked is False
    assert outcome.is_transform is False
    assert outcome.transforms == ()


def test_allow_carries_metadata():
    outcome = RailOutcome.allow(policy_violations=[], score=0.1)
    assert outcome.metadata == {"policy_violations": [], "score": 0.1}


def test_block_is_blocked_with_reason_and_metadata():
    outcome = RailOutcome.block(reason="unsafe", policy_violations=["S1: Violence"])
    assert outcome.decision is RailDecision.BLOCK
    assert outcome.is_blocked is True
    assert outcome.is_transform is False
    assert outcome.reason == "unsafe"
    assert outcome.metadata == {"policy_violations": ["S1: Violence"]}
    assert outcome.transforms == ()


def test_transform_targets_variables():
    outcome = RailOutcome.transform(
        [
            (TransformTarget.USER_MESSAGE, "masked input"),
            (TransformTarget.BOT_MESSAGE, "masked output"),
        ]
    )
    assert outcome.decision is RailDecision.TRANSFORM
    assert outcome.is_blocked is False
    assert outcome.is_transform is True
    assert outcome.transforms == (
        TransformSpec(target=TransformTarget.USER_MESSAGE, text="masked input"),
        TransformSpec(target=TransformTarget.BOT_MESSAGE, text="masked output"),
    )
    assert outcome.transform_text == {
        "user_message": "masked input",
        "bot_message": "masked output",
    }


def test_transform_payload_required_for_transform_decision():
    with pytest.raises(ValueError, match="transforms must be non-empty if and only if decision is TRANSFORM"):
        RailOutcome(decision=RailDecision.TRANSFORM)


def test_transform_payload_forbidden_for_block_decision():
    with pytest.raises(ValueError, match="transforms must be non-empty if and only if decision is TRANSFORM"):
        RailOutcome(
            decision=RailDecision.BLOCK,
            transforms=(TransformSpec(target=TransformTarget.USER_MESSAGE, text="x"),),
        )


def test_outcome_is_immutable():
    outcome = RailOutcome.allow()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, outcome).decision = RailDecision.BLOCK


def test_require_rail_outcome_accepts_outcome():
    outcome = RailOutcome.allow()
    assert require_rail_outcome(outcome) is outcome


@pytest.mark.parametrize("result", [None, False, {}, (RailOutcome.allow(), "success")])
def test_require_rail_outcome_rejects_other_results(result):
    with pytest.raises(TypeError, match="Output rail action must return RailOutcome"):
        require_rail_outcome(result)
