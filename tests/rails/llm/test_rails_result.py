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

from nemoguardrails.actions.rail_outcome import RailDecision, RailOutcome, TransformTarget
from nemoguardrails.rails.llm.llmrails import _build_rails_result
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    ExecutedAction,
    GenerationLog,
    GenerationResponse,
    RailsResult,
    RailStatus,
)


def _rail(name, outcome, *, stop=False):
    return ActivatedRail(
        type="output",
        name=name,
        stop=stop,
        executed_actions=[ExecutedAction(action_name=f"{name}_action", return_value=outcome)],
    )


def _response(final_content, rails):
    return GenerationResponse(
        response=[{"role": "assistant", "content": final_content}],
        log=GenerationLog(activated_rails=rails),
    )


def test_every_modifier_is_attributed_in_order():
    mask_pii = RailOutcome.transform(
        [(TransformTarget.BOT_MESSAGE, "redacted")],
        reason="masked 1 email",
        metadata={"entities": ["EMAIL"]},
    )
    strip_links = RailOutcome.transform([(TransformTarget.BOT_MESSAGE, "redacted2")], reason="stripped 2 urls")
    result = _build_rails_result(
        _response("redacted2", [_rail("mask pii", mask_pii), _rail("strip links", strip_links)]),
        original_content="original",
    )

    assert result.status is RailStatus.MODIFIED
    assert result.modified is True
    assert result.modified_by == ["mask pii", "strip links"]
    assert result.blocking_rails == []
    assert result.evaluations[0].decision is RailDecision.TRANSFORM
    assert result.evaluations[0].reason == "masked 1 email"


def test_multiple_blockers_and_prior_modifier_all_recorded():
    mask = RailOutcome.transform([(TransformTarget.BOT_MESSAGE, "redacted")], reason="masked pii")
    unsafe = RailOutcome.block(reason="unsafe", metadata={"categories": ["violence"]})
    jailbreak = RailOutcome.block(reason="jailbreak")
    result = _build_rails_result(
        _response(
            "I can't help with that.",
            [
                _rail("mask pii", mask),
                _rail("content safety", unsafe, stop=True),
                _rail("jailbreak", jailbreak, stop=True),
            ],
        ),
        original_content="original",
    )

    assert result.status is RailStatus.BLOCKED
    assert result.blocked is True
    assert result.blocking_rails == ["content safety", "jailbreak"]
    assert result.blocked_by == "content safety"
    assert result.reason == "unsafe"
    assert result.modified_by == ["mask pii"]


def test_rail_field_is_deprecated_but_still_works():
    result = _build_rails_result(
        _response("blocked", [_rail("content safety", RailOutcome.block(reason="unsafe"), stop=True)]),
        original_content="original",
    )

    with pytest.warns(DeprecationWarning, match="blocked_by"):
        legacy_rail = result.rail

    assert legacy_rail == result.blocked_by == "content safety"


def test_all_allow_is_passed():
    result = _build_rails_result(
        _response("original", [_rail("topic control", RailOutcome.allow(reason="ok"))]),
        original_content="original",
    )

    assert result.status is RailStatus.PASSED
    assert result.passed is True
    assert result.modified_by == []
    assert result.blocking_rails == []
    assert result.reason is None


def test_legacy_stop_without_rail_outcome_is_synthesized_as_block():
    stop_only = ActivatedRail(type="output", name="legacy self check", stop=True, executed_actions=[])
    result = _build_rails_result(_response("blocked", [stop_only]), original_content="original")

    assert result.status is RailStatus.BLOCKED
    assert result.blocking_rails == ["legacy self check"]
    assert result.evaluations[0].decision is RailDecision.BLOCK


def test_result_serializes_public_evaluation_without_internal_metadata():
    unsafe = RailOutcome.block(reason="unsafe", metadata={"provider_result": object()})
    result = _build_rails_result(
        _response("blocked", [_rail("content safety", unsafe, stop=True)]),
        original_content="original",
    )

    dumped = result.model_dump(mode="json")

    assert dumped["status"] == "blocked"
    assert dumped["rail"] == "content safety"
    assert dumped["evaluations"][0]["name"] == "content safety"
    assert dumped["evaluations"][0]["decision"] == "block"
    assert dumped["evaluations"][0]["reason"] == "unsafe"
    assert "metadata" not in dumped["evaluations"][0]
    assert RailsResult.model_validate(dumped).blocking_rails == ["content safety"]
