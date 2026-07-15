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
from collections.abc import Hashable
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
    outcome = RailOutcome.allow(metadata={"policy_violations": [], "score": 0.1})
    assert outcome.metadata == {"policy_violations": [], "score": 0.1}


def test_metadata_mapping_is_copied():
    metadata = {"score": 0.1}
    outcome = RailOutcome(decision=RailDecision.ALLOW, metadata=metadata)

    metadata["score"] = 0.9

    assert outcome.metadata == {"score": 0.1}
    assert outcome.metadata is not metadata


def test_metadata_accepts_backend_specific_values():
    provider_result = object()
    outcome = RailOutcome.allow(metadata={"provider_result": provider_result})

    assert outcome.metadata["provider_result"] is provider_result


def test_block_is_blocked_with_reason_and_metadata():
    outcome = RailOutcome.block(reason="unsafe", metadata={"policy_violations": ["S1: Violence"]})
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


def test_metadata_reason_key_does_not_replace_canonical_reason():
    outcome = RailOutcome.block(metadata={"reason": "provider reason"})

    assert outcome.reason is None
    assert outcome.metadata == {"reason": "provider reason"}


def test_factory_rejects_misspelled_reason_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cast(Any, RailOutcome.block)(resaon="unsafe")


def test_transform_normalizes_string_target():
    outcome = RailOutcome.transform([(cast(Any, "user_message"), "masked input")])

    assert outcome.transforms == (TransformSpec(target=TransformTarget.USER_MESSAGE, text="masked input"),)
    assert outcome.transform_text == {"user_message": "masked input"}


def test_transform_rejects_unknown_target():
    with pytest.raises(ValueError):
        RailOutcome.transform([(cast(Any, "unknown"), "masked input")])


def test_transform_rejects_non_string_text():
    with pytest.raises(TypeError, match="transform text must be a string"):
        TransformSpec(target=TransformTarget.USER_MESSAGE, text=cast(Any, 123))


def test_transform_rejects_duplicate_targets():
    with pytest.raises(ValueError, match="transforms must not repeat the same TransformTarget"):
        RailOutcome.transform(
            [
                (TransformTarget.USER_MESSAGE, "first"),
                (TransformTarget.USER_MESSAGE, "second"),
            ]
        )


def test_transform_container_is_defensively_normalized_to_tuple():
    transforms = [TransformSpec(target=TransformTarget.USER_MESSAGE, text="masked input")]
    outcome = RailOutcome(
        decision=RailDecision.TRANSFORM,
        transforms=cast(Any, transforms),
    )

    transforms.clear()

    assert outcome.transforms == (TransformSpec(target=TransformTarget.USER_MESSAGE, text="masked input"),)


def test_transform_container_rejects_non_transform_specs():
    with pytest.raises(TypeError, match="transforms must contain only TransformSpec instances"):
        RailOutcome(
            decision=RailDecision.TRANSFORM,
            transforms=cast(Any, [object()]),
        )


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


def test_outcome_is_explicitly_unhashable():
    outcome = RailOutcome.allow()

    assert not isinstance(outcome, Hashable)
    with pytest.raises(TypeError, match="unhashable type"):
        hash(outcome)


def test_reason_rejects_non_string_value():
    with pytest.raises(TypeError, match="reason must be a string or None"):
        RailOutcome.allow(reason=cast(Any, 123))


def test_metadata_rejects_non_string_keys():
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        RailOutcome.allow(metadata=cast(Any, {1: "value"}))


def test_metadata_rejects_non_mapping_value():
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        RailOutcome.allow(metadata=cast(Any, []))


def test_decision_string_is_normalized():
    outcome = RailOutcome(decision=cast(Any, "block"))

    assert outcome.decision is RailDecision.BLOCK
    assert outcome.is_blocked is True


def test_unknown_decision_is_rejected():
    with pytest.raises(ValueError):
        RailOutcome(decision=cast(Any, "unknown"))


def test_require_rail_outcome_accepts_outcome():
    outcome = RailOutcome.allow()
    assert require_rail_outcome(outcome) is outcome


@pytest.mark.parametrize("result", [None, False, {}, (RailOutcome.allow(), "success")])
def test_require_rail_outcome_rejects_other_results(result):
    with pytest.raises(TypeError, match="Rail action must return RailOutcome"):
        require_rail_outcome(result)
