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

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.activefence.actions import _activefence_outcome
from nemoguardrails.library.ai_defense.actions import _ai_defense_outcome
from nemoguardrails.library.clavata.actions import _clavata_outcome
from nemoguardrails.library.cleanlab.actions import _cleanlab_outcome
from nemoguardrails.library.fiddler.actions import _fiddler_outcome
from nemoguardrails.library.gcp_moderate_text.actions import _gcp_text_moderation_outcome
from nemoguardrails.library.trend_micro.actions import GuardResult, _trend_micro_outcome


@pytest.mark.parametrize(
    ("guard_result", "expected"),
    [
        (
            GuardResult(action="Allow", reason="No threats detected"),
            RailOutcome.allow(reason="No threats detected", action="Allow"),
        ),
        (
            GuardResult(action="Block", reason="Prompt Attack Detected"),
            RailOutcome.block(reason="Prompt Attack Detected", action="Block"),
        ),
    ],
)
def test_trend_micro_outcome_preserves_action_and_reason(guard_result, expected):
    assert _trend_micro_outcome(guard_result) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.59, RailOutcome.block(trustworthiness_score=0.59)),
        (0.6, RailOutcome.allow(trustworthiness_score=0.6)),
        (0.61, RailOutcome.allow(trustworthiness_score=0.61)),
    ],
)
def test_cleanlab_outcome_pins_threshold(score, expected):
    assert _cleanlab_outcome(score) == expected


@pytest.mark.parametrize(
    ("is_blocked", "expected"),
    [
        (False, RailOutcome.allow(is_blocked=False)),
        (True, RailOutcome.block(is_blocked=True)),
    ],
)
def test_ai_defense_outcome_preserves_fail_open_closed_decision(is_blocked, expected):
    assert _ai_defense_outcome(is_blocked) == expected


@pytest.mark.parametrize(
    ("policy_matched", "expected"),
    [
        (False, RailOutcome.allow(policy_matched=False)),
        (True, RailOutcome.block(policy_matched=True)),
    ],
)
def test_clavata_outcome_preserves_policy_match_decision(policy_matched, expected):
    assert _clavata_outcome(policy_matched) == expected


@pytest.mark.parametrize(
    ("blocked", "expected"),
    [
        (False, RailOutcome.allow(blocked=False)),
        (True, RailOutcome.block(blocked=True)),
    ],
)
def test_fiddler_outcome_preserves_detector_decision(blocked, expected):
    assert _fiddler_outcome(blocked) == expected


@pytest.mark.parametrize(
    ("max_risk_score", "threshold_mode", "violations", "expected_blocked"),
    [
        (0.7, "simple", {}, False),
        (0.71, "simple", {}, True),
        (0.31, "simple", {"adult_content.general": 0.31}, False),
        (0.3, "detailed", {"adult_content.general": 0.3}, False),
        (0.31, "detailed", {"adult_content.general": 0.31}, True),
    ],
)
def test_activefence_outcome_pins_simple_and_detailed_thresholds(
    max_risk_score,
    threshold_mode,
    violations,
    expected_blocked,
):
    outcome = _activefence_outcome(max_risk_score, violations, threshold_mode)

    assert outcome.is_blocked is expected_blocked
    assert outcome.metadata["max_risk_score"] == max_risk_score
    assert outcome.metadata["violations"] == violations
    assert outcome.metadata["threshold_mode"] == threshold_mode


@pytest.mark.parametrize(
    ("max_risk_score", "threshold_mode", "violations", "expected_blocked"),
    [
        (0.8, "simple", {}, False),
        (0.81, "simple", {}, True),
        (0.41, "simple", {"Derogatory": 0.41}, False),
        (0.4, "detailed", {"Derogatory": 0.4}, False),
        (0.41, "detailed", {"Derogatory": 0.41}, True),
    ],
)
def test_gcp_text_moderation_outcome_pins_simple_and_detailed_thresholds(
    max_risk_score,
    threshold_mode,
    violations,
    expected_blocked,
):
    outcome = _gcp_text_moderation_outcome(max_risk_score, violations, threshold_mode)

    assert outcome.is_blocked is expected_blocked
    assert outcome.metadata["max_risk_score"] == max_risk_score
    assert outcome.metadata["violations"] == violations
    assert outcome.metadata["threshold_mode"] == threshold_mode
