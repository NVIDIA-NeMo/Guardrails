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

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.activefence.actions import ACTIVEFENCE_DETAILED_RULES, _activefence_outcome
from nemoguardrails.library.ai_defense.actions import _ai_defense_outcome
from nemoguardrails.library.clavata.actions import _clavata_outcome
from nemoguardrails.library.cleanlab.actions import _cleanlab_outcome, call_cleanlab_api
from nemoguardrails.library.fiddler.actions import _fiddler_outcome
from nemoguardrails.library.gcp_moderate_text.actions import (
    GCP_TEXT_DETAILED_THRESHOLDS,
    _gcp_text_moderation_outcome,
    call_gcp_text_moderation_api,
)
from nemoguardrails.library.trend_micro.actions import GuardResult, _trend_micro_outcome


@pytest.mark.parametrize(
    ("guard_result", "expected"),
    [
        (
            GuardResult(action="Allow", reason="No threats detected"),
            RailOutcome.allow(reason="No threats detected", metadata={"action": "Allow"}),
        ),
        (
            GuardResult(action="Block", reason="Prompt Attack Detected"),
            RailOutcome.block(reason="Prompt Attack Detected", metadata={"action": "Block"}),
        ),
    ],
)
def test_trend_micro_outcome_preserves_action_and_reason(guard_result, expected):
    assert _trend_micro_outcome(guard_result) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.59, RailOutcome.block(metadata={"trustworthiness_score": 0.59})),
        (0.6, RailOutcome.allow(metadata={"trustworthiness_score": 0.6})),
        (0.61, RailOutcome.allow(metadata={"trustworthiness_score": 0.61})),
    ],
)
def test_cleanlab_outcome_pins_threshold(score, expected):
    assert _cleanlab_outcome(score) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "explicit", "expected"),
    [
        (
            {"user_message": "legacy prompt", "bot_message": "legacy response"},
            {"user_message": "bound prompt", "bot_message": "bound response"},
            ("bound prompt", "bound response"),
        ),
        (
            {"user_message": "legacy prompt", "bot_message": "legacy response"},
            {},
            ("legacy prompt", "legacy response"),
        ),
    ],
)
async def test_cleanlab_prefers_bound_messages_and_retains_context_fallback(monkeypatch, context, explicit, expected):
    score = AsyncMock(return_value={"trustworthiness_score": 0.9})
    studio = MagicMock()
    studio.TLM.return_value = SimpleNamespace(get_trustworthiness_score_async=score)
    studio_factory = MagicMock(return_value=studio)
    cleanlab_studio = ModuleType("cleanlab_studio")
    setattr(cleanlab_studio, "Studio", studio_factory)
    monkeypatch.setitem(sys.modules, "cleanlab_studio", cleanlab_studio)
    monkeypatch.setenv("CLEANLAB_API_KEY", "test-key")

    await call_cleanlab_api(context=context, **explicit)

    score.assert_awaited_once_with(expected[0], response=expected[1])


@pytest.mark.parametrize(
    ("is_blocked", "expected"),
    [
        (False, RailOutcome.allow(metadata={"is_blocked": False})),
        (True, RailOutcome.block(metadata={"is_blocked": True})),
    ],
)
def test_ai_defense_outcome_preserves_fail_open_closed_decision(is_blocked, expected):
    assert _ai_defense_outcome(is_blocked) == expected


@pytest.mark.parametrize(
    ("policy_matched", "expected"),
    [
        (False, RailOutcome.allow(metadata={"policy_matched": False})),
        (True, RailOutcome.block(metadata={"policy_matched": True})),
    ],
)
def test_clavata_outcome_preserves_policy_match_decision(policy_matched, expected):
    assert _clavata_outcome(policy_matched) == expected


@pytest.mark.parametrize(
    ("blocked", "expected"),
    [
        (False, RailOutcome.allow(metadata={"blocked": False})),
        (True, RailOutcome.block(metadata={"blocked": True})),
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
    assert outcome.metadata["triggered_violation"] == (
        "adult_content.general" if threshold_mode == "detailed" and expected_blocked else None
    )
    assert bool(outcome.reason) is expected_blocked


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
    assert outcome.metadata["triggered_violation"] == (
        "Derogatory" if threshold_mode == "detailed" and expected_blocked else None
    )
    assert bool(outcome.reason) is expected_blocked


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "explicit_user_message", "expected"),
    [
        ({"user_message": "legacy prompt"}, "bound prompt", "bound prompt"),
        ({"user_message": "legacy prompt"}, None, "legacy prompt"),
    ],
)
async def test_gcp_moderation_prefers_bound_message_and_retains_context_fallback(
    monkeypatch, context, explicit_user_message, expected
):
    class Document:
        class Type:
            PLAIN_TEXT = "plain_text"

    moderate_text = AsyncMock(return_value=SimpleNamespace(moderation_categories=[]))
    client_factory = MagicMock(return_value=SimpleNamespace(moderate_text=moderate_text))
    language_v2 = ModuleType("google.cloud.language_v2")
    setattr(language_v2, "Document", Document)
    setattr(language_v2, "LanguageServiceAsyncClient", client_factory)
    google_cloud = ModuleType("google.cloud")
    setattr(google_cloud, "language_v2", language_v2)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.language_v2", language_v2)

    await call_gcp_text_moderation_api(context=context, user_message=explicit_user_message)

    document = moderate_text.await_args.kwargs["document"]
    assert document.content == expected


@pytest.mark.parametrize(
    ("violation", "threshold", "reason"),
    [(violation, threshold, reason) for violation, (threshold, reason) in ACTIVEFENCE_DETAILED_RULES.items()],
)
def test_activefence_detailed_outcome_owns_each_rule(violation, threshold, reason):
    at_threshold = _activefence_outcome(threshold, {violation: threshold}, "detailed")
    above_threshold = _activefence_outcome(threshold + 0.01, {violation: threshold + 0.01}, "detailed")

    assert at_threshold == RailOutcome.allow(
        metadata={
            "max_risk_score": threshold,
            "violations": {violation: threshold},
            "threshold_mode": "detailed",
            "triggered_violation": None,
        }
    )
    assert above_threshold.is_blocked
    assert above_threshold.metadata["triggered_violation"] == violation
    assert above_threshold.reason == reason


@pytest.mark.parametrize(("violation", "threshold"), GCP_TEXT_DETAILED_THRESHOLDS.items())
def test_gcp_detailed_outcome_owns_each_threshold(violation, threshold):
    at_threshold = _gcp_text_moderation_outcome(threshold, {violation: threshold}, "detailed")
    above_threshold = _gcp_text_moderation_outcome(threshold + 0.01, {violation: threshold + 0.01}, "detailed")

    assert at_threshold == RailOutcome.allow(
        metadata={
            "max_risk_score": threshold,
            "violations": {violation: threshold},
            "threshold_mode": "detailed",
            "triggered_violation": None,
        }
    )
    assert above_threshold.is_blocked
    assert above_threshold.metadata["triggered_violation"] == violation
    assert above_threshold.reason


def test_detailed_outcomes_preserve_legacy_first_match_order():
    activefence = _activefence_outcome(
        0.91,
        {
            "adult_content.general": 0.91,
            "abusive_or_harmful.harassment_or_bullying": 0.91,
        },
        "detailed",
    )
    gcp = _gcp_text_moderation_outcome(
        0.91,
        {"Derogatory": 0.91, "Toxic": 0.91},
        "detailed",
    )

    assert activefence.metadata["triggered_violation"] == "abusive_or_harmful.harassment_or_bullying"
    assert gcp.metadata["triggered_violation"] == "Toxic"
