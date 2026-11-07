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

from typing import get_args

import pytest

from nemoguardrails.actions.rail_outcome import RailDecision, TransformTarget
from nemoguardrails.library.hiddenlayer._verdict import (
    _REDACTABLE,
    BLOCK_REASON,
    REDACT_REASON,
    HiddenLayerActionError,
    HiddenLayerVerdictError,
    Surface,
    _Evaluation,
    _outcome,
)
from tests.fixtures.hiddenlayer import (
    BOT_TEXT,
    PII_TEXT,
    REDACTED_BOT_TEXT,
    REDACTED_PII_TEXT,
    headers,
    input_body,
    output_body,
    tool_call_body,
    tool_result_body,
)

SURFACES = get_args(Surface)
TEXT_SURFACES = ("input", "output")
TOOL_SURFACES = ("tool_call", "tool_result")
# The text the rail sent, which a REDACT has to change to be usable.
CHECKED_TEXT = {"input": PII_TEXT, "output": BOT_TEXT, "tool_call": None, "tool_result": None}
# What each route answers when it returns the check as sent.
AS_SENT = {
    "input": input_body(PII_TEXT),
    "output": output_body(BOT_TEXT),
    "tool_call": tool_call_body(),
    "tool_result": tool_result_body(),
}
REDACTED = {"input": input_body(REDACTED_PII_TEXT), "output": output_body(REDACTED_BOT_TEXT)}


def evaluation(action, body, protection_mode="AS_CONFIGURED") -> _Evaluation:
    return _Evaluation(headers=headers(action, protection_mode), body=body)


def outcome_of(surface, action, body=None, protection_mode="AS_CONFIGURED"):
    body = AS_SENT[surface] if body is None else body
    return _outcome(surface, evaluation(action, body, protection_mode), CHECKED_TEXT[surface])


class TestAllowActions:
    @pytest.mark.parametrize("surface", SURFACES)
    @pytest.mark.parametrize("action", ["NONE", "DETECT"])
    def test_allows(self, surface, action):
        outcome = outcome_of(surface, action)
        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata == {"action": action, "protection_mode": "AS_CONFIGURED"}


class TestBlock:
    @pytest.mark.parametrize("surface", SURFACES)
    def test_blocks_and_does_not_forward_the_returned_check(self, surface):
        outcome = outcome_of(surface, "BLOCK")
        assert outcome.decision is RailDecision.BLOCK
        assert outcome.reason == BLOCK_REASON
        assert outcome.metadata == {"action": "BLOCK", "protection_mode": "AS_CONFIGURED"}

    @pytest.mark.parametrize("surface", TOOL_SURFACES)
    def test_a_promoted_tool_redaction_blocks(self, surface):
        """HiddenLayer never returns a redacted check carrying a tool call or result."""
        outcome = outcome_of(surface, "BLOCK", protection_mode="INTEGRATION_LIMITED_PROMOTE")
        assert outcome.decision is RailDecision.BLOCK
        assert outcome.metadata["protection_mode"] == "INTEGRATION_LIMITED_PROMOTE"


class TestMonitorMode:
    """Monitor Mode arrives as an allow action; the policy's own action is not enforced."""

    @pytest.mark.parametrize("surface", SURFACES)
    def test_detect_in_monitor_mode_allows(self, surface):
        outcome = outcome_of(surface, "DETECT", protection_mode="MONITOR_MODE")
        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata == {"action": "DETECT", "protection_mode": "MONITOR_MODE"}

    def test_a_block_in_monitor_mode_is_still_a_block(self):
        """The server applies Monitor Mode, so a BLOCK that arrives is one the rail must enforce."""
        assert outcome_of("input", "BLOCK", protection_mode="MONITOR_MODE").decision is RailDecision.BLOCK


class TestProtectionMode:
    def test_an_unknown_mode_passes_through(self):
        outcome = outcome_of("input", "NONE", protection_mode="SOMETHING_NEW")
        assert outcome.decision is RailDecision.ALLOW
        assert outcome.metadata["protection_mode"] == "SOMETHING_NEW"

    def test_a_missing_mode_is_none(self):
        evaluation = _Evaluation(headers={"HL-Runtime-Action": "NONE"}, body=AS_SENT["input"])
        assert _outcome("input", evaluation, PII_TEXT).metadata["protection_mode"] is None

    def test_headers_are_read_case_insensitively(self):
        evaluation = _Evaluation(
            headers={"hl-runtime-action": "BLOCK", "HL-RUNTIME-PROTECTION-MODE": "MONITOR_MODE"},
            body=AS_SENT["input"],
        )
        outcome = _outcome("input", evaluation, PII_TEXT)
        assert outcome.decision is RailDecision.BLOCK
        assert outcome.metadata == {"action": "BLOCK", "protection_mode": "MONITOR_MODE"}


class TestMissingOrUnknownAction:
    """No action means no verdict, which the rail classes with the transport failures."""

    def test_missing_action_header(self):
        with pytest.raises(HiddenLayerActionError):
            _outcome("input", _Evaluation(headers={}, body=AS_SENT["input"]), PII_TEXT)

    def test_unknown_action(self):
        with pytest.raises(HiddenLayerActionError):
            outcome_of("input", "QUARANTINE")

    def test_the_message_never_quotes_the_response(self):
        with pytest.raises(HiddenLayerActionError) as error:
            outcome_of("input", "QUARANTINE")
        assert "QUARANTINE" not in str(error.value)


class TestRedact:
    @pytest.mark.parametrize(("surface", "target"), [("input", "user_message"), ("output", "bot_message")])
    def test_transforms_the_text_surface(self, surface, target):
        outcome = outcome_of(surface, "REDACT", REDACTED[surface])
        assert outcome.decision is RailDecision.TRANSFORM
        assert outcome.reason == REDACT_REASON
        transform = outcome.transforms[0]
        assert (transform.target, transform.text) == (
            TransformTarget(target),
            {"input": REDACTED_PII_TEXT, "output": REDACTED_BOT_TEXT}[surface],
        )

    @pytest.mark.parametrize("surface", TOOL_SURFACES)
    def test_blocks_on_a_tool_surface(self, surface):
        """Defensive: the server promotes a REDACT on a tool evaluation, so the rail should not see one."""
        assert outcome_of(surface, "REDACT").decision is RailDecision.BLOCK

    @pytest.mark.parametrize("surface", TEXT_SURFACES)
    def test_unchanged_text_is_unusable(self, surface):
        with pytest.raises(HiddenLayerVerdictError):
            outcome_of(surface, "REDACT", AS_SENT[surface])

    @pytest.mark.parametrize(
        "body",
        [
            {"check": "input", "model": "m"},
            {"check": "input", "model": "m", "user_message": None},
            {"check": "output", "model": "m", "bot_message": "redacted"},
            "not an object",
        ],
        ids=["no-user-message", "null-user-message", "wrong-check", "not-an-object"],
    )
    def test_a_body_with_no_redacted_text_is_unusable(self, body):
        with pytest.raises(HiddenLayerVerdictError):
            outcome_of("input", "REDACT", body)

    def test_the_message_never_quotes_the_response(self):
        with pytest.raises(HiddenLayerVerdictError) as error:
            outcome_of("input", "REDACT", input_body(PII_TEXT))
        assert PII_TEXT not in str(error.value)


class TestRedactableSurfaces:
    def test_only_the_text_surfaces_have_a_rewrite_channel(self):
        assert set(_REDACTABLE) == set(TEXT_SURFACES)


class TestMetadata:
    @pytest.mark.parametrize("surface", SURFACES)
    def test_carries_the_verdict_and_nothing_from_the_check(self, surface):
        assert set(outcome_of(surface, "NONE").metadata) == {"action", "protection_mode"}
