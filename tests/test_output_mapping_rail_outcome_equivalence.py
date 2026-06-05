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

from nemoguardrails.actions import action
from nemoguardrails.actions.output_mapping import (
    is_output_blocked,
    outcome_from_output_mapping,
)
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.content_safety.actions import (
    content_safety_check_output_mapping,
)
from nemoguardrails.library.crowdstrike_aidr.actions import (
    GuardChatCompletionsResult,
    crowdstrike_aidr_guard,
)
from nemoguardrails.library.factchecking.align_score.actions import (
    alignscore_check_facts,
)
from nemoguardrails.library.gliner.actions import gliner_detect_pii, gliner_mask_pii
from nemoguardrails.library.hf_classifier.actions import hf_classifier_check_output
from nemoguardrails.library.llama_guard.actions import llama_guard_check_output
from nemoguardrails.library.pangea.actions import TextGuardResult, pangea_ai_guard
from nemoguardrails.library.policyai.actions import call_policyai_api
from nemoguardrails.library.privateai.actions import (
    detect_pii as privateai_detect_pii,
)
from nemoguardrails.library.privateai.actions import (
    mask_pii as privateai_mask_pii,
)
from nemoguardrails.library.prompt_security.actions import protect_text
from nemoguardrails.library.regex.actions import detect_regex_pattern
from nemoguardrails.library.self_check.facts.actions import self_check_facts
from nemoguardrails.library.self_check.output_check.actions import self_check_output
from nemoguardrails.library.sensitive_data_detection.actions import (
    detect_sensitive_data,
    mask_sensitive_data,
)


@action()
def detect_only_action(result):
    return result


def _candidate_is_blocked(result):
    if isinstance(result, bool):
        return not result
    if isinstance(result, (int, float)):
        return result < 0.5
    return False


REPRESENTATIVE_RAW_RETURNS = [
    True,
    False,
    1.0,
    0.0,
    0.49,
    0.5,
    0.51,
    42,
    "hi",
    None,
    {"allowed": False},
    [],
]


@pytest.mark.parametrize("raw_return", REPRESENTATIVE_RAW_RETURNS)
def test_candidate_interpretation_matches_is_output_blocked(raw_return):
    outcome_blocked = _candidate_is_blocked(raw_return)
    mapping_blocked = is_output_blocked(raw_return, detect_only_action)

    assert outcome_blocked == mapping_blocked


def test_numeric_threshold_boundary_is_pinned_on_both_sides():
    assert _candidate_is_blocked(0.49) is True
    assert is_output_blocked(0.49, detect_only_action) is True

    assert _candidate_is_blocked(0.5) is False
    assert is_output_blocked(0.5, detect_only_action) is False


def test_boolean_semantics_are_pinned_on_both_sides():
    assert _candidate_is_blocked(True) is False
    assert is_output_blocked(True, detect_only_action) is False

    assert _candidate_is_blocked(False) is True
    assert is_output_blocked(False, detect_only_action) is True


@action(output_mapping=content_safety_check_output_mapping)
def content_safety_action(result):
    return result


def test_content_safety_mapping_characterization_for_p3_golden():
    assert is_output_blocked(RailOutcome.block(), content_safety_action) is True
    assert is_output_blocked(RailOutcome.allow(), content_safety_action) is False


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (True, False),
        (False, True),
    ],
)
def test_self_check_output_mapping_matches_default_interpret(raw_return, expected_blocked):
    outcome_blocked = _candidate_is_blocked(raw_return)
    mapping_blocked = is_output_blocked(raw_return, self_check_output)

    assert outcome_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ((True, {"events": []}), False),
        ((False, {"events": []}), True),
    ],
)
def test_self_check_output_mapping_tuple_unwrap_matches_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, self_check_output)
    outcome_blocked = outcome_from_output_mapping(raw_return, self_check_output).is_blocked

    assert mapping_blocked is expected_blocked
    assert outcome_blocked is expected_blocked


@pytest.mark.parametrize(
    ("action_func", "action_name"),
    [
        (self_check_facts, "self_check_facts"),
        (alignscore_check_facts, "alignscore_check_facts"),
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (0.49, True),
        (0.5, False),
        (0.51, False),
    ],
)
def test_fact_check_mapping_threshold_matches_default_interpret(action_func, action_name, raw_return, expected_blocked):
    outcome_blocked = _candidate_is_blocked(raw_return)
    mapping_blocked = is_output_blocked(raw_return, action_func)

    assert outcome_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (True, False),
        (False, True),
    ],
)
def test_hf_classifier_output_mapping_matches_default_interpret(raw_return, expected_blocked):
    outcome_blocked = _candidate_is_blocked(raw_return)
    mapping_blocked = is_output_blocked(raw_return, hf_classifier_check_output)

    assert outcome_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"allowed": True, "policy_violations": None}, False),
        ({"allowed": False, "policy_violations": ["S1"]}, True),
    ],
)
def test_llama_guard_output_mapping_matches_interpretation(raw_return, expected_blocked):
    interpreted_blocked = not raw_return["allowed"]
    mapping_blocked = is_output_blocked(raw_return, llama_guard_check_output)

    assert interpreted_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"assessment": "SAFE", "category": "Safe"}, False),
        ({"assessment": "UNSAFE", "category": "violence"}, True),
        ({}, False),
    ],
)
def test_policyai_output_mapping_matches_interpretation(raw_return, expected_blocked):
    interpreted_blocked = raw_return.get("assessment", "SAFE") == "UNSAFE"
    mapping_blocked = is_output_blocked(raw_return, call_policyai_api)

    assert interpreted_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"is_match": False, "text": "hello", "detections": []}, False),
        ({"is_match": True, "text": "secret", "detections": ["secret"]}, True),
    ],
)
def test_regex_output_mapping_matches_interpretation(raw_return, expected_blocked):
    interpreted_blocked = raw_return["is_match"]
    mapping_blocked = is_output_blocked(raw_return, detect_regex_pattern)

    assert interpreted_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    "action_func",
    [
        privateai_detect_pii,
        gliner_detect_pii,
        detect_sensitive_data,
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (False, False),
        (True, True),
    ],
)
def test_detect_pii_output_mapping_matches_interpretation(action_func, raw_return, expected_blocked):
    interpreted_blocked = raw_return
    mapping_blocked = is_output_blocked(raw_return, action_func)

    assert interpreted_blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    "action_func",
    [
        privateai_mask_pii,
        gliner_mask_pii,
        mask_sensitive_data,
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "is_transform"),
    [
        ("NORMAL OUTPUT", False),
        ("MASKED OUTPUT", True),
    ],
)
def test_mask_output_default_mapping_cannot_express_transform(action_func, raw_return, is_transform):
    mapping_blocked = is_output_blocked(raw_return, action_func)

    assert mapping_blocked is False
    assert is_transform is (raw_return != "NORMAL OUTPUT")


@pytest.mark.parametrize(
    ("action_func", "raw_return", "interpreted_blocked", "is_transform"),
    [
        (
            pangea_ai_guard,
            TextGuardResult(blocked=False, transformed=False, bot_message="NORMAL OUTPUT"),
            False,
            False,
        ),
        (
            pangea_ai_guard,
            TextGuardResult(blocked=True, transformed=False, bot_message="NORMAL OUTPUT"),
            True,
            False,
        ),
        (
            pangea_ai_guard,
            TextGuardResult(blocked=False, transformed=True, bot_message="MASKED OUTPUT"),
            False,
            True,
        ),
        (
            crowdstrike_aidr_guard,
            GuardChatCompletionsResult(blocked=False, transformed=False, bot_message="NORMAL OUTPUT"),
            False,
            False,
        ),
        (
            crowdstrike_aidr_guard,
            GuardChatCompletionsResult(blocked=True, transformed=False, bot_message="NORMAL OUTPUT"),
            True,
            False,
        ),
        (
            crowdstrike_aidr_guard,
            GuardChatCompletionsResult(blocked=False, transformed=True, bot_message="MASKED OUTPUT"),
            False,
            True,
        ),
    ],
)
def test_vendor_object_default_mapping_cannot_express_block_or_transform(
    action_func,
    raw_return,
    interpreted_blocked,
    is_transform,
):
    mapping_blocked = is_output_blocked(raw_return, action_func)

    assert mapping_blocked is False
    assert interpreted_blocked is bool(raw_return.blocked)
    assert is_transform is (not raw_return.blocked and bool(raw_return.transformed))


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"is_blocked": False, "is_modified": False, "modified_text": None}, False),
        ({"is_blocked": True, "is_modified": False, "modified_text": None}, True),
        ({"is_blocked": True, "is_modified": True, "modified_text": "MASKED OUTPUT"}, True),
    ],
)
def test_prompt_security_output_mapping_matches_block_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, protect_text)

    assert raw_return["is_blocked"] is expected_blocked
    assert mapping_blocked is expected_blocked


def test_prompt_security_output_mapping_cannot_express_transform():
    raw_return = {"is_blocked": False, "is_modified": True, "modified_text": "MASKED OUTPUT"}
    mapping_blocked = is_output_blocked(raw_return, protect_text)

    assert mapping_blocked is False
    assert raw_return["is_modified"] is True
