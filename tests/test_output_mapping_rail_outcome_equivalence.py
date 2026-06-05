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
from nemoguardrails.library.activefence.actions import call_activefence_api
from nemoguardrails.library.ai_defense.actions import ai_defense_inspect
from nemoguardrails.library.autoalign.actions import (
    autoalign_factcheck_output_api,
    autoalign_groundedness_output_api,
    autoalign_output_api,
)
from nemoguardrails.library.clavata.actions import clavata_check
from nemoguardrails.library.cleanlab.actions import call_cleanlab_api
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
from nemoguardrails.library.fiddler.actions import call_fiddler_faithfulness, call_fiddler_safety_bot
from nemoguardrails.library.gcp_moderate_text.actions import call_gcp_text_moderation_api
from nemoguardrails.library.gliner.actions import gliner_detect_pii, gliner_mask_pii
from nemoguardrails.library.guardrails_ai.actions import validate_guardrails_ai_output
from nemoguardrails.library.hallucination.actions import self_check_hallucination as hallucination_action
from nemoguardrails.library.hf_classifier.actions import hf_classifier_check_output
from nemoguardrails.library.injection_detection.actions import injection_detection
from nemoguardrails.library.llama_guard.actions import llama_guard_check_output
from nemoguardrails.library.pangea.actions import TextGuardResult, pangea_ai_guard
from nemoguardrails.library.patronusai.actions import (
    patronus_api_check_output,
    patronus_lynx_check_output_hallucination,
)
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
from nemoguardrails.library.trend_micro.actions import GuardResult, trend_ai_guard
from tests.llama_guard_fixtures import (
    LLAMA_GUARD_SAFE_POLICY_VIOLATIONS,
    LLAMA_GUARD_UNSAFE_POLICY_VIOLATIONS,
)
from tests.policyai_fixtures import POLICYAI_SAFE_OUTCOME_KWARGS, POLICYAI_UNSAFE_OUTCOME_KWARGS


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
        (RailOutcome.allow(), False),
        (RailOutcome.block(), True),
    ],
)
def test_self_check_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, self_check_output)
    outcome = outcome_from_output_mapping(raw_return, self_check_output)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_self_check_output_action_has_no_legacy_output_mapping():
    assert getattr(self_check_output, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ((RailOutcome.allow(), {"events": []}), False),
        ((RailOutcome.block(), {"events": []}), True),
    ],
)
def test_self_check_output_bypass_tuple_unwrap_preserves_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, self_check_output)
    outcome_blocked = outcome_from_output_mapping(raw_return, self_check_output).is_blocked

    assert mapping_blocked is expected_blocked
    assert outcome_blocked is expected_blocked


@pytest.mark.parametrize(
    "action_func",
    [
        self_check_facts,
        alignscore_check_facts,
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.block(accuracy=0.49), True),
        (RailOutcome.allow(accuracy=0.5), False),
        (RailOutcome.allow(accuracy=0.51), False),
    ],
)
def test_fact_check_output_bypass_reads_rail_outcome(action_func, raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, action_func)
    outcome = outcome_from_output_mapping(raw_return, action_func)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


@pytest.mark.parametrize("action_func", [self_check_facts, alignscore_check_facts])
def test_fact_check_actions_have_no_legacy_output_mapping(action_func):
    assert getattr(action_func, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(), False),
        (RailOutcome.block(), True),
    ],
)
def test_hf_classifier_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, hf_classifier_check_output)
    outcome = outcome_from_output_mapping(raw_return, hf_classifier_check_output)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(policy_violations=LLAMA_GUARD_SAFE_POLICY_VIOLATIONS), False),
        (RailOutcome.block(policy_violations=LLAMA_GUARD_UNSAFE_POLICY_VIOLATIONS), True),
    ],
)
def test_llama_guard_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, llama_guard_check_output)
    outcome = outcome_from_output_mapping(raw_return, llama_guard_check_output)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(**POLICYAI_SAFE_OUTCOME_KWARGS), False),
        (RailOutcome.block(**POLICYAI_UNSAFE_OUTCOME_KWARGS), True),
    ],
)
def test_policyai_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, call_policyai_api)
    outcome = outcome_from_output_mapping(raw_return, call_policyai_api)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


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
    "raw_return",
    [
        "NORMAL OUTPUT",
        "MASKED OUTPUT",
    ],
)
def test_mask_output_default_mapping_cannot_express_transform(action_func, raw_return):
    mapping_blocked = is_output_blocked(raw_return, action_func)

    assert mapping_blocked is False


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


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (
            {"guardrails_triggered": False, "pii": {"guarded": False, "response": "NORMAL OUTPUT"}},
            False,
        ),
        (
            {"guardrails_triggered": True, "pii": {"guarded": False, "response": "NORMAL OUTPUT"}},
            True,
        ),
        (
            {"guardrails_triggered": True, "pii": {"guarded": True, "response": "MASKED OUTPUT"}},
            True,
        ),
    ],
)
def test_autoalign_output_mapping_matches_block_interpretation(
    raw_return,
    expected_blocked,
):
    mapping_blocked = is_output_blocked(raw_return, autoalign_output_api)

    assert mapping_blocked is expected_blocked
    assert raw_return["guardrails_triggered"] is expected_blocked


def test_autoalign_output_mapping_cannot_express_transform():
    raw_return = {"guardrails_triggered": False, "pii": {"guarded": True, "response": "MASKED OUTPUT"}}
    mapping_blocked = is_output_blocked(raw_return, autoalign_output_api)

    assert mapping_blocked is False
    assert raw_return["pii"]["guarded"] is True


@pytest.mark.parametrize(
    "action_func",
    [
        autoalign_groundedness_output_api,
        autoalign_factcheck_output_api,
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
def test_autoalign_score_output_mappings_block_below_threshold(action_func, raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, action_func)

    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "reject_blocks", "omit_transforms"),
    [
        (
            {"is_injection": False, "text": "NORMAL OUTPUT", "detections": []},
            False,
            False,
        ),
        (
            {"is_injection": True, "text": "NORMAL OUTPUT", "detections": ["sqli"]},
            True,
            False,
        ),
        (
            {"is_injection": False, "text": "NORMALIZED OUTPUT", "detections": []},
            False,
            True,
        ),
        (
            {"is_injection": True, "text": "OMITTED OUTPUT", "detections": ["sqli"]},
            True,
            True,
        ),
    ],
)
def test_injection_detection_default_mapping_cannot_express_block_or_transform(
    raw_return,
    reject_blocks,
    omit_transforms,
):
    mapping_blocked = is_output_blocked(raw_return, injection_detection)

    assert mapping_blocked is False
    assert reject_blocks is raw_return["is_injection"]
    assert omit_transforms is (raw_return["text"] != "NORMAL OUTPUT")


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (GuardResult(action="Allow", reason="allowed"), False),
        (GuardResult(action="Block", reason="blocked"), True),
    ],
)
def test_trend_micro_output_mapping_matches_block_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, trend_ai_guard)

    assert raw_return.blocked is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"trustworthiness_score": 0.59}, True),
        ({"trustworthiness_score": 0.6}, False),
        ({"trustworthiness_score": 0.61}, False),
        ({}, False),
    ],
)
def test_cleanlab_output_mapping_matches_threshold_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, call_cleanlab_api)

    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"is_blocked": False}, False),
        ({"is_blocked": True}, True),
        ({}, True),
        (None, True),
    ],
)
def test_ai_defense_output_mapping_matches_fail_closed_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, ai_defense_inspect)

    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    "action_func",
    [
        clavata_check,
        call_fiddler_safety_bot,
        call_fiddler_faithfulness,
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "flow_blocked", "mapping_blocked"),
    [
        (False, False, True),
        (True, True, False),
    ],
)
def test_boolean_flag_actions_without_output_mapping_invert_output_bypass(
    action_func,
    raw_return,
    flow_blocked,
    mapping_blocked,
):
    assert is_output_blocked(raw_return, action_func) is mapping_blocked
    assert raw_return is flow_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"max_risk_score": 0.7, "violations": {}}, False),
        ({"max_risk_score": 0.71, "violations": {}}, True),
        ({"max_risk_score": 0.31, "violations": {"adult_content.general": 0.31}}, True),
    ],
)
def test_activefence_output_mapping_matches_max_and_detailed_thresholds(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, call_activefence_api)

    assert mapping_blocked is expected_blocked


def test_activefence_simple_output_flow_allows_detail_only_mapping_block():
    raw_return = {"max_risk_score": 0.31, "violations": {"adult_content.general": 0.31}}
    simple_flow_blocked = raw_return["max_risk_score"] > 0.7
    mapping_blocked = is_output_blocked(raw_return, call_activefence_api)

    assert simple_flow_blocked is False
    assert mapping_blocked is True


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"max_risk_score": 0.8, "violations": {}}, False),
        ({"max_risk_score": 0.81, "violations": {}}, True),
        ({"max_risk_score": 0.41, "violations": {"Derogatory": 0.41}}, True),
    ],
)
def test_gcp_text_moderation_output_mapping_matches_max_and_detailed_thresholds(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, call_gcp_text_moderation_api)

    assert mapping_blocked is expected_blocked


def test_gcp_simple_output_flow_allows_detail_only_mapping_block():
    raw_return = {"max_risk_score": 0.41, "violations": {"Derogatory": 0.41}}
    simple_flow_blocked = raw_return["max_risk_score"] > 0.8
    mapping_blocked = is_output_blocked(raw_return, call_gcp_text_moderation_api)

    assert simple_flow_blocked is False
    assert mapping_blocked is True


@pytest.mark.parametrize(
    ("raw_return", "flow_blocked", "mapping_blocked"),
    [
        ({"valid": True, "validation_result": {"validation_passed": True}}, False, False),
        ({"valid": False, "validation_result": {"validation_passed": False}}, True, True),
    ],
)
def test_guardrails_ai_output_mapping_matches_flow_polarity(raw_return, flow_blocked, mapping_blocked):
    assert is_output_blocked(raw_return, validate_guardrails_ai_output) is mapping_blocked
    assert (not raw_return["valid"]) is flow_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"hallucination": False, "reasoning": ["grounded"]}, False),
        ({"hallucination": True, "reasoning": ["unsupported"]}, True),
    ],
)
def test_patronus_lynx_output_mapping_matches_hallucination_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, patronus_lynx_check_output_hallucination)

    assert raw_return["hallucination"] is expected_blocked
    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        ({"pass": True}, False),
        ({"pass": False}, True),
        ({}, False),
    ],
)
def test_patronus_api_output_mapping_matches_pass_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, patronus_api_check_output)

    assert mapping_blocked is expected_blocked


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (False, False),
        (True, True),
    ],
)
def test_hallucination_output_mapping_matches_blocking_flow_interpretation(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, hallucination_action)

    assert mapping_blocked is expected_blocked
