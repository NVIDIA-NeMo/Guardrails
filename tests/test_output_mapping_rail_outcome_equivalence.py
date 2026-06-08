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
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.library.activefence.actions import call_activefence_api
from nemoguardrails.library.ai_defense.actions import ai_defense_inspect
from nemoguardrails.library.autoalign.actions import (
    autoalign_factcheck_output_api,
    autoalign_groundedness_output_api,
    autoalign_output_api,
)
from nemoguardrails.library.clavata.actions import clavata_check
from nemoguardrails.library.cleanlab.actions import call_cleanlab_api
from nemoguardrails.library.content_safety.actions import content_safety_check_output
from nemoguardrails.library.crowdstrike_aidr.actions import crowdstrike_aidr_guard
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
from nemoguardrails.library.pangea.actions import pangea_ai_guard
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
from nemoguardrails.library.trend_micro.actions import trend_ai_guard
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


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(policy_violations=[]), False),
        (RailOutcome.block(policy_violations=["violence"]), True),
    ],
)
def test_content_safety_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, content_safety_check_output)
    outcome = outcome_from_output_mapping(raw_return, content_safety_check_output)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_content_safety_output_action_has_no_legacy_output_mapping():
    assert getattr(content_safety_check_output, "action_meta")["output_mapping"] is None


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
        (RailOutcome.allow(is_match=False, text="hello", detections=[]), False),
        (RailOutcome.block(is_match=True, text="secret", detections=["secret"]), True),
    ],
)
def test_regex_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, detect_regex_pattern)
    outcome = outcome_from_output_mapping(raw_return, detect_regex_pattern)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_regex_action_has_no_legacy_output_mapping():
    assert getattr(detect_regex_pattern, "action_meta")["output_mapping"] is None


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
        (RailOutcome.allow(), False),
        (RailOutcome.block(), True),
    ],
)
def test_detect_pii_output_bypass_reads_rail_outcome(action_func, raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, action_func)
    outcome = outcome_from_output_mapping(raw_return, action_func)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


@pytest.mark.parametrize("action_func", [privateai_detect_pii, gliner_detect_pii, detect_sensitive_data])
def test_detect_pii_actions_have_no_legacy_output_mapping(action_func):
    assert getattr(action_func, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    "action_func",
    [
        privateai_mask_pii,
        gliner_mask_pii,
        mask_sensitive_data,
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(text="NORMAL OUTPUT", masked_text="NORMAL OUTPUT"), False),
        (
            RailOutcome.transform(
                [(TransformTarget.BOT_MESSAGE, "MASKED OUTPUT")],
                text="NORMAL OUTPUT",
                masked_text="MASKED OUTPUT",
            ),
            False,
        ),
    ],
)
def test_mask_output_bypass_reads_rail_outcome(action_func, raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, action_func)
    outcome = outcome_from_output_mapping(raw_return, action_func)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


@pytest.mark.parametrize("action_func", [privateai_mask_pii, gliner_mask_pii, mask_sensitive_data])
def test_mask_actions_have_no_legacy_output_mapping(action_func):
    assert getattr(action_func, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(blocked=False, transformed=False), False),
        (RailOutcome.block(blocked=True, transformed=False), True),
        (
            RailOutcome.transform(
                [
                    (TransformTarget.USER_MESSAGE, "hello"),
                    (TransformTarget.BOT_MESSAGE, "MASKED OUTPUT"),
                ],
                blocked=False,
                transformed=True,
            ),
            False,
        ),
    ],
)
def test_pangea_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, pangea_ai_guard)
    outcome = outcome_from_output_mapping(raw_return, pangea_ai_guard)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_pangea_output_action_has_no_legacy_output_mapping():
    assert getattr(pangea_ai_guard, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(blocked=False, transformed=False), False),
        (RailOutcome.block(blocked=True, transformed=False), True),
        (
            RailOutcome.transform(
                [
                    (TransformTarget.USER_MESSAGE, "hello"),
                    (TransformTarget.BOT_MESSAGE, "MASKED OUTPUT"),
                ],
                blocked=False,
                transformed=True,
            ),
            False,
        ),
    ],
)
def test_crowdstrike_aidr_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, crowdstrike_aidr_guard)
    outcome = outcome_from_output_mapping(raw_return, crowdstrike_aidr_guard)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_crowdstrike_aidr_action_has_no_legacy_output_mapping():
    assert getattr(crowdstrike_aidr_guard, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(is_blocked=False, is_modified=False, modified_text=None), False),
        (RailOutcome.block(is_blocked=True, is_modified=False, modified_text=None), True),
        (
            RailOutcome.transform(
                [(TransformTarget.BOT_MESSAGE, "MASKED OUTPUT")],
                is_blocked=False,
                is_modified=True,
                modified_text="MASKED OUTPUT",
            ),
            False,
        ),
    ],
)
def test_prompt_security_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, protect_text)
    outcome = outcome_from_output_mapping(raw_return, protect_text)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_prompt_security_action_has_no_legacy_output_mapping():
    assert getattr(protect_text, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(guardrails_triggered=False, pii={"guarded": False, "response": "NORMAL OUTPUT"}), False),
        (RailOutcome.block(guardrails_triggered=True, pii={"guarded": False, "response": "NORMAL OUTPUT"}), True),
        (
            RailOutcome.transform(
                [(TransformTarget.BOT_MESSAGE, "MASKED OUTPUT")],
                guardrails_triggered=False,
                pii={"guarded": True, "response": "MASKED OUTPUT"},
            ),
            False,
        ),
    ],
)
def test_autoalign_output_bypass_reads_rail_outcome(
    raw_return,
    expected_blocked,
):
    mapping_blocked = is_output_blocked(raw_return, autoalign_output_api)
    outcome = outcome_from_output_mapping(raw_return, autoalign_output_api)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_autoalign_output_action_has_no_legacy_output_mapping():
    assert getattr(autoalign_output_api, "action_meta")["output_mapping"] is None


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
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(is_injection=False, text="NORMAL OUTPUT", detections=[]), False),
        (RailOutcome.block(is_injection=True, text="NORMAL OUTPUT", detections=["sqli"]), True),
        (
            RailOutcome.transform(
                [(TransformTarget.BOT_MESSAGE, "OMITTED OUTPUT")],
                is_injection=True,
                text="OMITTED OUTPUT",
                detections=["sqli"],
            ),
            False,
        ),
    ],
)
def test_injection_detection_output_bypass_reads_rail_outcome(
    raw_return,
    expected_blocked,
):
    mapping_blocked = is_output_blocked(raw_return, injection_detection)
    outcome = outcome_from_output_mapping(raw_return, injection_detection)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_injection_detection_action_has_no_legacy_output_mapping():
    assert getattr(injection_detection, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(reason="allowed", action="Allow"), False),
        (RailOutcome.block(reason="blocked", action="Block"), True),
    ],
)
def test_trend_micro_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, trend_ai_guard)
    outcome = outcome_from_output_mapping(raw_return, trend_ai_guard)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_trend_micro_action_has_no_legacy_output_mapping():
    assert getattr(trend_ai_guard, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.block(trustworthiness_score=0.59), True),
        (RailOutcome.allow(trustworthiness_score=0.6), False),
        (RailOutcome.allow(trustworthiness_score=0.61), False),
    ],
)
def test_cleanlab_output_bypass_reads_rail_outcome(raw_return, expected_blocked):
    mapping_blocked = is_output_blocked(raw_return, call_cleanlab_api)
    outcome = outcome_from_output_mapping(raw_return, call_cleanlab_api)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


def test_cleanlab_action_has_no_legacy_output_mapping():
    assert getattr(call_cleanlab_api, "action_meta")["output_mapping"] is None


@pytest.mark.parametrize(
    "action_func",
    [
        ai_defense_inspect,
        clavata_check,
        call_fiddler_safety_bot,
        call_fiddler_faithfulness,
        call_activefence_api,
        call_gcp_text_moderation_api,
    ],
)
@pytest.mark.parametrize(
    ("raw_return", "expected_blocked"),
    [
        (RailOutcome.allow(), False),
        (RailOutcome.block(), True),
    ],
)
def test_vendor_block_output_bypass_reads_rail_outcome(
    action_func,
    raw_return,
    expected_blocked,
):
    mapping_blocked = is_output_blocked(raw_return, action_func)
    outcome = outcome_from_output_mapping(raw_return, action_func)

    assert mapping_blocked is expected_blocked
    assert outcome is raw_return


@pytest.mark.parametrize(
    "action_func",
    [
        ai_defense_inspect,
        clavata_check,
        call_fiddler_safety_bot,
        call_fiddler_faithfulness,
        call_activefence_api,
        call_gcp_text_moderation_api,
    ],
)
def test_vendor_block_actions_have_no_legacy_output_mapping(action_func):
    assert getattr(action_func, "action_meta")["output_mapping"] is None


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
