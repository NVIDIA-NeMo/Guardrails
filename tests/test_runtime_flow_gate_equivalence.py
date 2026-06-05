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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.crowdstrike_aidr.actions import GuardChatCompletionsResult
from nemoguardrails.library.pangea.actions import TextGuardResult
from nemoguardrails.library.trend_micro.actions import GuardResult
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.utils import TestChat

NORMAL_OUTPUT = "NORMAL OUTPUT"
REFUSAL = "I'm sorry, I can't respond to that."
ANSWER_UNKNOWN = "I don't know the answer to that."
CLEANLAB_WARNING_SUFFIX = "\nCAUTION: THIS ANSWER HAS BEEN FLAGGED AS POTENTIALLY UNTRUSTWORTHY"
RENDERED_BLOCK_MESSAGES = {
    "I will not engage in any abusive_or_harmful behavior.",
    "I will not engage in any abusive or harmful behavior.",
    "I will not engage in any self harm behavior.",
    "I will not engage with inappropriate content.",
    "I will not engage with sensitive content.",
}
INJECTION_DETECTION_REFUSAL_PREFIX = (
    "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of "
)
USER_INPUT = "hello"
RELEVANT_CHUNKS = "RELEVANT CHUNKS"
RETRIEVAL_COLANG = """
define user express greeting
  "hello"

define flow
  user express greeting
  bot express greeting

define bot express greeting
  "NORMAL OUTPUT"
"""


class ObservableOutcome(Enum):
    ALLOW = "allow"
    REFUSAL = "refusal"
    ANSWER_UNKNOWN = "answer_unknown"
    EXCEPTION = "exception"
    TRANSFORM = "transform"


class FlowDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


@dataclass(frozen=True)
class RailSpec:
    name: str
    flow: str
    direction: str
    action: str
    interpret: Callable[[Any], FlowDecision] | None = None
    model_type: str | None = None
    task: str | None = None
    output_parser: str | None = None
    rails_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class FlowEquivalenceCase:
    case_id: str
    spec: RailSpec
    raw_return: Any
    expected_observable: ObservableOutcome
    expected_decision: FlowDecision
    enable_rails_exceptions: bool = False
    context: dict[str, Any] | None = None
    expected_content: str | None = None
    output_vars: list[str] | None = None
    baseline_output_data: dict[str, Any] | None = None
    expected_output_data: dict[str, Any] | None = None


JAILBREAK_RAILS_CONFIG = {"jailbreak_detection": {"server_endpoint": "http://localhost:9999"}}
HF_CLASSIFIER_RAILS_CONFIG = {
    "hf_classifier": {
        "toxicity": {
            "engine": "local",
            "model": "placeholder",
            "blocked_labels": ["toxic"],
        }
    }
}
INJECTION_DETECTION_REJECT_RAILS_CONFIG = {"injection_detection": {"action": "reject"}}
INJECTION_DETECTION_OMIT_RAILS_CONFIG = {"injection_detection": {"action": "omit"}}
GUARDRAILS_AI_RAILS_CONFIG = {"guardrails_ai": {"validators": [{"name": "regex_match"}]}}
ACTIVEFENCE_DETAILED_THRESHOLDS = {"adult_content.general": 0.3}
GCP_DETAILED_THRESHOLDS = {"Derogatory": 0.4}


def _blocked_if_not_allowed(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if not raw_return["allowed"] else FlowDecision.ALLOW


def _blocked_if_policyai_unsafe(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return.get("assessment", "SAFE") == "UNSAFE" else FlowDecision.ALLOW


def _blocked_if_regex_match(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return["is_match"] else FlowDecision.ALLOW


def _transform_if_regex_retrieval_match(raw_return: Any) -> FlowDecision:
    return FlowDecision.TRANSFORM if raw_return["is_match"] else FlowDecision.ALLOW


def _blocked_if_true(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return else FlowDecision.ALLOW


def _transform_if_false(raw_return: Any) -> FlowDecision:
    return FlowDecision.TRANSFORM if not raw_return else FlowDecision.ALLOW


def _blocked_if_cleanlab_low_score(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return["trustworthiness_score"] < 0.6 else FlowDecision.ALLOW


def _blocked_if_attr_blocked(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return.blocked else FlowDecision.ALLOW


def _blocked_if_is_blocked(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return["is_blocked"] else FlowDecision.ALLOW


def _blocked_if_activefence_simple(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return["max_risk_score"] > 0.7 else FlowDecision.ALLOW


def _blocked_if_activefence_detailed(raw_return: Any) -> FlowDecision:
    violations = raw_return.get("violations", {})
    blocked = any(violations.get(name, 0) > threshold for name, threshold in ACTIVEFENCE_DETAILED_THRESHOLDS.items())
    return FlowDecision.BLOCK if blocked else FlowDecision.ALLOW


def _blocked_if_gcp_simple(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return["max_risk_score"] > 0.8 else FlowDecision.ALLOW


def _blocked_if_gcp_detailed(raw_return: Any) -> FlowDecision:
    violations = raw_return.get("violations", {})
    blocked = any(violations.get(name, 0) > threshold for name, threshold in GCP_DETAILED_THRESHOLDS.items())
    return FlowDecision.BLOCK if blocked else FlowDecision.ALLOW


def _blocked_if_not_valid(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if not raw_return["valid"] else FlowDecision.ALLOW


def _blocked_if_hallucination(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if raw_return["hallucination"] else FlowDecision.ALLOW


def _blocked_if_not_passed(raw_return: Any) -> FlowDecision:
    return FlowDecision.BLOCK if not raw_return["pass"] else FlowDecision.ALLOW


def _transform_if_changed_from_normal_output(raw_return: Any) -> FlowDecision:
    return FlowDecision.TRANSFORM if raw_return != NORMAL_OUTPUT else FlowDecision.ALLOW


def _transform_if_changed_from_user_input(raw_return: Any) -> FlowDecision:
    return FlowDecision.TRANSFORM if raw_return != USER_INPUT else FlowDecision.ALLOW


def _transform_if_changed_from_relevant_chunks(raw_return: Any) -> FlowDecision:
    return FlowDecision.TRANSFORM if raw_return != RELEVANT_CHUNKS else FlowDecision.ALLOW


def _blocked_or_transformed(raw_return: Any) -> FlowDecision:
    if raw_return.blocked:
        return FlowDecision.BLOCK
    if raw_return.transformed:
        return FlowDecision.TRANSFORM
    return FlowDecision.ALLOW


def _prompt_security_decision(raw_return: Any) -> FlowDecision:
    if raw_return["is_blocked"]:
        return FlowDecision.BLOCK
    if raw_return["is_modified"]:
        return FlowDecision.TRANSFORM
    return FlowDecision.ALLOW


def _autoalign_decision(raw_return: Any) -> FlowDecision:
    if raw_return["guardrails_triggered"]:
        return FlowDecision.BLOCK
    if raw_return["pii"] and raw_return["pii"]["guarded"]:
        return FlowDecision.TRANSFORM
    return FlowDecision.ALLOW


def _injection_detection_reject_decision(raw_return: Any) -> FlowDecision:
    if raw_return["is_injection"]:
        return FlowDecision.BLOCK
    if raw_return["text"] != NORMAL_OUTPUT:
        return FlowDecision.TRANSFORM
    return FlowDecision.ALLOW


def _injection_detection_transform_decision(raw_return: Any) -> FlowDecision:
    return FlowDecision.TRANSFORM if raw_return["text"] != NORMAL_OUTPUT else FlowDecision.ALLOW


SELF_CHECK_OUTPUT = RailSpec(
    name="self_check_output",
    flow="self check output",
    direction="output",
    action="self_check_output",
    task="self_check_output",
)

SELF_CHECK_INPUT = RailSpec(
    name="self_check_input",
    flow="self check input",
    direction="input",
    action="self_check_input",
    task="self_check_input",
)

SELF_CHECK_FACTS = RailSpec(
    name="self_check_facts",
    flow="self check facts",
    direction="output",
    action="self_check_facts",
    task="self_check_facts",
)

ALIGNSCORE_CHECK_FACTS = RailSpec(
    name="alignscore_check_facts",
    flow="alignscore check facts",
    direction="output",
    action="alignscore_check_facts",
)

HF_CLASSIFIER_INPUT = RailSpec(
    name="hf_classifier_input",
    flow='hf classifier check input $classifier="toxicity"',
    direction="input",
    action="hf_classifier_check_input",
    rails_config=HF_CLASSIFIER_RAILS_CONFIG,
)

HF_CLASSIFIER_OUTPUT = RailSpec(
    name="hf_classifier_output",
    flow='hf classifier check output $classifier="toxicity"',
    direction="output",
    action="hf_classifier_check_output",
    rails_config=HF_CLASSIFIER_RAILS_CONFIG,
)

HF_CLASSIFIER_RETRIEVAL = RailSpec(
    name="hf_classifier_retrieval",
    flow="hf classifier check retrieval",
    direction="retrieval",
    action="hf_classifier_check_retrieval",
    interpret=_transform_if_false,
    rails_config=HF_CLASSIFIER_RAILS_CONFIG,
)

LLAMA_GUARD_INPUT = RailSpec(
    name="llama_guard_input",
    flow="llama guard check input",
    direction="input",
    action="llama_guard_check_input",
    interpret=_blocked_if_not_allowed,
    task="llama_guard_check_input",
)

LLAMA_GUARD_OUTPUT = RailSpec(
    name="llama_guard_output",
    flow="llama guard check output",
    direction="output",
    action="llama_guard_check_output",
    interpret=_blocked_if_not_allowed,
    task="llama_guard_check_output",
)

POLICYAI_INPUT = RailSpec(
    name="policyai_input",
    flow="policyai moderation on input",
    direction="input",
    action="call_policyai_api",
    interpret=_blocked_if_policyai_unsafe,
)

POLICYAI_OUTPUT = RailSpec(
    name="policyai_output",
    flow="policyai moderation on output",
    direction="output",
    action="call_policyai_api",
    interpret=_blocked_if_policyai_unsafe,
)

REGEX_INPUT = RailSpec(
    name="regex_input",
    flow="regex check input",
    direction="input",
    action="detect_regex_pattern",
    interpret=_blocked_if_regex_match,
)

REGEX_OUTPUT = RailSpec(
    name="regex_output",
    flow="regex check output",
    direction="output",
    action="detect_regex_pattern",
    interpret=_blocked_if_regex_match,
)

REGEX_RETRIEVAL = RailSpec(
    name="regex_retrieval",
    flow="regex check retrieval",
    direction="retrieval",
    action="detect_regex_pattern",
    interpret=_transform_if_regex_retrieval_match,
)

PRIVATEAI_DETECT_INPUT = RailSpec(
    name="privateai_detect_input",
    flow="detect pii on input",
    direction="input",
    action="detect_pii",
    interpret=_blocked_if_true,
)

PRIVATEAI_DETECT_OUTPUT = RailSpec(
    name="privateai_detect_output",
    flow="detect pii on output",
    direction="output",
    action="detect_pii",
    interpret=_blocked_if_true,
)

PRIVATEAI_MASK_INPUT = RailSpec(
    name="privateai_mask_input",
    flow="mask pii on input",
    direction="input",
    action="mask_pii",
    interpret=_transform_if_changed_from_user_input,
)

PRIVATEAI_MASK_OUTPUT = RailSpec(
    name="privateai_mask_output",
    flow="mask pii on output",
    direction="output",
    action="mask_pii",
    interpret=_transform_if_changed_from_normal_output,
)

PRIVATEAI_MASK_RETRIEVAL = RailSpec(
    name="privateai_mask_retrieval",
    flow="mask pii on retrieval",
    direction="retrieval",
    action="mask_pii",
    interpret=_transform_if_changed_from_relevant_chunks,
)

GLINER_DETECT_INPUT = RailSpec(
    name="gliner_detect_input",
    flow="gliner detect pii on input",
    direction="input",
    action="gliner_detect_pii",
    interpret=_blocked_if_true,
)

GLINER_DETECT_OUTPUT = RailSpec(
    name="gliner_detect_output",
    flow="gliner detect pii on output",
    direction="output",
    action="gliner_detect_pii",
    interpret=_blocked_if_true,
)

GLINER_MASK_INPUT = RailSpec(
    name="gliner_mask_input",
    flow="gliner mask pii on input",
    direction="input",
    action="gliner_mask_pii",
    interpret=_transform_if_changed_from_user_input,
)

GLINER_MASK_OUTPUT = RailSpec(
    name="gliner_mask_output",
    flow="gliner mask pii on output",
    direction="output",
    action="gliner_mask_pii",
    interpret=_transform_if_changed_from_normal_output,
)

GLINER_MASK_RETRIEVAL = RailSpec(
    name="gliner_mask_retrieval",
    flow="gliner mask pii on retrieval",
    direction="retrieval",
    action="gliner_mask_pii",
    interpret=_transform_if_changed_from_relevant_chunks,
)

SENSITIVE_DATA_DETECT_INPUT = RailSpec(
    name="sensitive_data_detect_input",
    flow="detect sensitive data on input",
    direction="input",
    action="detect_sensitive_data",
    interpret=_blocked_if_true,
)

SENSITIVE_DATA_DETECT_OUTPUT = RailSpec(
    name="sensitive_data_detect_output",
    flow="detect sensitive data on output",
    direction="output",
    action="detect_sensitive_data",
    interpret=_blocked_if_true,
)

SENSITIVE_DATA_MASK_INPUT = RailSpec(
    name="sensitive_data_mask_input",
    flow="mask sensitive data on input",
    direction="input",
    action="mask_sensitive_data",
    interpret=_transform_if_changed_from_user_input,
)

SENSITIVE_DATA_MASK_OUTPUT = RailSpec(
    name="sensitive_data_mask_output",
    flow="mask sensitive data on output",
    direction="output",
    action="mask_sensitive_data",
    interpret=_transform_if_changed_from_normal_output,
)

SENSITIVE_DATA_MASK_RETRIEVAL = RailSpec(
    name="sensitive_data_mask_retrieval",
    flow="mask sensitive data on retrieval",
    direction="retrieval",
    action="mask_sensitive_data",
    interpret=_transform_if_changed_from_relevant_chunks,
)

CONTENT_SAFETY_INPUT = RailSpec(
    name="content_safety_input",
    flow="content safety check input $model=content_safety",
    direction="input",
    action="content_safety_check_input",
    model_type="content_safety",
    task="content_safety_check_input $model=content_safety",
    output_parser="nemoguard_parse_prompt_safety",
)

CONTENT_SAFETY_OUTPUT = RailSpec(
    name="content_safety_output",
    flow="content safety check output $model=content_safety",
    direction="output",
    action="content_safety_check_output",
    model_type="content_safety",
    task="content_safety_check_output $model=content_safety",
    output_parser="nemoguard_parse_prompt_safety",
)

CONTENT_SAFETY_INPUT = RailSpec(
    name="content_safety_input",
    flow="content safety check input $model=content_safety",
    direction="input",
    action="content_safety_check_input",
    model_type="content_safety",
    task="content_safety_check_input $model=content_safety",
    output_parser="nemoguard_parse_prompt_safety",
)

TOPIC_SAFETY_INPUT = RailSpec(
    name="topic_safety_input",
    flow="topic safety check input $model=topic_control",
    direction="input",
    action="topic_safety_check_input",
    model_type="topic_control",
    task="topic_safety_check_input $model=topic_control",
)

JAILBREAK_HEURISTICS_INPUT = RailSpec(
    name="jailbreak_heuristics_input",
    flow="jailbreak detection heuristics",
    direction="input",
    action="jailbreak_detection_heuristics",
    rails_config=JAILBREAK_RAILS_CONFIG,
)

JAILBREAK_MODEL_INPUT = RailSpec(
    name="jailbreak_model_input",
    flow="jailbreak detection model",
    direction="input",
    action="jailbreak_detection_model",
    rails_config=JAILBREAK_RAILS_CONFIG,
)

PANGEA_INPUT = RailSpec(
    name="pangea_input",
    flow="pangea ai guard input",
    direction="input",
    action="pangea_ai_guard",
    interpret=_blocked_or_transformed,
)

PANGEA_OUTPUT = RailSpec(
    name="pangea_output",
    flow="pangea ai guard output",
    direction="output",
    action="pangea_ai_guard",
    interpret=_blocked_or_transformed,
)

CROWDSTRIKE_AIDR_INPUT = RailSpec(
    name="crowdstrike_aidr_input",
    flow="crowdstrike aidr guard input",
    direction="input",
    action="crowdstrike_aidr_guard",
    interpret=_blocked_or_transformed,
)

CROWDSTRIKE_AIDR_OUTPUT = RailSpec(
    name="crowdstrike_aidr_output",
    flow="crowdstrike aidr guard output",
    direction="output",
    action="crowdstrike_aidr_guard",
    interpret=_blocked_or_transformed,
)

PROMPT_SECURITY_INPUT = RailSpec(
    name="prompt_security_input",
    flow="protect prompt",
    direction="input",
    action="protect_text",
    interpret=_prompt_security_decision,
)

PROMPT_SECURITY_OUTPUT = RailSpec(
    name="prompt_security_output",
    flow="protect response",
    direction="output",
    action="protect_text",
    interpret=_prompt_security_decision,
)

AUTOALIGN_INPUT = RailSpec(
    name="autoalign_input",
    flow="autoalign check input",
    direction="input",
    action="autoalign_input_api",
    interpret=_autoalign_decision,
)

AUTOALIGN_OUTPUT = RailSpec(
    name="autoalign_output",
    flow="autoalign check output",
    direction="output",
    action="autoalign_output_api",
    interpret=_autoalign_decision,
)

INJECTION_DETECTION_REJECT = RailSpec(
    name="injection_detection_reject",
    flow="injection detection",
    direction="output",
    action="injection_detection",
    interpret=_injection_detection_reject_decision,
    rails_config=INJECTION_DETECTION_REJECT_RAILS_CONFIG,
)

INJECTION_DETECTION_OMIT = RailSpec(
    name="injection_detection_omit",
    flow="injection detection",
    direction="output",
    action="injection_detection",
    interpret=_injection_detection_transform_decision,
    rails_config=INJECTION_DETECTION_OMIT_RAILS_CONFIG,
)

CLAVATA_INPUT = RailSpec(
    name="clavata_input",
    flow="clavata check input",
    direction="input",
    action="ClavataCheckAction",
    interpret=_blocked_if_true,
)

CLAVATA_OUTPUT = RailSpec(
    name="clavata_output",
    flow="clavata check output",
    direction="output",
    action="ClavataCheckAction",
    interpret=_blocked_if_true,
)

FIDDLER_USER_SAFETY = RailSpec(
    name="fiddler_user_safety",
    flow="fiddler user safety",
    direction="input",
    action="call fiddler safety on user message",
    interpret=_blocked_if_true,
)

FIDDLER_BOT_SAFETY = RailSpec(
    name="fiddler_bot_safety",
    flow="fiddler bot safety",
    direction="output",
    action="call fiddler safety on bot message",
    interpret=_blocked_if_true,
)

FIDDLER_BOT_FAITHFULNESS = RailSpec(
    name="fiddler_bot_faithfulness",
    flow="fiddler bot faithfulness",
    direction="output",
    action="call fiddler faithfulness",
    interpret=_blocked_if_true,
)

ACTIVEFENCE_INPUT = RailSpec(
    name="activefence_input",
    flow="activefence moderation on input",
    direction="input",
    action="call_activefence_api",
    interpret=_blocked_if_activefence_simple,
)

ACTIVEFENCE_OUTPUT = RailSpec(
    name="activefence_output",
    flow="activefence moderation on output",
    direction="output",
    action="call_activefence_api",
    interpret=_blocked_if_activefence_simple,
)

ACTIVEFENCE_INPUT_DETAILED = RailSpec(
    name="activefence_input_detailed",
    flow="activefence moderation on input detailed",
    direction="input",
    action="call_activefence_api",
    interpret=_blocked_if_activefence_detailed,
)

GCP_MODERATION_OUTPUT = RailSpec(
    name="gcp_moderation_output",
    flow="gcpnlp moderation",
    direction="output",
    action="call gcpnlp api",
    interpret=_blocked_if_gcp_simple,
)

GCP_MODERATION_OUTPUT_DETAILED = RailSpec(
    name="gcp_moderation_output_detailed",
    flow="gcpnlp moderation detailed",
    direction="output",
    action="call gcpnlp api",
    interpret=_blocked_if_gcp_detailed,
)

GUARDRAILS_AI_INPUT = RailSpec(
    name="guardrails_ai_input",
    flow='guardrailsai check input $validator="regex_match"',
    direction="input",
    action="validate_guardrails_ai_input",
    interpret=_blocked_if_not_valid,
    rails_config=GUARDRAILS_AI_RAILS_CONFIG,
)

GUARDRAILS_AI_OUTPUT = RailSpec(
    name="guardrails_ai_output",
    flow='guardrailsai check output $validator="regex_match"',
    direction="output",
    action="validate_guardrails_ai_output",
    interpret=_blocked_if_not_valid,
    rails_config=GUARDRAILS_AI_RAILS_CONFIG,
)

PATRONUS_LYNX_OUTPUT = RailSpec(
    name="patronus_lynx_output",
    flow="patronus lynx check output hallucination",
    direction="output",
    action="patronus_lynx_check_output_hallucination",
    interpret=_blocked_if_hallucination,
    task="patronus_lynx_check_output_hallucination",
)

PATRONUS_API_OUTPUT = RailSpec(
    name="patronus_api_output",
    flow="patronus api check output",
    direction="output",
    action="patronus_api_check_output",
    interpret=_blocked_if_not_passed,
)

SELF_CHECK_HALLUCINATION = RailSpec(
    name="self_check_hallucination",
    flow="self check hallucination",
    direction="output",
    action="self_check_hallucination",
    interpret=_blocked_if_true,
)

TREND_MICRO_INPUT = RailSpec(
    name="trend_micro_input",
    flow="trend ai guard input",
    direction="input",
    action="trend_ai_guard",
    interpret=_blocked_if_attr_blocked,
)

TREND_MICRO_OUTPUT = RailSpec(
    name="trend_micro_output",
    flow="trend ai guard output",
    direction="output",
    action="trend_ai_guard",
    interpret=_blocked_if_attr_blocked,
)

CLEANLAB_OUTPUT = RailSpec(
    name="cleanlab_output",
    flow="cleanlab trustworthiness",
    direction="output",
    action="call cleanlab api",
    interpret=_blocked_if_cleanlab_low_score,
)

AI_DEFENSE_INPUT = RailSpec(
    name="ai_defense_input",
    flow="ai defense inspect prompt",
    direction="input",
    action="ai_defense_inspect",
    interpret=_blocked_if_is_blocked,
)

AI_DEFENSE_OUTPUT = RailSpec(
    name="ai_defense_output",
    flow="ai defense inspect response",
    direction="output",
    action="ai_defense_inspect",
    interpret=_blocked_if_is_blocked,
)


def _case(
    case_id: str,
    spec: RailSpec,
    raw_return: Any,
    expected_observable: ObservableOutcome,
    expected_decision: FlowDecision,
    *,
    enable_rails_exceptions: bool = False,
    context: dict[str, Any] | None = None,
    expected_content: str | None = None,
    output_vars: list[str] | None = None,
    baseline_output_data: dict[str, Any] | None = None,
    expected_output_data: dict[str, Any] | None = None,
) -> FlowEquivalenceCase:
    return FlowEquivalenceCase(
        case_id=case_id,
        spec=spec,
        raw_return=raw_return,
        expected_observable=expected_observable,
        expected_decision=expected_decision,
        enable_rails_exceptions=enable_rails_exceptions,
        context=context,
        expected_content=expected_content,
        output_vars=output_vars,
        baseline_output_data=baseline_output_data,
        expected_output_data=expected_output_data,
    )


def _rail_outcome_cases(
    spec: RailSpec,
    *,
    allow_return: RailOutcome | None = None,
    block_return: RailOutcome | None = None,
    include_exception_case: bool = False,
) -> list[FlowEquivalenceCase]:
    allow_return = allow_return or RailOutcome.allow()
    block_return = block_return or RailOutcome.block()
    cases = [
        _case(
            f"{spec.name}_allows_outcome_allow",
            spec,
            allow_return,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
        ),
        _case(
            f"{spec.name}_blocks_outcome_block",
            spec,
            block_return,
            ObservableOutcome.REFUSAL,
            FlowDecision.BLOCK,
        ),
    ]
    if include_exception_case:
        cases.append(
            _case(
                f"{spec.name}_blocks_outcome_block_exception",
                spec,
                block_return,
                ObservableOutcome.EXCEPTION,
                FlowDecision.BLOCK,
                enable_rails_exceptions=True,
            )
        )
    return cases


def _boolean_allowed_cases(
    spec: RailSpec,
    *,
    include_exception_case: bool = False,
) -> list[FlowEquivalenceCase]:
    cases = [
        _case(
            f"{spec.name}_allows_true",
            spec,
            True,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
        ),
        _case(
            f"{spec.name}_blocks_false",
            spec,
            False,
            ObservableOutcome.REFUSAL,
            FlowDecision.BLOCK,
        ),
    ]
    if include_exception_case:
        cases.append(
            _case(
                f"{spec.name}_blocks_false_exception",
                spec,
                False,
                ObservableOutcome.EXCEPTION,
                FlowDecision.BLOCK,
                enable_rails_exceptions=True,
            )
        )
    return cases


def _boolean_flag_cases(
    spec: RailSpec,
    *,
    block_observable: ObservableOutcome,
) -> list[FlowEquivalenceCase]:
    return [
        _case(
            f"{spec.name}_allows_false",
            spec,
            False,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
        ),
        _case(
            f"{spec.name}_blocks_true",
            spec,
            True,
            block_observable,
            FlowDecision.BLOCK,
        ),
    ]


def _output_transform_cases(spec: RailSpec) -> list[FlowEquivalenceCase]:
    transformed_output = f"{spec.name} masked output"
    return [
        _case(
            f"{spec.name}_allows_unchanged_output",
            spec,
            NORMAL_OUTPUT,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
            expected_content=NORMAL_OUTPUT,
        ),
        _case(
            f"{spec.name}_transforms_changed_output",
            spec,
            transformed_output,
            ObservableOutcome.TRANSFORM,
            FlowDecision.TRANSFORM,
            expected_content=transformed_output,
        ),
    ]


def _output_var_transform_cases(
    spec: RailSpec,
    *,
    output_var: str,
    original_value: str,
    transformed_value: str,
    context: dict[str, Any] | None = None,
) -> list[FlowEquivalenceCase]:
    return [
        _case(
            f"{spec.name}_allows_unchanged_{output_var}",
            spec,
            original_value,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
            context=context,
            output_vars=[output_var],
            baseline_output_data={output_var: original_value},
            expected_output_data={output_var: original_value},
        ),
        _case(
            f"{spec.name}_transforms_changed_{output_var}",
            spec,
            transformed_value,
            ObservableOutcome.TRANSFORM,
            FlowDecision.TRANSFORM,
            context=context,
            output_vars=[output_var],
            baseline_output_data={output_var: original_value},
            expected_output_data={output_var: transformed_value},
        ),
    ]


def _guard_result_cases(
    spec: RailSpec,
    *,
    allow_return: Any,
    block_return: Any,
    transform_return: Any,
    block_and_transform_return: Any,
    transformed_value: str,
) -> list[FlowEquivalenceCase]:
    if spec.direction == "output":
        transform_kwargs: dict[str, Any] = {"expected_content": transformed_value}
        allow_kwargs: dict[str, Any] = {"expected_content": NORMAL_OUTPUT}
    else:
        transform_kwargs = {
            "output_vars": ["user_message"],
            "baseline_output_data": {"user_message": USER_INPUT},
            "expected_output_data": {"user_message": transformed_value},
        }
        allow_kwargs = {
            "output_vars": ["user_message"],
            "baseline_output_data": {"user_message": USER_INPUT},
            "expected_output_data": {"user_message": USER_INPUT},
        }

    return [
        _case(
            f"{spec.name}_allows",
            spec,
            allow_return,
            ObservableOutcome.ALLOW,
            FlowDecision.ALLOW,
            **allow_kwargs,
        ),
        _case(
            f"{spec.name}_blocks",
            spec,
            block_return,
            ObservableOutcome.ANSWER_UNKNOWN,
            FlowDecision.BLOCK,
        ),
        _case(
            f"{spec.name}_blocks_exception",
            spec,
            block_return,
            ObservableOutcome.EXCEPTION,
            FlowDecision.BLOCK,
            enable_rails_exceptions=True,
        ),
        _case(
            f"{spec.name}_transforms",
            spec,
            transform_return,
            ObservableOutcome.TRANSFORM,
            FlowDecision.TRANSFORM,
            **transform_kwargs,
        ),
        _case(
            f"{spec.name}_block_wins_over_transform",
            spec,
            block_and_transform_return,
            ObservableOutcome.ANSWER_UNKNOWN,
            FlowDecision.BLOCK,
        ),
    ]


def _text_guard_result(*, blocked: bool, transformed: bool, user_message: str, bot_message: str) -> TextGuardResult:
    return TextGuardResult(
        blocked=blocked,
        transformed=transformed,
        user_message=user_message,
        bot_message=bot_message,
    )


def _crowdstrike_result(
    *,
    blocked: bool,
    transformed: bool,
    user_message: str,
    bot_message: str,
) -> GuardChatCompletionsResult:
    return GuardChatCompletionsResult(
        blocked=blocked,
        transformed=transformed,
        user_message=user_message,
        bot_message=bot_message,
    )


def _prompt_security_result(*, blocked: bool, modified: bool, text: str | None = None) -> dict[str, Any]:
    return {
        "is_blocked": blocked,
        "is_modified": modified,
        "modified_text": text,
    }


def _autoalign_result(
    *,
    guardrails_triggered: bool,
    pii_guarded: bool,
    pii_response: str,
) -> dict[str, Any]:
    return {
        "guardrails_triggered": guardrails_triggered,
        "combined_response": "AutoAlign guardrail response",
        "pii": {"guarded": pii_guarded, "response": pii_response},
    }


def _injection_detection_result(
    *, is_injection: bool, text: str, detections: list[str] | None = None
) -> dict[str, Any]:
    return {
        "is_injection": is_injection,
        "text": text,
        "detections": detections or [],
    }


def _trend_micro_result(*, action: Literal["Allow", "Block"]) -> GuardResult:
    return GuardResult(action=action, reason=f"{action} reason")


def _cleanlab_result(score: float) -> dict[str, float]:
    return {"trustworthiness_score": score}


def _risk_result(max_risk_score: float, violations: dict[str, float] | None = None) -> dict[str, Any]:
    return {"max_risk_score": max_risk_score, "violations": violations or {}}


FIXTURES = [
    _case(
        "self_check_output_allows_true",
        SELF_CHECK_OUTPUT,
        True,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "self_check_output_blocks_false",
        SELF_CHECK_OUTPUT,
        False,
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "self_check_input_allows_true",
        SELF_CHECK_INPUT,
        True,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "self_check_input_blocks_false",
        SELF_CHECK_INPUT,
        False,
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "self_check_input_blocks_false_exception",
        SELF_CHECK_INPUT,
        False,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "self_check_facts_blocks_below_threshold",
        SELF_CHECK_FACTS,
        0.49,
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
        context={"check_facts": True},
    ),
    _case(
        "self_check_facts_allows_at_threshold",
        SELF_CHECK_FACTS,
        0.5,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"check_facts": True},
    ),
    _case(
        "self_check_facts_allows_above_threshold",
        SELF_CHECK_FACTS,
        0.51,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"check_facts": True},
    ),
    _case(
        "self_check_facts_blocks_below_threshold_exception",
        SELF_CHECK_FACTS,
        0.49,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
        context={"check_facts": True},
    ),
    _case(
        "alignscore_check_facts_blocks_below_threshold",
        ALIGNSCORE_CHECK_FACTS,
        0.49,
        ObservableOutcome.ANSWER_UNKNOWN,
        FlowDecision.BLOCK,
        context={"check_facts": True},
    ),
    _case(
        "alignscore_check_facts_allows_at_threshold",
        ALIGNSCORE_CHECK_FACTS,
        0.5,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"check_facts": True},
    ),
    _case(
        "alignscore_check_facts_allows_above_threshold",
        ALIGNSCORE_CHECK_FACTS,
        0.51,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"check_facts": True},
    ),
    _case(
        "alignscore_check_facts_blocks_below_threshold_exception",
        ALIGNSCORE_CHECK_FACTS,
        0.49,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
        context={"check_facts": True},
    ),
    *_boolean_allowed_cases(
        HF_CLASSIFIER_INPUT,
        include_exception_case=True,
    ),
    *_boolean_allowed_cases(
        HF_CLASSIFIER_OUTPUT,
        include_exception_case=True,
    ),
    _case(
        "hf_classifier_retrieval_allows_true",
        HF_CLASSIFIER_RETRIEVAL,
        True,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"relevant_chunks": RELEVANT_CHUNKS},
        output_vars=["relevant_chunks"],
        baseline_output_data={"relevant_chunks": RELEVANT_CHUNKS},
        expected_output_data={"relevant_chunks": RELEVANT_CHUNKS},
    ),
    _case(
        "hf_classifier_retrieval_transforms_false",
        HF_CLASSIFIER_RETRIEVAL,
        False,
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        context={"relevant_chunks": RELEVANT_CHUNKS},
        output_vars=["relevant_chunks"],
        baseline_output_data={"relevant_chunks": RELEVANT_CHUNKS},
        expected_output_data={"relevant_chunks": ""},
    ),
    _case(
        "llama_guard_input_allows_allowed_true",
        LLAMA_GUARD_INPUT,
        {"allowed": True, "policy_violations": None},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "llama_guard_input_blocks_allowed_false",
        LLAMA_GUARD_INPUT,
        {"allowed": False, "policy_violations": ["S1"]},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "llama_guard_input_blocks_allowed_false_exception",
        LLAMA_GUARD_INPUT,
        {"allowed": False, "policy_violations": ["S1"]},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "llama_guard_output_allows_allowed_true",
        LLAMA_GUARD_OUTPUT,
        {"allowed": True, "policy_violations": None},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "llama_guard_output_blocks_allowed_false",
        LLAMA_GUARD_OUTPUT,
        {"allowed": False, "policy_violations": ["S1"]},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "llama_guard_output_blocks_allowed_false_exception",
        LLAMA_GUARD_OUTPUT,
        {"allowed": False, "policy_violations": ["S1"]},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "policyai_input_allows_safe",
        POLICYAI_INPUT,
        {"assessment": "SAFE", "category": "Safe"},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "policyai_input_blocks_unsafe",
        POLICYAI_INPUT,
        {"assessment": "UNSAFE", "category": "violence"},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "policyai_input_blocks_unsafe_exception",
        POLICYAI_INPUT,
        {"assessment": "UNSAFE", "category": "violence"},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "policyai_output_allows_safe",
        POLICYAI_OUTPUT,
        {"assessment": "SAFE", "category": "Safe"},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "policyai_output_blocks_unsafe",
        POLICYAI_OUTPUT,
        {"assessment": "UNSAFE", "category": "violence"},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "policyai_output_blocks_unsafe_exception",
        POLICYAI_OUTPUT,
        {"assessment": "UNSAFE", "category": "violence"},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "regex_input_allows_no_match",
        REGEX_INPUT,
        {"is_match": False, "text": "hello", "detections": []},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "regex_input_blocks_match",
        REGEX_INPUT,
        {"is_match": True, "text": "secret", "detections": ["secret"]},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "regex_output_allows_no_match",
        REGEX_OUTPUT,
        {"is_match": False, "text": "hello", "detections": []},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "regex_output_blocks_match",
        REGEX_OUTPUT,
        {"is_match": True, "text": "secret", "detections": ["secret"]},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "regex_retrieval_allows_no_match",
        REGEX_RETRIEVAL,
        {"is_match": False, "text": RELEVANT_CHUNKS, "detections": []},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"relevant_chunks": RELEVANT_CHUNKS},
        output_vars=["relevant_chunks"],
        baseline_output_data={"relevant_chunks": RELEVANT_CHUNKS},
        expected_output_data={"relevant_chunks": RELEVANT_CHUNKS},
    ),
    _case(
        "regex_retrieval_transforms_match",
        REGEX_RETRIEVAL,
        {"is_match": True, "text": RELEVANT_CHUNKS, "detections": ["secret"]},
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        context={"relevant_chunks": RELEVANT_CHUNKS},
        output_vars=["relevant_chunks"],
        baseline_output_data={"relevant_chunks": RELEVANT_CHUNKS},
        expected_output_data={"relevant_chunks": ""},
    ),
    *_boolean_flag_cases(
        PRIVATEAI_DETECT_INPUT,
        block_observable=ObservableOutcome.ANSWER_UNKNOWN,
    ),
    *_boolean_flag_cases(
        PRIVATEAI_DETECT_OUTPUT,
        block_observable=ObservableOutcome.ANSWER_UNKNOWN,
    ),
    *_boolean_flag_cases(
        GLINER_DETECT_INPUT,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    *_boolean_flag_cases(
        GLINER_DETECT_OUTPUT,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    *_boolean_flag_cases(
        SENSITIVE_DATA_DETECT_INPUT,
        block_observable=ObservableOutcome.ANSWER_UNKNOWN,
    ),
    *_boolean_flag_cases(
        SENSITIVE_DATA_DETECT_OUTPUT,
        block_observable=ObservableOutcome.ANSWER_UNKNOWN,
    ),
    *_output_transform_cases(PRIVATEAI_MASK_OUTPUT),
    *_output_transform_cases(GLINER_MASK_OUTPUT),
    *_output_transform_cases(SENSITIVE_DATA_MASK_OUTPUT),
    *_output_var_transform_cases(
        PRIVATEAI_MASK_INPUT,
        output_var="user_message",
        original_value=USER_INPUT,
        transformed_value="privateai masked input",
    ),
    *_output_var_transform_cases(
        GLINER_MASK_INPUT,
        output_var="user_message",
        original_value=USER_INPUT,
        transformed_value="gliner masked input",
    ),
    *_output_var_transform_cases(
        SENSITIVE_DATA_MASK_INPUT,
        output_var="user_message",
        original_value=USER_INPUT,
        transformed_value="sensitive data masked input",
    ),
    *_output_var_transform_cases(
        PRIVATEAI_MASK_RETRIEVAL,
        output_var="relevant_chunks",
        original_value=RELEVANT_CHUNKS,
        transformed_value="privateai masked chunks",
        context={"relevant_chunks": RELEVANT_CHUNKS},
    ),
    *_output_var_transform_cases(
        GLINER_MASK_RETRIEVAL,
        output_var="relevant_chunks",
        original_value=RELEVANT_CHUNKS,
        transformed_value="gliner masked chunks",
        context={"relevant_chunks": RELEVANT_CHUNKS},
    ),
    *_output_var_transform_cases(
        SENSITIVE_DATA_MASK_RETRIEVAL,
        output_var="relevant_chunks",
        original_value=RELEVANT_CHUNKS,
        transformed_value="sensitive data masked chunks",
        context={"relevant_chunks": RELEVANT_CHUNKS},
    ),
    *_guard_result_cases(
        PANGEA_INPUT,
        allow_return=_text_guard_result(
            blocked=False,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        block_return=_text_guard_result(
            blocked=True,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        transform_return=_text_guard_result(
            blocked=False,
            transformed=True,
            user_message="pangea transformed input",
            bot_message=NORMAL_OUTPUT,
        ),
        block_and_transform_return=_text_guard_result(
            blocked=True,
            transformed=True,
            user_message="pangea transformed input",
            bot_message=NORMAL_OUTPUT,
        ),
        transformed_value="pangea transformed input",
    ),
    *_guard_result_cases(
        PANGEA_OUTPUT,
        allow_return=_text_guard_result(
            blocked=False,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        block_return=_text_guard_result(
            blocked=True,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        transform_return=_text_guard_result(
            blocked=False,
            transformed=True,
            user_message=USER_INPUT,
            bot_message="pangea transformed output",
        ),
        block_and_transform_return=_text_guard_result(
            blocked=True,
            transformed=True,
            user_message=USER_INPUT,
            bot_message="pangea transformed output",
        ),
        transformed_value="pangea transformed output",
    ),
    *_guard_result_cases(
        CROWDSTRIKE_AIDR_INPUT,
        allow_return=_crowdstrike_result(
            blocked=False,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        block_return=_crowdstrike_result(
            blocked=True,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        transform_return=_crowdstrike_result(
            blocked=False,
            transformed=True,
            user_message="crowdstrike aidr transformed input",
            bot_message=NORMAL_OUTPUT,
        ),
        block_and_transform_return=_crowdstrike_result(
            blocked=True,
            transformed=True,
            user_message="crowdstrike aidr transformed input",
            bot_message=NORMAL_OUTPUT,
        ),
        transformed_value="crowdstrike aidr transformed input",
    ),
    *_guard_result_cases(
        CROWDSTRIKE_AIDR_OUTPUT,
        allow_return=_crowdstrike_result(
            blocked=False,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        block_return=_crowdstrike_result(
            blocked=True,
            transformed=False,
            user_message=USER_INPUT,
            bot_message=NORMAL_OUTPUT,
        ),
        transform_return=_crowdstrike_result(
            blocked=False,
            transformed=True,
            user_message=USER_INPUT,
            bot_message="crowdstrike aidr transformed output",
        ),
        block_and_transform_return=_crowdstrike_result(
            blocked=True,
            transformed=True,
            user_message=USER_INPUT,
            bot_message="crowdstrike aidr transformed output",
        ),
        transformed_value="crowdstrike aidr transformed output",
    ),
    *_guard_result_cases(
        PROMPT_SECURITY_INPUT,
        allow_return=_prompt_security_result(blocked=False, modified=False),
        block_return=_prompt_security_result(blocked=True, modified=False),
        transform_return=_prompt_security_result(
            blocked=False,
            modified=True,
            text="prompt security transformed input",
        ),
        block_and_transform_return=_prompt_security_result(
            blocked=True,
            modified=True,
            text="prompt security transformed input",
        ),
        transformed_value="prompt security transformed input",
    ),
    *_guard_result_cases(
        PROMPT_SECURITY_OUTPUT,
        allow_return=_prompt_security_result(blocked=False, modified=False),
        block_return=_prompt_security_result(blocked=True, modified=False),
        transform_return=_prompt_security_result(
            blocked=False,
            modified=True,
            text="prompt security transformed output",
        ),
        block_and_transform_return=_prompt_security_result(
            blocked=True,
            modified=True,
            text="prompt security transformed output",
        ),
        transformed_value="prompt security transformed output",
    ),
    _case(
        "autoalign_input_allows",
        AUTOALIGN_INPUT,
        _autoalign_result(
            guardrails_triggered=False,
            pii_guarded=False,
            pii_response=USER_INPUT,
        ),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        output_vars=["user_message"],
        baseline_output_data={"user_message": USER_INPUT},
        expected_output_data={"user_message": USER_INPUT},
    ),
    _case(
        "autoalign_input_blocks_guardrail",
        AUTOALIGN_INPUT,
        _autoalign_result(
            guardrails_triggered=True,
            pii_guarded=False,
            pii_response=USER_INPUT,
        ),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "autoalign_input_blocks_guardrail_exception",
        AUTOALIGN_INPUT,
        _autoalign_result(
            guardrails_triggered=True,
            pii_guarded=False,
            pii_response=USER_INPUT,
        ),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "autoalign_input_transforms_pii",
        AUTOALIGN_INPUT,
        _autoalign_result(
            guardrails_triggered=False,
            pii_guarded=True,
            pii_response="autoalign transformed input",
        ),
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        output_vars=["user_message"],
        baseline_output_data={"user_message": USER_INPUT},
        expected_output_data={"user_message": "autoalign transformed input"},
    ),
    _case(
        "autoalign_input_block_wins_over_pii_transform",
        AUTOALIGN_INPUT,
        _autoalign_result(
            guardrails_triggered=True,
            pii_guarded=True,
            pii_response="autoalign transformed input",
        ),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "autoalign_output_allows",
        AUTOALIGN_OUTPUT,
        _autoalign_result(
            guardrails_triggered=False,
            pii_guarded=False,
            pii_response=NORMAL_OUTPUT,
        ),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        expected_content=NORMAL_OUTPUT,
    ),
    _case(
        "autoalign_output_blocks_guardrail",
        AUTOALIGN_OUTPUT,
        _autoalign_result(
            guardrails_triggered=True,
            pii_guarded=False,
            pii_response=NORMAL_OUTPUT,
        ),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "autoalign_output_blocks_guardrail_exception",
        AUTOALIGN_OUTPUT,
        _autoalign_result(
            guardrails_triggered=True,
            pii_guarded=False,
            pii_response=NORMAL_OUTPUT,
        ),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "autoalign_output_transforms_pii",
        AUTOALIGN_OUTPUT,
        _autoalign_result(
            guardrails_triggered=False,
            pii_guarded=True,
            pii_response="autoalign transformed output",
        ),
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        expected_content="autoalign transformed output",
    ),
    _case(
        "autoalign_output_block_wins_over_pii_transform",
        AUTOALIGN_OUTPUT,
        _autoalign_result(
            guardrails_triggered=True,
            pii_guarded=True,
            pii_response="autoalign transformed output",
        ),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "injection_detection_reject_allows_unchanged_output",
        INJECTION_DETECTION_REJECT,
        _injection_detection_result(is_injection=False, text=NORMAL_OUTPUT),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        expected_content=NORMAL_OUTPUT,
    ),
    _case(
        "injection_detection_reject_transforms_non_injection_text",
        INJECTION_DETECTION_REJECT,
        _injection_detection_result(is_injection=False, text="injection detection transformed output"),
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        expected_content="injection detection transformed output",
    ),
    _case(
        "injection_detection_reject_blocks_injection",
        INJECTION_DETECTION_REJECT,
        _injection_detection_result(is_injection=True, text=NORMAL_OUTPUT, detections=["sqli"]),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
        expected_content=f"{INJECTION_DETECTION_REFUSAL_PREFIX}sqli.",
    ),
    _case(
        "injection_detection_reject_blocks_injection_exception",
        INJECTION_DETECTION_REJECT,
        _injection_detection_result(is_injection=True, text=NORMAL_OUTPUT, detections=["sqli"]),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "injection_detection_reject_block_wins_over_text_transform",
        INJECTION_DETECTION_REJECT,
        _injection_detection_result(
            is_injection=True,
            text="injection detection transformed output",
            detections=["sqli"],
        ),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
        expected_content=f"{INJECTION_DETECTION_REFUSAL_PREFIX}sqli.",
    ),
    _case(
        "injection_detection_omit_allows_unchanged_output",
        INJECTION_DETECTION_OMIT,
        _injection_detection_result(is_injection=False, text=NORMAL_OUTPUT),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        expected_content=NORMAL_OUTPUT,
    ),
    _case(
        "injection_detection_omit_transforms_non_injection_text",
        INJECTION_DETECTION_OMIT,
        _injection_detection_result(is_injection=False, text="injection detection normalized output"),
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        expected_content="injection detection normalized output",
    ),
    _case(
        "injection_detection_omit_transforms_injection_text",
        INJECTION_DETECTION_OMIT,
        _injection_detection_result(
            is_injection=True,
            text="injection detection omitted output",
            detections=["sqli"],
        ),
        ObservableOutcome.TRANSFORM,
        FlowDecision.TRANSFORM,
        expected_content="injection detection omitted output",
    ),
    _case(
        "trend_micro_input_allows_allow_action",
        TREND_MICRO_INPUT,
        _trend_micro_result(action="Allow"),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "trend_micro_input_blocks_block_action",
        TREND_MICRO_INPUT,
        _trend_micro_result(action="Block"),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "trend_micro_input_blocks_block_action_exception",
        TREND_MICRO_INPUT,
        _trend_micro_result(action="Block"),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "trend_micro_output_allows_allow_action",
        TREND_MICRO_OUTPUT,
        _trend_micro_result(action="Allow"),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "trend_micro_output_blocks_block_action",
        TREND_MICRO_OUTPUT,
        _trend_micro_result(action="Block"),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "trend_micro_output_blocks_block_action_exception",
        TREND_MICRO_OUTPUT,
        _trend_micro_result(action="Block"),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "cleanlab_output_blocks_below_threshold",
        CLEANLAB_OUTPUT,
        _cleanlab_result(0.59),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
        expected_content=f"{NORMAL_OUTPUT} {CLEANLAB_WARNING_SUFFIX}",
    ),
    _case(
        "cleanlab_output_allows_at_threshold",
        CLEANLAB_OUTPUT,
        _cleanlab_result(0.6),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "cleanlab_output_allows_above_threshold",
        CLEANLAB_OUTPUT,
        _cleanlab_result(0.61),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "cleanlab_output_blocks_below_threshold_exception",
        CLEANLAB_OUTPUT,
        _cleanlab_result(0.59),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "ai_defense_input_allows_false",
        AI_DEFENSE_INPUT,
        {"is_blocked": False},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "ai_defense_input_blocks_true",
        AI_DEFENSE_INPUT,
        {"is_blocked": True},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "ai_defense_output_allows_false",
        AI_DEFENSE_OUTPUT,
        {"is_blocked": False},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "ai_defense_output_blocks_true",
        AI_DEFENSE_OUTPUT,
        {"is_blocked": True},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "ai_defense_input_blocks_true_exception",
        AI_DEFENSE_INPUT,
        {"is_blocked": True},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "ai_defense_output_blocks_true_exception",
        AI_DEFENSE_OUTPUT,
        {"is_blocked": True},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    *_boolean_flag_cases(
        CLAVATA_INPUT,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    *_boolean_flag_cases(
        CLAVATA_OUTPUT,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    _case(
        "clavata_input_blocks_true_exception",
        CLAVATA_INPUT,
        True,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "clavata_output_blocks_true_exception",
        CLAVATA_OUTPUT,
        True,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    *_boolean_flag_cases(
        FIDDLER_USER_SAFETY,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    *_boolean_flag_cases(
        FIDDLER_BOT_SAFETY,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    *_boolean_flag_cases(
        FIDDLER_BOT_FAITHFULNESS,
        block_observable=ObservableOutcome.REFUSAL,
    ),
    _case(
        "fiddler_user_safety_blocks_true_exception",
        FIDDLER_USER_SAFETY,
        True,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "fiddler_bot_safety_blocks_true_exception",
        FIDDLER_BOT_SAFETY,
        True,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "fiddler_bot_faithfulness_blocks_true_exception",
        FIDDLER_BOT_FAITHFULNESS,
        True,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "activefence_input_allows_at_simple_threshold",
        ACTIVEFENCE_INPUT,
        _risk_result(0.7),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "activefence_input_blocks_above_simple_threshold",
        ACTIVEFENCE_INPUT,
        _risk_result(0.71),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "activefence_input_blocks_above_simple_threshold_exception",
        ACTIVEFENCE_INPUT,
        _risk_result(0.71),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "activefence_output_allows_at_simple_threshold",
        ACTIVEFENCE_OUTPUT,
        _risk_result(0.7),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "activefence_output_blocks_above_simple_threshold",
        ACTIVEFENCE_OUTPUT,
        _risk_result(0.71),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "activefence_output_blocks_above_simple_threshold_exception",
        ACTIVEFENCE_OUTPUT,
        _risk_result(0.71),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "activefence_input_detailed_allows_at_adult_content_threshold",
        ACTIVEFENCE_INPUT_DETAILED,
        _risk_result(0.3, {"adult_content.general": 0.3}),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "activefence_input_detailed_blocks_above_adult_content_threshold",
        ACTIVEFENCE_INPUT_DETAILED,
        _risk_result(0.31, {"adult_content.general": 0.31}),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
        expected_content="I will not engage with inappropriate content.",
    ),
    _case(
        "activefence_input_detailed_blocks_above_adult_content_threshold_exception",
        ACTIVEFENCE_INPUT_DETAILED,
        _risk_result(0.31, {"adult_content.general": 0.31}),
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "gcp_moderation_output_allows_at_simple_threshold",
        GCP_MODERATION_OUTPUT,
        _risk_result(0.8),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "gcp_moderation_output_blocks_above_simple_threshold",
        GCP_MODERATION_OUTPUT,
        _risk_result(0.81),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "gcp_moderation_output_detailed_allows_at_derogatory_threshold",
        GCP_MODERATION_OUTPUT_DETAILED,
        _risk_result(0.4, {"Derogatory": 0.4}),
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "gcp_moderation_output_detailed_blocks_above_derogatory_threshold",
        GCP_MODERATION_OUTPUT_DETAILED,
        _risk_result(0.41, {"Derogatory": 0.41}),
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
        expected_content="I will not engage in any abusive or harmful behavior.",
    ),
    _case(
        "guardrails_ai_input_allows_valid",
        GUARDRAILS_AI_INPUT,
        {"valid": True, "validation_result": {"validation_passed": True}},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "guardrails_ai_input_blocks_invalid",
        GUARDRAILS_AI_INPUT,
        {"valid": False, "validation_result": {"validation_passed": False}},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "guardrails_ai_input_blocks_invalid_exception",
        GUARDRAILS_AI_INPUT,
        {"valid": False, "validation_result": {"validation_passed": False}},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "guardrails_ai_output_allows_valid",
        GUARDRAILS_AI_OUTPUT,
        {"valid": True, "validation_result": {"validation_passed": True}},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "guardrails_ai_output_blocks_invalid",
        GUARDRAILS_AI_OUTPUT,
        {"valid": False, "validation_result": {"validation_passed": False}},
        ObservableOutcome.REFUSAL,
        FlowDecision.BLOCK,
    ),
    _case(
        "guardrails_ai_output_blocks_invalid_exception",
        GUARDRAILS_AI_OUTPUT,
        {"valid": False, "validation_result": {"validation_passed": False}},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "patronus_lynx_output_allows_no_hallucination",
        PATRONUS_LYNX_OUTPUT,
        {"hallucination": False, "reasoning": ["grounded"]},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "patronus_lynx_output_blocks_hallucination",
        PATRONUS_LYNX_OUTPUT,
        {"hallucination": True, "reasoning": ["unsupported"]},
        ObservableOutcome.ANSWER_UNKNOWN,
        FlowDecision.BLOCK,
    ),
    _case(
        "patronus_lynx_output_blocks_hallucination_exception",
        PATRONUS_LYNX_OUTPUT,
        {"hallucination": True, "reasoning": ["unsupported"]},
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
    ),
    _case(
        "patronus_api_output_allows_passed",
        PATRONUS_API_OUTPUT,
        {"pass": True},
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
    ),
    _case(
        "patronus_api_output_blocks_failed",
        PATRONUS_API_OUTPUT,
        {"pass": False},
        ObservableOutcome.ANSWER_UNKNOWN,
        FlowDecision.BLOCK,
    ),
    _case(
        "self_check_hallucination_allows_false",
        SELF_CHECK_HALLUCINATION,
        False,
        ObservableOutcome.ALLOW,
        FlowDecision.ALLOW,
        context={"check_hallucination": True},
    ),
    _case(
        "self_check_hallucination_blocks_true",
        SELF_CHECK_HALLUCINATION,
        True,
        ObservableOutcome.ANSWER_UNKNOWN,
        FlowDecision.BLOCK,
        context={"check_hallucination": True},
    ),
    _case(
        "self_check_hallucination_blocks_true_exception",
        SELF_CHECK_HALLUCINATION,
        True,
        ObservableOutcome.EXCEPTION,
        FlowDecision.BLOCK,
        enable_rails_exceptions=True,
        context={"check_hallucination": True},
    ),
    *_rail_outcome_cases(
        CONTENT_SAFETY_INPUT,
        allow_return=RailOutcome.allow(policy_violations=[]),
        block_return=RailOutcome.block(policy_violations=["violence"]),
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        CONTENT_SAFETY_OUTPUT,
        allow_return=RailOutcome.allow(policy_violations=[]),
        block_return=RailOutcome.block(policy_violations=["violence"]),
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        TOPIC_SAFETY_INPUT,
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        JAILBREAK_HEURISTICS_INPUT,
        include_exception_case=True,
    ),
    *_rail_outcome_cases(
        JAILBREAK_MODEL_INPUT,
        include_exception_case=True,
    ),
]


def _build_config(spec: RailSpec, *, enable_rails_exceptions: bool) -> dict[str, Any]:
    models = []
    if spec.model_type:
        models.append({"type": spec.model_type, "engine": "openai", "model": "placeholder"})

    rails: dict[str, Any] = {spec.direction: {"flows": [spec.flow]}}
    if spec.rails_config:
        rails["config"] = spec.rails_config

    config: dict[str, Any] = {
        "models": models,
        "rails": rails,
        "enable_rails_exceptions": enable_rails_exceptions,
    }

    if spec.task:
        prompt = {"task": spec.task, "content": "..."}
        if spec.output_parser:
            prompt["output_parser"] = spec.output_parser
        config["prompts"] = [prompt]

    return config


def _run_flow(case: FlowEquivalenceCase) -> tuple[dict[str, Any], dict[str, Any] | None]:
    colang_content = RETRIEVAL_COLANG if case.spec.direction == "retrieval" else None
    config = RailsConfig.from_content(
        colang_content=colang_content,
        config=_build_config(
            case.spec,
            enable_rails_exceptions=case.enable_rails_exceptions,
        ),
    )
    llm_completions = ["  express greeting"] if case.spec.direction == "retrieval" else [NORMAL_OUTPUT]
    chat = TestChat(config, llm_completions=llm_completions)

    async def stub_action(**kwargs):
        return case.raw_return

    chat.app.register_action(stub_action, case.spec.action)
    if case.spec.direction == "retrieval":

        def retrieve_relevant_chunks():
            return ActionResult(
                return_value=RELEVANT_CHUNKS,
                context_updates={"relevant_chunks": RELEVANT_CHUNKS},
            )

        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    messages: list[dict[str, Any]] = [{"role": "user", "content": USER_INPUT}]
    if case.context:
        messages.insert(0, {"role": "context", "content": case.context})

    options = {"output_vars": case.output_vars} if case.output_vars else None
    response = chat.app.generate(messages=messages, options=options)
    if isinstance(response, GenerationResponse):
        message = response.response
        if not isinstance(message, list) or len(message) != 1:
            raise AssertionError(f"Unexpected runtime response: {response!r}")
        if not isinstance(message[0], dict):
            raise AssertionError(f"Unexpected runtime response: {response!r}")
        return message[0], response.output_data

    if not isinstance(response, dict):
        raise AssertionError(f"Unexpected runtime response type: {response!r}")
    return response, None


def _output_data_changed(case: FlowEquivalenceCase, output_data: dict[str, Any] | None) -> bool:
    if case.baseline_output_data is None or output_data is None:
        return False
    return any(output_data.get(key) != value for key, value in case.baseline_output_data.items())


def _classify_response(
    response: dict[str, Any],
    output_data: dict[str, Any] | None,
    case: FlowEquivalenceCase,
) -> ObservableOutcome:
    if response == {"role": "assistant", "content": NORMAL_OUTPUT}:
        if _output_data_changed(case, output_data):
            return ObservableOutcome.TRANSFORM
        return ObservableOutcome.ALLOW
    if response == {"role": "assistant", "content": REFUSAL}:
        return ObservableOutcome.REFUSAL
    if response.get("role") == "assistant" and response.get("content") in RENDERED_BLOCK_MESSAGES:
        return ObservableOutcome.REFUSAL
    if response.get("role") == "assistant" and response.get("content", "").endswith(CLEANLAB_WARNING_SUFFIX):
        return ObservableOutcome.REFUSAL
    if response.get("role") == "assistant" and response.get("content", "").startswith(
        INJECTION_DETECTION_REFUSAL_PREFIX
    ):
        return ObservableOutcome.REFUSAL
    if response == {"role": "assistant", "content": ANSWER_UNKNOWN}:
        return ObservableOutcome.ANSWER_UNKNOWN
    if response.get("role") == "exception":
        return ObservableOutcome.EXCEPTION
    if response.get("role") == "assistant":
        return ObservableOutcome.TRANSFORM

    raise AssertionError(f"Unexpected runtime response: {response!r}")


def _decision_from_observable(observable: ObservableOutcome) -> FlowDecision:
    if observable is ObservableOutcome.ALLOW:
        return FlowDecision.ALLOW
    if observable in (
        ObservableOutcome.REFUSAL,
        ObservableOutcome.ANSWER_UNKNOWN,
        ObservableOutcome.EXCEPTION,
    ):
        return FlowDecision.BLOCK
    return FlowDecision.TRANSFORM


def _outcome_decision(raw_return: Any, spec: RailSpec) -> FlowDecision:
    if isinstance(raw_return, RailOutcome):
        blocked = raw_return.is_blocked
    elif spec.interpret:
        return spec.interpret(raw_return)
    elif isinstance(raw_return, bool):
        blocked = not raw_return
    elif isinstance(raw_return, (int, float)):
        blocked = raw_return < 0.5
    else:
        blocked = False

    return FlowDecision.BLOCK if blocked else FlowDecision.ALLOW


@pytest.mark.parametrize("case", FIXTURES, ids=[case.case_id for case in FIXTURES])
def test_runtime_flow_gate_matches_rail_outcome(case: FlowEquivalenceCase):
    response, output_data = _run_flow(case)
    observable = _classify_response(response, output_data, case)

    assert observable is case.expected_observable
    assert _decision_from_observable(observable) is case.expected_decision
    assert _outcome_decision(case.raw_return, case.spec) is case.expected_decision
    if case.expected_content is not None:
        assert response == {"role": "assistant", "content": case.expected_content}
    if case.expected_output_data is not None:
        assert output_data == case.expected_output_data
