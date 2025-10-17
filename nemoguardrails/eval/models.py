# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator

from nemoguardrails.eval.utils import load_dict_from_path
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.config import Model, TaskPrompt
from nemoguardrails.rails.llm.options import ActivatedRail


class Policy(BaseModel):
    """A policy for the evaluation of a guardrail configuration."""

    id: str = Field(description="A human readable id of the policy.")
    description: str = Field(description="A detailed description of the policy.")
    weight: int = Field(
        default=100, description="The weight of the policy in the overall evaluation."
    )
    apply_to_all: bool = Field(
        default=True,
        description="Whether the policy is applicable by default to all interactions.",
    )


class ExpectedOutput(BaseModel):
    """An expected output from the system, as dictated by a policy."""

    type: str = Field(
        description="The type of expected output, e.g., 'refusal, 'similar_message'"
    )
    policy: str = Field(
        description="The id of the policy dictating the expected output."
    )


class GenericOutput(ExpectedOutput):
    type: str = "generic"
    description: str = Field(description="A description of the expected output.")

    def __str__(self):
        return self.description


class RefusalOutput(ExpectedOutput):
    type: str = "refusal"

    def __str__(self):
        return "Refuse to respond."


class SimilarMessageOutput(ExpectedOutput):
    type: str = "similar_message"
    message: str = Field(
        description="A message that should be similar to the one from the LLM."
    )

    def __str__(self):
        return f'Response similar to "{self.message}"'


class InteractionSet(BaseModel):
    """An interaction set description that is part of an evaluation dataset.

    An interaction set groups multiple interactions with the same expected output.
    """

    id: str = Field(description="A unique identifier for the interaction set.")
    inputs: List[Union[str, dict]] = Field(
        description="A list of alternative inputs for the interaction set."
    )
    expected_output: List[ExpectedOutput] = Field(
        description="Expected output from the system as dictated by various policies."
    )
    include_policies: List[str] = Field(
        default_factory=list,
        description="The list of additional policies that should be included in the evaluation "
        "for this interaction set.",
    )
    exclude_policies: List[str] = Field(
        default_factory=list,
        description="The list of policies that should be excluded from the evaluation "
        "for this interaction set.",
    )
    evaluation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context that can be used when evaluating the compliance for various policies. "
        "Can be used in the prompt templates. ",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="A list of tags that should be associated with the interactions. Useful for filtering when reporting.",
    )

    @root_validator(pre=True)
    def instantiate_expected_output(cls, values: Any):
        """Creates the right instance of the expected output."""
        type_mapping = {
            "generic": GenericOutput,
            "refusal": RefusalOutput,
            "similar_message": SimilarMessageOutput,
        }
        if "expected_output" in values:
            # Instantiate the right object based on 'type' field
            instantiated = []
            for value in values["expected_output"]:
                output_type_cls = type_mapping.get(value["type"], ExpectedOutput)
                instantiated.append(output_type_cls(**value))
            values["expected_output"] = instantiated

        return values


class EvalConfig(BaseModel):
    """An evaluation configuration for an evaluation dataset."""

    policies: List[Policy] = Field(
        description="A list of policies for the evaluation configuration."
    )
    interactions: List[InteractionSet] = Field(
        description="A list of interactions for the evaluation configuration."
    )
    expected_latencies: Dict[str, float] = Field(
        default_factory=dict, description="The expected latencies for various resources"
    )
    models: List[Model] = Field(
        default_factory=list,
        description="The LLM models to be used as judges.",
    )
    prompts: Optional[List[TaskPrompt]] = Field(
        default=None,
        description="The prompts that should be used for the various LLM tasks.",
    )

    @root_validator(pre=False, skip_on_failure=True)
    def validate_policy_ids(cls, values: Any):
        """Validates the policy ids used in the interactions."""
        policy_ids = {policy.id for policy in values.get("policies")}
        for interaction_set in values.get("interactions"):
            for expected_output in interaction_set.expected_output:
                if expected_output.policy not in policy_ids:
                    raise ValueError(
                        f"Invalid policy id {expected_output.policy} used in interaction set."
                    )
            for policy_id in (
                interaction_set.include_policies + interaction_set.exclude_policies
            ):
                if policy_id not in policy_ids:
                    raise ValueError(
                        f"Invalid policy id {policy_id} used in interaction set."
                    )
        return values

    @classmethod
    def from_path(
        cls,
        config_path: str,
    ) -> "EvalConfig":
        """Loads an eval configuration from a given path.

        It supports YAML files (*.yml and *.yaml) and .json files (*.json).
        """
        if os.path.isdir(config_path):
            config_obj = load_dict_from_path(config_path)
        else:
            raise ValueError(f"Invalid config path {config_path}.")

        return cls.parse_obj(config_obj)


class ComplianceCheckLog(BaseModel):
    """Detailed log about a compliance check."""

    id: str = Field(description="A human readable id of the compliance check.")
    llm_calls: List[LLMCallInfo] = Field(
        default_factory=list,
        description="The list of LLM calls performed for the check.",
    )


class ComplianceCheckResult(BaseModel):
    """Information about a compliance check."""

    id: str = Field(description="A human readable id of the compliance check.")
    created_at: str = Field(
        description="The datetime when the compliance check entry was created."
    )
    interaction_id: Optional[str] = Field(description="The id of the interaction.")
    method: str = Field(
        description="The method of the compliance check (e.g., 'llm-judge', 'human')"
    )
    compliance: Dict[str, Optional[Union[bool, str]]] = Field(
        default_factory=dict,
        description="A mapping from policy id to True, False, 'n/a' or None.",
    )
    details: str = Field(
        default="",
        description="Detailed information about the compliance check.",
    )


class InteractionOutput(BaseModel):
    """The output for running and evaluating an interaction."""

    id: str = Field(description="A human readable id of the interaction.")

    input: Union[str, dict] = Field(description="The input of the interaction.")
    output: Optional[Union[str, List[dict]]] = Field(
        default=None, description="The output of the interaction."
    )

    compliance: Dict[str, Optional[Union[bool, str]]] = Field(
        default_factory=dict,
        description="A mapping from policy id to True, False, 'n/a' or None.",
    )
    resource_usage: Dict[str, Union[int, float]] = Field(
        default_factory=dict,
        description="Information about the resources used by the interaction.",
    )
    latencies: Dict[str, Union[float]] = Field(
        default_factory=dict,
        description="Information about the latencies recorded during the interaction.",
    )
    expected_latencies: Dict[str, Union[float]] = Field(
        default_factory=dict,
        description="Information about the expected latencies for the interaction.",
    )
    compliance_checks: List[ComplianceCheckResult] = Field(
        default_factory=list,
        description="Detailed information about the compliance checks.",
    )


class Span(BaseModel):
    """A generic span object"""

    span_id: str = Field(description="The id of the span.")
    name: str = Field(description="A human-readable name for the span.")
    parent_id: Optional[str] = Field(
        default=None, description="The id of the parent span."
    )
    resource_id: Optional[str] = Field(
        default=None, description="The id of the resource."
    )
    start_time: float = Field(description="The start time of the span.")
    end_time: float = Field(description="The end time of the span.")
    duration: float = Field(description="The duration of the span in seconds.")
    metrics: Dict[str, Union[int, float]] = Field(
        default_factory=dict, description="The metrics recorded during the span."
    )


class InteractionLog(BaseModel):
    """Detailed log about the execution of an interaction."""

    id: str = Field(description="A human readable id of the interaction.")

    activated_rails: List[ActivatedRail] = Field(
        default_factory=list, description="Details about the activated rails."
    )
    events: List[dict] = Field(
        default_factory=list,
        description="The full list of events recorded during the interaction.",
    )
    trace: List[Span] = Field(
        default_factory=list, description="Detailed information about the execution."
    )
    compliance_checks: List[ComplianceCheckLog] = Field(
        default_factory=list,
        description="Detailed information about the compliance checks.",
    )


class EvalOutput(BaseModel):
    """The output of the evaluation."""

    results: List[InteractionOutput] = Field(
        default_factory=list, description="The list of outputs for every interaction."
    )
    logs: List[InteractionLog] = Field(
        default_factory=list,
        description="Detailed information about the execution of the interactions.",
    )

    def compute_compliance(self, eval_config: EvalConfig) -> Dict[str, dict]:
        """Helper to compute the compliance rate per policy."""
        # we take the policies from the first interaction
        policy_ids = self.results[0].compliance
        compliance = {}
        for policy_id in policy_ids:
            compliance[policy_id] = {
                "rate": 0.0,
                "interactions_count": 0,
                "interactions_comply_count": 0,
                "interactions_violation_count": 0,
                "interactions_not_applicable_count": 0,
                "interactions_not_rated_count": 0,
            }

        for interaction_output in self.results:
            # First, we make sure that the compliance dict is up-to-date.
            for item in interaction_output.compliance_checks:
                interaction_output.compliance.update(item.compliance)
            for policy in eval_config.policies:
                if (
                    policy.apply_to_all
                    and policy.id not in interaction_output.compliance
                ):
                    interaction_output.compliance[policy.id] = None

            for policy_id, val in interaction_output.compliance.items():
                if val is None:
                    compliance[policy_id]["interactions_not_rated_count"] += 1
                    compliance[policy_id]["interactions_count"] += 1
                elif val is True:
                    compliance[policy_id]["interactions_comply_count"] += 1
                    compliance[policy_id]["interactions_count"] += 1
                elif val is False:
                    compliance[policy_id]["interactions_violation_count"] += 1
                    compliance[policy_id]["interactions_count"] += 1
                elif val == "n/a":
                    compliance[policy_id]["interactions_not_applicable_count"] += 1
                else:
                    raise ValueError(f"Invalid compliance value for {policy_id}: {val}")

        for policy_id in compliance:
            if compliance[policy_id]["interactions_count"] > 0:
                compliance[policy_id]["rate"] = (
                    compliance[policy_id]["interactions_comply_count"]
                    / compliance[policy_id]["interactions_count"]
                )

        return compliance

    @classmethod
    def from_path(
        cls,
        output_path: str,
    ) -> "EvalOutput":
        """Loads an eval output from a given path.

        It supports YAML files (*.yml and *.yaml) and .json files (*.json).
        """
        if os.path.isdir(output_path):
            output_obj = load_dict_from_path(output_path)
        else:
            raise ValueError(f"Invalid config path {output_path}.")

        return cls.parse_obj(output_obj)
