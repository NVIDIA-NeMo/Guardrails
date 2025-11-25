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

"""Check for matches against a Clavata policy."""

import logging
import uuid
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

from pydantic import BaseModel, Field

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.clavata.errs import (
    ClavataPluginAPIError,
    ClavataPluginConfigurationError,
    ClavataPluginValueError,
)
from nemoguardrails.library.clavata.request import ClavataClient, Job
from nemoguardrails.rails.llm.config import ClavataRailConfig, ClavataRailOptions

if TYPE_CHECKING:
    from .request import Report, SectionReport

log = logging.getLogger(__name__)

VALID_RAILS = ["input", "output"]
ValidRailsType = Literal["input", "output"]


class LabelResult(BaseModel):
    """Result of a label evaluation"""

    label: str = Field(description="The label that was evaluated")
    message: str = Field(description="An arbitrary message attached to the label in the policy.")
    matched: bool = Field(description="Whether the label matched the policy")

    @classmethod
    def from_section_report(cls, report: "SectionReport") -> "LabelResult":
        """Convert a Clavata section report to a LabelResult"""
        return cls(
            label=report.name,
            message=report.message,
            matched=report.result == "OUTCOME_TRUE",
        )


class PolicyResult(BaseModel):
    """Result of Clavata Policy Evaluation"""

    failed: bool = Field(default=False, description="Whether the policy evaluation failed")
    policy_matched: bool = Field(default=False, description="Whether any part of the policy matched the input")
    label_matches: List[LabelResult] = Field(
        default=[],
        description="List of section results from the policy evaluation",
    )

    @classmethod
    def from_report(cls, report: "Report") -> "PolicyResult":
        """Convert a Clavata report to a PolicyResult"""
        return cls(
            failed=report.result == "OUTCOME_FAILED",
            policy_matched=report.result == "OUTCOME_TRUE",
            label_matches=[LabelResult.from_section_report(report) for report in report.sectionEvaluationReports],
        )

    @classmethod
    def from_job(cls, job: "Job") -> "PolicyResult":
        """Convert a Clavata job to a PolicyResult"""
        failed = job.status in ["JOB_STATUS_FAILED", "JOB_STATUS_CANCELED"]
        if failed:
            return cls(failed=True)

        if job.status != "JOB_STATUS_COMPLETED":
            raise ClavataPluginAPIError(f"Policy evaluation is not complete. Status: {job.status}")

        reports = [res.report for res in job.results]
        # We should only ever have one report per job as we're only sending one content item
        if len(reports) != 1:
            raise ClavataPluginAPIError(f"Expected 1 report per job, got {len(reports)}")

        report = reports[0]
        return cls.from_report(report)


def get_clavata_config(config: Any) -> ClavataRailConfig:
    """Get the Clavata config and flow config for the given source."""
    if not isinstance(config, RailsConfig):
        raise ClavataPluginValueError("Passed configuration object is not a RailsConfig")

    if not hasattr(config.rails.config, "clavata") or config.rails.config.clavata is None:
        raise ClavataPluginConfigurationError("Clavata config is not defined in the Rails config.")

    return cast(ClavataRailConfig, config.rails.config.clavata)


def get_policy_id(
    config: ClavataRailConfig,
    policy: Union[str, None] = None,
    rail: Optional[ValidRailsType] = None,
) -> uuid.UUID:
    """
    Get Policy ID will check the input policy. If the input is already a UUID, that UUID will be used. Otherwise,
    the config will be checked to try to match the input policy alias to a policy ID. If no match is found, an error
    will be raised.
    """
    if policy is None:
        if rail is not None:
            policy_name = getattr(config, rail).policy
            return get_policy_id(config, policy_name)

        raise ClavataPluginValueError("'policy' is required, or 'rail' must be provided.")

    # Policy was provided, so we try to convert to a UUID
    try:
        return uuid.UUID(policy)
    except ValueError:
        pass

    # Not a valid UUID, try to match the provided alias to a policy ID and return that
    try:
        policy_id = config.policies.get(policy)
        if policy_id is None:
            raise ClavataPluginValueError(f"Policy with alias '{policy}' not found.")
        return uuid.UUID(policy_id)
    except ValueError as e:
        # Specifically catch the ValueError for badly formed UUIDs so we can provide a more helpful error message
        if "badly formed" in str(e):
            raise ClavataPluginConfigurationError(
                f"Policy ID '{policy_id}' for alias '{policy}' is not a valid UUID. "
                "Please check the Clavata configuration."
            ) from e
        raise


def get_labels(
    config: ClavataRailConfig,
    labels: Optional[Union[List[str], str]] = None,
    rail: Optional[ValidRailsType] = None,
) -> List[str]:
    """
    Checks whether the provided text matches the specified Clavata policy ID.
    Note that this action will return True if any label in the policy matches the text.
    """
    # If labels is provided, just return them
    if labels is not None:
        if isinstance(labels, str):
            # If a string, we'll convert it to a list
            labels = labels.split(",")
        return labels

    # If labels are not provided, we need to get them from the config
    if rail is None:
        raise ClavataPluginValueError("Rail is required when labels are not provided.")

    rail_config: ClavataRailOptions = getattr(config, rail)
    labels = rail_config.labels
    return labels


def is_label_match(
    result: PolicyResult,
    labels: List[str],
    clavata_config: ClavataRailConfig,
) -> bool:
    """Check whether the labels matched the policy"""
    labels_to_match = set(labels)
    labels_matched = set(lbl.label for lbl in result.label_matches if lbl.matched)

    match_all = clavata_config.label_match_logic == "ALL"
    if match_all:
        return labels_to_match.issubset(labels_matched)

    # If matching any of the labels is fine, then we can just check whether
    # there is any intersection between the labels to match and the labels that matched
    return bool(labels_to_match.intersection(labels_matched))


def get_server_endpoint(config: ClavataRailConfig) -> str:
    """Get the server endpoint from the Clavata config."""
    return str(config.server_endpoint).rstrip("/")


async def evaluate_with_policy(
    text: str,
    policy_id: str,
    clavata_config: ClavataRailConfig,
) -> PolicyResult:
    """Get the policy result for the given source."""
    client = ClavataClient(
        base_endpoint=get_server_endpoint(clavata_config),
    )

    job = await client.create_job(text, policy_id)
    rv = PolicyResult.from_job(job)

    if rv.failed:
        raise ClavataPluginAPIError("Policy evaluation failed.")

    return rv


@action(name="ClavataCheckAction")
async def clavata_check(
    text: str,
    policy: Union[str, None] = None,
    labels: Optional[Union[List[str], str]] = None,
    rail: Union[ValidRailsType, None] = None,
    config: Optional[RailsConfig] = None,
    **kwargs: Any,
) -> bool:
    """Check for matches against a Clavata policy."""
    if not config:
        raise ClavataPluginValueError("Rails config is required.")

    if not text:
        raise ClavataPluginValueError("Text to evaluate is required.")

    # Grab the Clavata Config
    clavata_config = get_clavata_config(config)

    # We should have received either a policy and label as params, or a rail that will tell us how to get
    # the policy and labels from the config.
    policy_id = get_policy_id(clavata_config, policy, rail)

    # Try to get the labels either from the provided labels or from the rails config
    try:
        labels = get_labels(clavata_config, labels=labels, rail=rail)
    except ClavataPluginValueError:
        labels = None

    result = await evaluate_with_policy(text, str(policy_id), clavata_config)

    if labels:
        return is_label_match(result, labels, clavata_config)

    return result.policy_matched
