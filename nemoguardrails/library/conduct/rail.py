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

"""Conduct guard rail manifest."""

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
)

CONDUCT_CHECK = ActionRef(
    name="ConductCheckAction",
    target="nemoguardrails.library.conduct.actions:conduct_check",
)


RAIL = RailManifest(
    name="conduct",
    metadata=RailMetadata(
        display_name="Conduct Guard",
        description=(
            "Runtime policy enforcement for LLM prompts and responses. "
            "Blocks credential leaks, prompt injection, PII, and dual-use "
            "framing before a request reaches the model. Every decision is "
            "attributed and hash-chained in the Conduct audit trail."
        ),
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "moderate"),
        tags=("third-party", "api", "policy", "audit"),
        docs_url="docs/configure-rails/guardrail-catalog/community/conduct.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="conduct",
            spec=ConfigSpecRef(target="nemoguardrails.library.conduct.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("conduct check input", "conduct check output")),
        actions=RailActions(refs=(CONDUCT_CHECK,)),
        surfaces=(
            RailSurface(
                name="conduct check input",
                direction=RailDirection.INPUT,
                action=CONDUCT_CHECK,
                bindings=(
                    Binding.context("text", "user_message"),
                    Binding.literal("rail", "input"),
                ),
            ),
            RailSurface(
                name="conduct check output",
                direction=RailDirection.OUTPUT,
                action=CONDUCT_CHECK,
                bindings=(
                    Binding.context("text", "bot_message"),
                    Binding.literal("rail", "output"),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="CONDUCT_AGENT_TOKEN", required=True),
                EnvVar(name="CONDUCT_WORKSPACE_ID", required=False),
                EnvVar(name="CONDUCT_API_URL", required=False),
            ),
            services=(ServiceRequirement(name="Conduct API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            remote_services=("Conduct API",),
        ),
    ),
)
