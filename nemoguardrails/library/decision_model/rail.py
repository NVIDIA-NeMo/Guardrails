# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DECISION_MODEL_CHECK = ActionRef(
    name="decision_model_check",
    target="nemoguardrails.library.decision_model.actions:decision_model_check",
)

RAIL = RailManifest(
    name="decision_model",
    metadata=RailMetadata(
        display_name="Decision Model",
        description="Screens input and output text with user-defined yes/no questions answered by a configured decision model.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_jailbreak"),
        tags=("third-party", "api", "decision_model"),
        docs_url="docs/configure-rails/guardrail-catalog/community/decision-model.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="decision_model",
            spec=ConfigSpecRef(target="nemoguardrails.library.decision_model.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("decision model check input", "decision model check output")),
        actions=RailActions(refs=(DECISION_MODEL_CHECK,)),
        surfaces=(
            RailSurface(
                name="decision model check input",
                direction=RailDirection.INPUT,
                action=DECISION_MODEL_CHECK,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
            ),
            RailSurface(
                name="decision model check output",
                direction=RailDirection.OUTPUT,
                action=DECISION_MODEL_CHECK,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(
                    name="DECISION_MODEL_API_KEY",
                    required=False,
                    description="Bearer token used when api_key_env_var selects this variable (the default).",
                ),
            ),
            services=(ServiceRequirement(name="Configured decision model endpoint", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True, sends_bot_text=True, remote_services=("Configured decision model endpoint",)
        ),
    ),
)
