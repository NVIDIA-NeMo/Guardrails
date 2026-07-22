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

from nemoguardrails.manifests import (
    ActionRef,
    Binding,
    ConfigSpecRef,
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

VALIDATE_GUARDRAILS_AI_INPUT = ActionRef(
    name="validate_guardrails_ai_input",
    target="nemoguardrails.library.guardrails_ai.actions:validate_guardrails_ai_input",
)
VALIDATE_GUARDRAILS_AI_OUTPUT = ActionRef(
    name="validate_guardrails_ai_output",
    target="nemoguardrails.library.guardrails_ai.actions:validate_guardrails_ai_output",
)

RAIL = RailManifest(
    name="guardrails_ai",
    metadata=RailMetadata(
        display_name="Guardrails AI",
        description="Runs configured Guardrails AI validators on input or output text.",
        categories=("input", "output"),
        capabilities=(
            "allow",
            "block",
            "classify",
            "content_safety",
            "detect_jailbreak",
            "detect_pii",
            "topic_control",
        ),
        tags=("third-party", "validators"),
        docs_url="docs/configure-rails/guardrail-catalog/community/guardrails-ai.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="guardrails_ai",
            spec=ConfigSpecRef(target="nemoguardrails.library.guardrails_ai.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("guardrailsai check input", "guardrailsai check output")),
        actions=RailActions(refs=(VALIDATE_GUARDRAILS_AI_INPUT, VALIDATE_GUARDRAILS_AI_OUTPUT)),
        surfaces=(
            RailSurface(
                name="guardrailsai check input",
                direction=RailDirection.INPUT,
                action=VALIDATE_GUARDRAILS_AI_INPUT,
                bindings=(
                    Binding.surface_param("validator", "validator"),
                    Binding.context("text", "user_message"),
                ),
            ),
            RailSurface(
                name="guardrailsai check output",
                direction=RailDirection.OUTPUT,
                action=VALIDATE_GUARDRAILS_AI_OUTPUT,
                bindings=(
                    Binding.surface_param("validator", "validator"),
                    Binding.context("text", "bot_message"),
                ),
            ),
        ),
        requirements=RailRequirements(
            services=(ServiceRequirement(name="Guardrails Hub", required=False),),
            optional_dependencies=("guardrails-ai",),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True),
    ),
)
