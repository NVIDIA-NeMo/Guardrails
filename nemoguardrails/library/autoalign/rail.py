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
    TransformTarget,
)

AUTOALIGN_INPUT_API = ActionRef(
    name="autoalign_input_api",
    target="nemoguardrails.library.autoalign.actions:autoalign_input_api",
)
AUTOALIGN_OUTPUT_API = ActionRef(
    name="autoalign_output_api",
    target="nemoguardrails.library.autoalign.actions:autoalign_output_api",
)
AUTOALIGN_GROUNDEDNESS_OUTPUT_API = ActionRef(
    name="autoalign_groundedness_output_api",
    target="nemoguardrails.library.autoalign.actions:autoalign_groundedness_output_api",
)
AUTOALIGN_FACTCHECK_OUTPUT_API = ActionRef(
    name="autoalign_factcheck_output_api",
    target="nemoguardrails.library.autoalign.actions:autoalign_factcheck_output_api",
)

RAIL = RailManifest(
    name="autoalign",
    metadata=RailMetadata(
        display_name="AutoAlign",
        description="Uses AutoAlign APIs for input safety, output safety, groundedness, and fact checking.",
        categories=("input", "output", "retrieval"),
        capabilities=(
            "block",
            "classify",
            "content_safety",
            "detect_jailbreak",
            "detect_pii",
            "fact_check",
            "mask",
            "moderate",
            "transform",
        ),
        tags=("third-party", "api", "moderation", "pii", "fact-checking"),
        docs_url="docs/configure-rails/guardrail-catalog/community/auto-align.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="autoalign",
            spec=ConfigSpecRef(target="nemoguardrails.library.autoalign.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "autoalign check input",
                "autoalign check output",
                "autoalign groundedness output",
                "autoalign factcheck output",
            ),
        ),
        actions=RailActions(
            refs=(
                AUTOALIGN_INPUT_API,
                AUTOALIGN_OUTPUT_API,
                AUTOALIGN_GROUNDEDNESS_OUTPUT_API,
                AUTOALIGN_FACTCHECK_OUTPUT_API,
            ),
        ),
        surfaces=(
            RailSurface(
                name="autoalign check input",
                direction=RailDirection.INPUT,
                action=AUTOALIGN_INPUT_API,
                bindings=(Binding.literal("show_autoalign_message", True),),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="autoalign check output",
                direction=RailDirection.OUTPUT,
                action=AUTOALIGN_OUTPUT_API,
                bindings=(Binding.literal("show_autoalign_message", True),),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
            RailSurface(
                name="autoalign groundedness output",
                direction=RailDirection.OUTPUT,
                action=AUTOALIGN_GROUNDEDNESS_OUTPUT_API,
                bindings=(
                    Binding.literal("factcheck_threshold", 0.5),
                    Binding.literal("show_autoalign_message", True),
                ),
            ),
            RailSurface(
                name="autoalign factcheck output",
                direction=RailDirection.OUTPUT,
                action=AUTOALIGN_FACTCHECK_OUTPUT_API,
                bindings=(
                    Binding.literal("factcheck_threshold", 0.5),
                    Binding.literal("show_autoalign_message", True),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="AUTOALIGN_API_KEY", required=False),),
            services=(ServiceRequirement(name="AutoAlign API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("AutoAlign API",),
        ),
    ),
)
