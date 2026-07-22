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

PANGEA_AI_GUARD = ActionRef(
    name="pangea_ai_guard",
    target="nemoguardrails.library.pangea.actions:pangea_ai_guard",
)

RAIL = RailManifest(
    name="pangea",
    metadata=RailMetadata(
        display_name="Pangea AI Guard",
        description="Guards input and output messages with Pangea AI Guard.",
        categories=("input", "output"),
        capabilities=(
            "allow",
            "block",
            "classify",
            "content_safety",
            "detect_jailbreak",
            "detect_pii",
            "moderate",
            "transform",
        ),
        tags=("third-party", "api", "security"),
        docs_url="docs/configure-rails/guardrail-catalog/community/pangea.mdx",
        lifecycle="deprecated",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="pangea",
            spec=ConfigSpecRef(target="nemoguardrails.library.pangea.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("pangea ai guard input", "pangea ai guard output")),
        actions=RailActions(refs=(PANGEA_AI_GUARD,)),
        surfaces=(
            RailSurface(
                name="pangea ai guard input",
                direction=RailDirection.INPUT,
                action=PANGEA_AI_GUARD,
                bindings=(
                    Binding.literal("mode", "input"),
                    Binding.context("user_message", "user_message"),
                ),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="pangea ai guard output",
                direction=RailDirection.OUTPUT,
                action=PANGEA_AI_GUARD,
                bindings=(
                    Binding.literal("mode", "output"),
                    Binding.context("user_message", "user_message", required=False),
                    Binding.context("bot_message", "bot_message"),
                ),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="PANGEA_API_TOKEN", required=True),
                EnvVar(name="PANGEA_BASE_URL_TEMPLATE", required=False),
            ),
            services=(ServiceRequirement(name="Pangea AI Guard", required=True),),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("Pangea AI Guard",)),
    ),
)
