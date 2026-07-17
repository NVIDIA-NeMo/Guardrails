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

CROWDSTRIKE_AIDR_GUARD = ActionRef(
    name="crowdstrike_aidr_guard",
    target="nemoguardrails.library.crowdstrike_aidr.actions:crowdstrike_aidr_guard",
)

RAIL = RailManifest(
    name="crowdstrike_aidr",
    metadata=RailMetadata(
        display_name="CrowdStrike AIDR",
        description="Guards input and output messages with CrowdStrike AIDR.",
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
        docs_url="docs/configure-rails/guardrail-catalog/community/crowdstrike-aidr.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="crowdstrike_aidr",
            spec=ConfigSpecRef(target="nemoguardrails.library.crowdstrike_aidr.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("crowdstrike aidr guard input", "crowdstrike aidr guard output")),
        actions=RailActions(refs=(CROWDSTRIKE_AIDR_GUARD,)),
        surfaces=(
            RailSurface(
                name="crowdstrike aidr guard input",
                direction=RailDirection.INPUT,
                action=CROWDSTRIKE_AIDR_GUARD,
                bindings=(
                    Binding.literal("mode", "input"),
                    Binding.context("user_message", "user_message"),
                ),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="crowdstrike aidr guard output",
                direction=RailDirection.OUTPUT,
                action=CROWDSTRIKE_AIDR_GUARD,
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
                EnvVar(name="CS_AIDR_TOKEN", required=True),
                EnvVar(name="CS_AIDR_BASE_URL_TEMPLATE", required=False),
            ),
            services=(ServiceRequirement(name="CrowdStrike AIDR API", required=True),),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("CrowdStrike AIDR API",)),
    ),
)
