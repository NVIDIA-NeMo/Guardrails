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
)

TREND_AI_GUARD = ActionRef(
    name="trend_ai_guard",
    target="nemoguardrails.library.trend_micro.actions:trend_ai_guard",
)

RAIL = RailManifest(
    name="trend_micro",
    metadata=RailMetadata(
        display_name="Trend Micro Vision One AI Guard",
        description="Guards input and output text with Trend Micro Vision One AI Guard.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_jailbreak", "detect_pii", "moderate"),
        tags=("third-party", "api", "security"),
        docs_url="docs/configure-rails/guardrail-catalog/community/trend-micro.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="trend_micro",
            spec=ConfigSpecRef(target="nemoguardrails.library.trend_micro.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("trend ai guard input", "trend ai guard output")),
        actions=RailActions(refs=(TREND_AI_GUARD,)),
        surfaces=(
            RailSurface(
                name="trend ai guard input",
                direction=RailDirection.INPUT,
                action=TREND_AI_GUARD,
                bindings=(Binding.context("text", "user_message"),),
            ),
            RailSurface(
                name="trend ai guard output",
                direction=RailDirection.OUTPUT,
                action=TREND_AI_GUARD,
                bindings=(Binding.context("text", "bot_message"),),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="V1_API_KEY", required=False),),
            services=(ServiceRequirement(name="Trend Micro Vision One AI Guard", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            remote_services=("Trend Micro Vision One AI Guard",),
        ),
    ),
)
