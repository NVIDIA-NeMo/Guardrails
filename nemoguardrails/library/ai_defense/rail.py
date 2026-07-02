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

AI_DEFENSE_INSPECT = ActionRef(
    name="ai_defense_inspect",
    target="nemoguardrails.library.ai_defense.actions:ai_defense_inspect",
)

RAIL = RailManifest(
    name="ai_defense",
    metadata=RailMetadata(
        display_name="Cisco AI Defense",
        description="Inspects prompts and responses with Cisco AI Defense.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_jailbreak", "detect_pii", "moderate"),
        tags=("third-party", "api", "security"),
        docs_url="docs/configure-rails/guardrail-catalog/community/ai-defense.md",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="ai_defense",
            spec=ConfigSpecRef(target="nemoguardrails.library.ai_defense.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("ai defense inspect prompt", "ai defense inspect response")),
        actions=RailActions(refs=(AI_DEFENSE_INSPECT,)),
        surfaces=(
            RailSurface(
                name="ai defense inspect prompt",
                direction=RailDirection.INPUT,
                action=AI_DEFENSE_INSPECT,
                bindings=(Binding.context("user_prompt", "user_message"),),
            ),
            RailSurface(
                name="ai defense inspect response",
                direction=RailDirection.OUTPUT,
                action=AI_DEFENSE_INSPECT,
                bindings=(Binding.context("bot_response", "bot_message"),),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="AI_DEFENSE_API_ENDPOINT", required=True),
                EnvVar(name="AI_DEFENSE_API_KEY", required=True),
            ),
            services=(ServiceRequirement(name="Cisco AI Defense", required=True),),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("Cisco AI Defense",)),
    ),
)
