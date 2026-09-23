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
    EnvVar,
    RailActions,
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

PROTECT_TEXT = ActionRef(
    name="protect_text",
    target="nemoguardrails.library.prompt_security.actions:protect_text",
)

RAIL = RailManifest(
    name="prompt_security",
    metadata=RailMetadata(
        display_name="Prompt Security",
        description="Protects prompts and responses with the Prompt Security API.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "detect_jailbreak", "transform"),
        tags=("third-party", "api", "security"),
        docs_url="docs/configure-rails/guardrail-catalog/community/prompt-security.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("protect prompt", "protect response")),
        actions=RailActions(refs=(PROTECT_TEXT,)),
        surfaces=(
            RailSurface(
                name="protect prompt",
                direction=RailDirection.INPUT,
                action=PROTECT_TEXT,
                bindings=(Binding.context("user_prompt", "user_message"),),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="protect response",
                direction=RailDirection.OUTPUT,
                action=PROTECT_TEXT,
                bindings=(Binding.context("bot_response", "bot_message"),),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="PS_PROTECT_URL", required=True),
                EnvVar(name="PS_APP_ID", required=True),
            ),
            services=(ServiceRequirement(name="Prompt Security Protect API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True, sends_bot_text=True, remote_services=("Prompt Security Protect API",)
        ),
    ),
)
