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
)

CALL_ZSCALER_AIGUARD_API = ActionRef(
    name="call_zscaler_aiguard_api",
    target="nemoguardrails.library.zscaler_aiguard.actions:call_zscaler_aiguard_api",
)

RAIL = RailManifest(
    name="zscaler_aiguard",
    metadata=RailMetadata(
        display_name="Zscaler AI Guard",
        description="Moderates input and output text with the Zscaler AI Guard API.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_pii", "moderate"),
        tags=("third-party", "api", "security"),
        docs_url="docs/configure-rails/guardrail-catalog/community/zscaler-aiguard.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(
            flow_names=(
                "zscaler aiguard moderation on input",
                "zscaler aiguard moderation on output",
            ),
        ),
        actions=RailActions(refs=(CALL_ZSCALER_AIGUARD_API,)),
        surfaces=(
            RailSurface(
                name="zscaler aiguard moderation on input",
                direction=RailDirection.INPUT,
                action=CALL_ZSCALER_AIGUARD_API,
                bindings=(
                    Binding.context("text", "user_message"),
                    Binding.literal("direction", "IN"),
                ),
            ),
            RailSurface(
                name="zscaler aiguard moderation on output",
                direction=RailDirection.OUTPUT,
                action=CALL_ZSCALER_AIGUARD_API,
                bindings=(
                    Binding.context("text", "bot_message"),
                    Binding.literal("direction", "OUT"),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="AIGUARD_API_KEY", required=True),
                EnvVar(name="AIGUARD_CLOUD", required=False),
                EnvVar(name="AIGUARD_POLICY_ID", required=False),
            ),
            services=(ServiceRequirement(name="Zscaler AI Guard API", required=True),),
            optional_dependencies=("zscaler-sdk-python",),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            remote_services=("Zscaler AI Guard API",),
        ),
    ),
)
