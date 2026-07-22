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

CALL_ACTIVEFENCE_API = ActionRef(
    name="call_activefence_api",
    target="nemoguardrails.library.activefence.actions:call_activefence_api",
)

RAIL = RailManifest(
    name="activefence",
    metadata=RailMetadata(
        display_name="ActiveFence",
        description="Moderates input and output text with the ActiveFence ActiveScore API.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_pii", "moderate"),
        tags=("third-party", "api", "moderation"),
        docs_url="docs/configure-rails/guardrail-catalog/community/active-fence.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(
            flow_names=(
                "activefence moderation on input",
                "activefence moderation on output",
                "activefence moderation on input detailed",
            ),
        ),
        actions=RailActions(refs=(CALL_ACTIVEFENCE_API,)),
        surfaces=(
            RailSurface(
                name="activefence moderation on input",
                direction=RailDirection.INPUT,
                action=CALL_ACTIVEFENCE_API,
                bindings=(
                    Binding.context("text", "user_message"),
                    Binding.literal("threshold_mode", "simple"),
                ),
            ),
            RailSurface(
                name="activefence moderation on output",
                direction=RailDirection.OUTPUT,
                action=CALL_ACTIVEFENCE_API,
                bindings=(
                    Binding.context("text", "bot_message"),
                    Binding.literal("threshold_mode", "simple"),
                ),
            ),
            RailSurface(
                name="activefence moderation on input detailed",
                direction=RailDirection.INPUT,
                action=CALL_ACTIVEFENCE_API,
                bindings=(
                    Binding.context("text", "user_message"),
                    Binding.literal("threshold_mode", "detailed"),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="ACTIVEFENCE_API_KEY", required=True),),
            services=(ServiceRequirement(name="ActiveFence ActiveScore API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True, sends_bot_text=True, remote_services=("ActiveFence ActiveScore API",)
        ),
    ),
)
