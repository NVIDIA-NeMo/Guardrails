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

CALL_POLICYAI_API = ActionRef(
    name="call_policyai_api",
    target="nemoguardrails.library.policyai.actions:call_policyai_api",
)

RAIL = RailManifest(
    name="policyai",
    metadata=RailMetadata(
        display_name="PolicyAI",
        description="Moderates input and output text with PolicyAI.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "moderate"),
        tags=("third-party", "api", "policy"),
        docs_url="docs/configure-rails/guardrail-catalog/community/policyai.md",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("policyai moderation on input", "policyai moderation on output")),
        actions=RailActions(refs=(CALL_POLICYAI_API,)),
        surfaces=(
            RailSurface(
                name="policyai moderation on input",
                direction=RailDirection.INPUT,
                action=CALL_POLICYAI_API,
                bindings=(Binding.context("text", "user_message"),),
            ),
            RailSurface(
                name="policyai moderation on output",
                direction=RailDirection.OUTPUT,
                action=CALL_POLICYAI_API,
                bindings=(Binding.context("text", "bot_message"),),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="POLICYAI_API_KEY", required=True),
                EnvVar(name="POLICYAI_BASE_URL", required=False),
                EnvVar(name="POLICYAI_TAG_NAME", required=False),
            ),
            services=(ServiceRequirement(name="PolicyAI API", required=True),),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("PolicyAI API",)),
    ),
)
