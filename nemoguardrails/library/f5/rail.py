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

F5_GUARDRAILS_SCAN = ActionRef(
    name="f5_guardrails_scan",
    target="nemoguardrails.library.f5.actions:f5_guardrails_scan",
)

RAIL = RailManifest(
    name="f5",
    metadata=RailMetadata(
        display_name="F5 AI Guardrails",
        description="Scans input and output text with the F5 AI Guardrails API.",
        categories=("input", "output"),
        capabilities=("allow", "block", "content_safety", "moderate"),
        tags=("third-party", "api", "safety"),
        docs_url="docs/configure-rails/guardrail-catalog/community/f5-ai-guardrails.md",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="f5",
            spec=ConfigSpecRef(target="nemoguardrails.library.f5.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("f5 guardrails scan input", "f5 guardrails scan output")),
        actions=RailActions(refs=(F5_GUARDRAILS_SCAN,)),
        surfaces=(
            RailSurface(
                name="f5 guardrails scan input",
                direction=RailDirection.INPUT,
                action=F5_GUARDRAILS_SCAN,
                bindings=(Binding.context("text", "user_message"),),
            ),
            RailSurface(
                name="f5 guardrails scan output",
                direction=RailDirection.OUTPUT,
                action=F5_GUARDRAILS_SCAN,
                bindings=(Binding.context("text", "bot_message"),),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="F5_GUARDRAILS_API_KEY", required=True),
                EnvVar(name="F5_GUARDRAILS_API_URL", required=False),
            ),
            services=(ServiceRequirement(name="F5 AI Guardrails API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            remote_services=("F5 AI Guardrails API",),
        ),
    ),
)
