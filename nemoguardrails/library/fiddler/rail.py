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

FIDDLER_USER = ActionRef(
    name="call fiddler safety on user message",
    target="nemoguardrails.library.fiddler.actions:call_fiddler_safety_user",
)
FIDDLER_BOT = ActionRef(
    name="call fiddler safety on bot message",
    target="nemoguardrails.library.fiddler.actions:call_fiddler_safety_bot",
)
FIDDLER_FAITHFULNESS = ActionRef(
    name="call fiddler faithfulness",
    target="nemoguardrails.library.fiddler.actions:call_fiddler_faithfulness",
)

RAIL = RailManifest(
    name="fiddler",
    metadata=RailMetadata(
        display_name="Fiddler Guardrails",
        description="Checks safety and faithfulness with Fiddler Guardrails.",
        categories=("input", "output", "retrieval"),
        capabilities=("allow", "block", "classify", "content_safety", "detect_jailbreak", "fact_check"),
        tags=("third-party", "api", "safety", "faithfulness"),
        docs_url="docs/configure-rails/guardrail-catalog/community/fiddler.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="fiddler",
            spec=ConfigSpecRef(target="nemoguardrails.library.fiddler.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("fiddler user safety", "fiddler bot safety", "fiddler bot faithfulness")),
        actions=RailActions(refs=(FIDDLER_USER, FIDDLER_BOT, FIDDLER_FAITHFULNESS)),
        surfaces=(
            RailSurface(name="fiddler user safety", direction=RailDirection.INPUT, action=FIDDLER_USER),
            RailSurface(name="fiddler bot safety", direction=RailDirection.OUTPUT, action=FIDDLER_BOT),
            RailSurface(name="fiddler bot faithfulness", direction=RailDirection.OUTPUT, action=FIDDLER_FAITHFULNESS),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="FIDDLER_API_KEY", required=True),),
            services=(ServiceRequirement(name="Fiddler Guardrails API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("Fiddler Guardrails API",),
        ),
    ),
)
