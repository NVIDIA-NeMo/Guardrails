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
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    ServiceRequirement,
)

CLAVATA_CHECK = ActionRef(
    name="ClavataCheckAction",
    target="nemoguardrails.library.clavata.actions:clavata_check",
)

RAIL = RailManifest(
    name="clavata",
    metadata=RailMetadata(
        display_name="Clavata",
        description="Evaluates input and output text against Clavata policies.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "moderate"),
        tags=("third-party", "api", "policy"),
        docs_url="docs/configure-rails/guardrail-catalog/third-party.md",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="clavata",
            spec=ConfigSpecRef(target="nemoguardrails.library.clavata.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("clavata check for",)),
        actions=RailActions(refs=(CLAVATA_CHECK,)),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="CLAVATA_API_KEY", required=True),),
            services=(ServiceRequirement(name="Clavata API", required=True),),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("Clavata API",)),
    ),
)
