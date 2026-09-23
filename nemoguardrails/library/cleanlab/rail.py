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

CALL_CLEANLAB_API = ActionRef(
    name="call_cleanlab_api",
    target="nemoguardrails.library.cleanlab.actions:call_cleanlab_api",
)

RAIL = RailManifest(
    name="cleanlab",
    metadata=RailMetadata(
        display_name="Cleanlab",
        description="Checks bot response trustworthiness using Cleanlab TLM.",
        categories=("output",),
        capabilities=("allow", "block", "classify", "fact_check"),
        tags=("third-party", "api", "hallucination", "trustworthiness"),
        docs_url="docs/configure-rails/guardrail-catalog/community/cleanlab.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("cleanlab trustworthiness",)),
        actions=RailActions(refs=(CALL_CLEANLAB_API,)),
        surfaces=(
            RailSurface(
                name="cleanlab trustworthiness",
                direction=RailDirection.OUTPUT,
                action=CALL_CLEANLAB_API,
                bindings=(
                    Binding.context("user_message", "user_message", required=False),
                    Binding.context("bot_message", "bot_message"),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="CLEANLAB_API_KEY", required=True),),
            services=(ServiceRequirement(name="Cleanlab TLM", required=True),),
            optional_dependencies=("cleanlab-studio",),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, remote_services=("Cleanlab TLM",)),
    ),
)
