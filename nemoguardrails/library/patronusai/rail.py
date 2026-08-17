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
    ModelRequirement,
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

PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION = ActionRef(
    name="patronus_lynx_check_output_hallucination",
    target="nemoguardrails.library.patronusai.actions:patronus_lynx_check_output_hallucination",
)
PATRONUS_API_CHECK_OUTPUT = ActionRef(
    name="patronus_api_check_output",
    target="nemoguardrails.library.patronusai.actions:patronus_api_check_output",
)

RAIL = RailManifest(
    name="patronusai",
    metadata=RailMetadata(
        display_name="Patronus AI",
        description="Checks bot output using Patronus Lynx or the Patronus Evaluate API.",
        categories=("output",),
        capabilities=("allow", "block", "classify", "fact_check", "moderate"),
        tags=("patronus", "hallucination", "rag", "api"),
        docs_url="docs/configure-rails/guardrail-catalog/community/patronus-evaluate-api.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="patronus",
            spec=ConfigSpecRef(target="nemoguardrails.library.patronusai.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("patronus lynx check output hallucination", "patronus api check output")),
        actions=RailActions(refs=(PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION, PATRONUS_API_CHECK_OUTPUT)),
        surfaces=(
            RailSurface(
                name="patronus lynx check output hallucination",
                direction=RailDirection.OUTPUT,
                action=PATRONUS_LYNX_CHECK_OUTPUT_HALLUCINATION,
                bindings=(
                    Binding.model("model_name", "patronus_lynx"),
                    Binding.context("user_message", "user_message"),
                    Binding.context("bot_message", "bot_message"),
                    Binding.context("relevant_chunks", "relevant_chunks"),
                ),
            ),
            RailSurface(
                name="patronus api check output",
                direction=RailDirection.OUTPUT,
                action=PATRONUS_API_CHECK_OUTPUT,
                bindings=(
                    Binding.context("user_message", "user_message"),
                    Binding.context("bot_message", "bot_message"),
                    Binding.context("relevant_chunks", "relevant_chunks"),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="PATRONUS_API_KEY", required=False),),
            services=(ServiceRequirement(name="Patronus Evaluate API", required=False),),
            models=(ModelRequirement(type="patronus_lynx", required=False),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("Patronus Evaluate API",),
        ),
    ),
)
