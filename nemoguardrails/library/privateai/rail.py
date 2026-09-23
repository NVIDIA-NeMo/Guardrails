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
    TransformTarget,
)

DETECT_PII = ActionRef(
    name="detect_pii",
    target="nemoguardrails.library.privateai.actions:detect_pii",
)
MASK_PII = ActionRef(
    name="mask_pii",
    target="nemoguardrails.library.privateai.actions:mask_pii",
)

RAIL = RailManifest(
    name="privateai",
    metadata=RailMetadata(
        display_name="Private AI",
        description="Detects and masks PII in input, output, and retrieved chunks using Private AI.",
        categories=("input", "output", "retrieval"),
        capabilities=("block", "classify", "detect_pii", "mask", "transform"),
        tags=("third-party", "api", "pii"),
        docs_url="docs/configure-rails/guardrail-catalog/community/privateai.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="privateai",
            spec=ConfigSpecRef(target="nemoguardrails.library.privateai.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "detect pii on input",
                "detect pii on output",
                "detect pii on retrieval",
                "mask pii on input",
                "mask pii on output",
                "mask pii on retrieval",
            ),
        ),
        actions=RailActions(refs=(DETECT_PII, MASK_PII)),
        surfaces=(
            RailSurface(
                name="detect pii on input",
                direction=RailDirection.INPUT,
                action=DETECT_PII,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
            ),
            RailSurface(
                name="detect pii on output",
                direction=RailDirection.OUTPUT,
                action=DETECT_PII,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
            ),
            RailSurface(
                name="detect pii on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=DETECT_PII,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
            ),
            RailSurface(
                name="mask pii on input",
                direction=RailDirection.INPUT,
                action=MASK_PII,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="mask pii on output",
                direction=RailDirection.OUTPUT,
                action=MASK_PII,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
            RailSurface(
                name="mask pii on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=MASK_PII,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="PAI_API_KEY", required=False),),
            services=(ServiceRequirement(name="Private AI API", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("Private AI API",),
        ),
    ),
)
