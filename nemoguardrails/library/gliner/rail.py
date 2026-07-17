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

GLINER_DETECT_PII = ActionRef(
    name="gliner_detect_pii",
    target="nemoguardrails.library.gliner.actions:gliner_detect_pii",
)
GLINER_MASK_PII = ActionRef(
    name="gliner_mask_pii",
    target="nemoguardrails.library.gliner.actions:gliner_mask_pii",
)

RAIL = RailManifest(
    name="gliner",
    metadata=RailMetadata(
        display_name="GLiNER",
        description="Detects and masks PII in input, output, and retrieved chunks using GLiNER.",
        categories=("input", "output", "retrieval"),
        capabilities=("block", "classify", "detect_pii", "mask", "transform"),
        tags=("pii", "ner", "nim", "gliner"),
        docs_url="docs/configure-rails/guardrail-catalog/community/gliner.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="gliner",
            spec=ConfigSpecRef(target="nemoguardrails.library.gliner.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "gliner detect pii on input",
                "gliner detect pii on output",
                "gliner detect pii on retrieval",
                "gliner mask pii on input",
                "gliner mask pii on output",
                "gliner mask pii on retrieval",
            ),
        ),
        actions=RailActions(refs=(GLINER_DETECT_PII, GLINER_MASK_PII)),
        surfaces=(
            RailSurface(
                name="gliner detect pii on input",
                direction=RailDirection.INPUT,
                action=GLINER_DETECT_PII,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
            ),
            RailSurface(
                name="gliner detect pii on output",
                direction=RailDirection.OUTPUT,
                action=GLINER_DETECT_PII,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
            ),
            RailSurface(
                name="gliner detect pii on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=GLINER_DETECT_PII,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
            ),
            RailSurface(
                name="gliner mask pii on input",
                direction=RailDirection.INPUT,
                action=GLINER_MASK_PII,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="gliner mask pii on output",
                direction=RailDirection.OUTPUT,
                action=GLINER_MASK_PII,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
            RailSurface(
                name="gliner mask pii on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=GLINER_MASK_PII,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="NVIDIA_API_KEY", required=False),),
            services=(ServiceRequirement(name="GLiNER endpoint", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("GLiNER endpoint",),
        ),
    ),
)
