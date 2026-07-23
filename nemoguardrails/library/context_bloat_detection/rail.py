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
    RailActions,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailSpec,
    RailSurface,
    TransformTarget,
)

CONTEXT_BLOAT_DETECTION = ActionRef(
    name="context_bloat_detection",
    target="nemoguardrails.library.context_bloat_detection.actions:context_bloat_detection",
)

RAIL = RailManifest(
    name="context_bloat_detection",
    metadata=RailMetadata(
        display_name="Context Bloat Detection",
        description="Detects oversized, repetitive, or low-entropy input and retrieved context.",
        categories=("input", "retrieval"),
        capabilities=("classify",),
        tags=("context", "retrieval", "denial-of-service"),
        docs_url="docs/configure-rails/guardrail-catalog/context-bloat-detection.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="context_bloat_detection",
            spec=ConfigSpecRef(target="nemoguardrails.library.context_bloat_detection.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=("context bloat detection on input", "context bloat detection on retrieval"),
        ),
        actions=RailActions(refs=(CONTEXT_BLOAT_DETECTION,)),
        surfaces=(
            RailSurface(
                name="context bloat detection on input",
                direction=RailDirection.INPUT,
                action=CONTEXT_BLOAT_DETECTION,
                bindings=(
                    Binding.literal("source", "input"),
                    Binding.context("text", "user_message"),
                ),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="context bloat detection on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=CONTEXT_BLOAT_DETECTION,
                bindings=(
                    Binding.literal("source", "retrieval"),
                    Binding.context("text", "relevant_chunks"),
                ),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
    ),
)
