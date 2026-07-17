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
    TransformTarget,
)

DETECT_SENSITIVE_DATA = ActionRef(
    name="detect_sensitive_data",
    target="nemoguardrails.library.sensitive_data_detection.actions:detect_sensitive_data",
)
MASK_SENSITIVE_DATA = ActionRef(
    name="mask_sensitive_data",
    target="nemoguardrails.library.sensitive_data_detection.actions:mask_sensitive_data",
)

RAIL = RailManifest(
    name="sensitive_data_detection",
    metadata=RailMetadata(
        display_name="Sensitive Data Detection",
        description="Detects and masks sensitive data using Presidio.",
        categories=("input", "output", "retrieval"),
        capabilities=("block", "classify", "detect_pii", "mask", "transform"),
        tags=("built-in", "pii", "presidio", "sensitive-data"),
        docs_url="docs/configure-rails/guardrail-catalog/pii-detection.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="sensitive_data_detection",
            spec=ConfigSpecRef(target="nemoguardrails.library.sensitive_data_detection.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "detect sensitive data on input",
                "mask sensitive data on input",
                "detect sensitive data on output",
                "mask sensitive data on output",
                "detect sensitive data on retrieval",
                "mask sensitive data on retrieval",
            ),
        ),
        actions=RailActions(refs=(DETECT_SENSITIVE_DATA, MASK_SENSITIVE_DATA)),
        surfaces=(
            RailSurface(
                name="detect sensitive data on input",
                direction=RailDirection.INPUT,
                action=DETECT_SENSITIVE_DATA,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
            ),
            RailSurface(
                name="mask sensitive data on input",
                direction=RailDirection.INPUT,
                action=MASK_SENSITIVE_DATA,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="detect sensitive data on output",
                direction=RailDirection.OUTPUT,
                action=DETECT_SENSITIVE_DATA,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
            ),
            RailSurface(
                name="mask sensitive data on output",
                direction=RailDirection.OUTPUT,
                action=MASK_SENSITIVE_DATA,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
            RailSurface(
                name="detect sensitive data on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=DETECT_SENSITIVE_DATA,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
            ),
            RailSurface(
                name="mask sensitive data on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=MASK_SENSITIVE_DATA,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
        requirements=RailRequirements(
            extras=("sdd",),
            models=(ModelRequirement(type="spacy:en_core_web_lg", required=True),),
            optional_dependencies=("presidio-analyzer", "presidio-anonymizer", "spacy"),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True, sends_retrieved_chunks=True),
    ),
)
