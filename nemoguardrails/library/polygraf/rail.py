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
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
    TransformTarget,
)

POLYGRAF_DETECT_PII = ActionRef(
    name="polygraf_detect_pii",
    target="nemoguardrails.library.polygraf.actions:polygraf_detect_pii",
)
POLYGRAF_MASK_PII = ActionRef(
    name="polygraf_mask_pii",
    target="nemoguardrails.library.polygraf.actions:polygraf_mask_pii",
)


def _bindings(source: str, context_key: str):
    return Binding.literal("source", source), Binding.context("text", context_key)


RAIL = RailManifest(
    name="polygraf",
    metadata=RailMetadata(
        display_name="Polygraf PII Detection",
        description="Detects and masks personally identifiable information using Polygraf.",
        categories=("input", "output", "retrieval"),
        capabilities=("allow", "block", "detect_pii", "mask", "transform"),
        tags=("pii", "polygraf", "privacy"),
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="polygraf",
            spec=ConfigSpecRef(target="nemoguardrails.library.polygraf.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "polygraf detect pii on input",
                "polygraf detect pii on output",
                "polygraf detect pii on retrieval",
                "polygraf mask pii on input",
                "polygraf mask pii on output",
                "polygraf mask pii on retrieval",
            ),
        ),
        actions=RailActions(refs=(POLYGRAF_DETECT_PII, POLYGRAF_MASK_PII)),
        surfaces=(
            RailSurface(
                name="polygraf detect pii on input",
                direction=RailDirection.INPUT,
                action=POLYGRAF_DETECT_PII,
                bindings=_bindings("input", "user_message"),
            ),
            RailSurface(
                name="polygraf detect pii on output",
                direction=RailDirection.OUTPUT,
                action=POLYGRAF_DETECT_PII,
                bindings=_bindings("output", "bot_message"),
            ),
            RailSurface(
                name="polygraf detect pii on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=POLYGRAF_DETECT_PII,
                bindings=_bindings("retrieval", "relevant_chunks"),
            ),
            RailSurface(
                name="polygraf mask pii on input",
                direction=RailDirection.INPUT,
                action=POLYGRAF_MASK_PII,
                bindings=_bindings("input", "user_message"),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="polygraf mask pii on output",
                direction=RailDirection.OUTPUT,
                action=POLYGRAF_MASK_PII,
                bindings=_bindings("output", "bot_message"),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
            RailSurface(
                name="polygraf mask pii on retrieval",
                direction=RailDirection.RETRIEVAL,
                action=POLYGRAF_MASK_PII,
                bindings=_bindings("retrieval", "relevant_chunks"),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
        requirements=RailRequirements(
            services=(ServiceRequirement(name="Polygraf endpoint", required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("Polygraf endpoint",),
        ),
    ),
)
