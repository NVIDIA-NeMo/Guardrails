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
    RailSpec,
    RailSurface,
    TransformTarget,
)

DETECT_REGEX_PATTERN = ActionRef(
    name="detect_regex_pattern",
    target="nemoguardrails.library.regex.actions:detect_regex_pattern",
)
RAIL = RailManifest(
    name="regex",
    metadata=RailMetadata(
        display_name="Regex Detection",
        description="Detects and blocks text matching configured regular expression patterns.",
        categories=("input", "output", "retrieval"),
        capabilities=("block", "classify", "moderate", "transform"),
        tags=("built-in", "regex", "pattern-matching"),
        docs_url="docs/configure-rails/guardrail-catalog/community/regex.md",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="regex_detection",
            spec=ConfigSpecRef(target="nemoguardrails.library.regex.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("regex check input", "regex check output", "regex check retrieval")),
        actions=RailActions(refs=(DETECT_REGEX_PATTERN,)),
        surfaces=(
            RailSurface(
                name="regex check input",
                direction=RailDirection.INPUT,
                action=DETECT_REGEX_PATTERN,
                bindings=(Binding.literal("source", "input"), Binding.context("text", "user_message")),
            ),
            RailSurface(
                name="regex check output",
                direction=RailDirection.OUTPUT,
                action=DETECT_REGEX_PATTERN,
                bindings=(Binding.literal("source", "output"), Binding.context("text", "bot_message")),
            ),
            RailSurface(
                name="regex check retrieval",
                direction=RailDirection.RETRIEVAL,
                action=DETECT_REGEX_PATTERN,
                bindings=(Binding.literal("source", "retrieval"), Binding.context("text", "relevant_chunks")),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
        privacy=RailPrivacy(),
    ),
)
