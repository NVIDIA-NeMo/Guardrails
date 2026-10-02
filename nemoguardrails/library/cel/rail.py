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
)

EVALUATE_TOOL_OUTPUT_CEL = ActionRef(
    name="evaluate_tool_output_cel",
    target="nemoguardrails.library.cel.actions:evaluate_tool_output_cel",
)
EVALUATE_TOOL_INPUT_CEL = ActionRef(
    name="evaluate_tool_input_cel",
    target="nemoguardrails.library.cel.actions:evaluate_tool_input_cel",
)
RAIL = RailManifest(
    name="cel",
    metadata=RailMetadata(
        display_name="Common Expression Language (CEL)",
        description="Detects and blocks content matching configured Common Expression Language (CEL) expressions.",
        categories=("tool_output", "tool_input"),
        capabilities=("allow", "block", "moderate"),
        tags=("built-in", "cel", "tool-calling"),
        docs_url="docs/configure-rails/guardrail-catalog/cel.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="cel",
            spec=ConfigSpecRef(target="nemoguardrails.library.cel.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            # Every surface here is TOOL_OUTPUT/TOOL_INPUT (IORails-only), so this rail
            # ships no Colang flow definitions at all.
            files=(),
            v1_files=(),
            flow_names=(
                "cel check tool output",
                "cel check tool input",
            ),
        ),
        actions=RailActions(refs=(EVALUATE_TOOL_OUTPUT_CEL, EVALUATE_TOOL_INPUT_CEL)),
        surfaces=(
            RailSurface(
                name="cel check tool output",
                direction=RailDirection.TOOL_OUTPUT,
                action=EVALUATE_TOOL_OUTPUT_CEL,
                bindings=(
                    Binding.literal("source", "tool_output"),
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_definition", "tool_definition"),
                ),
            ),
            RailSurface(
                name="cel check tool input",
                direction=RailDirection.TOOL_INPUT,
                action=EVALUATE_TOOL_INPUT_CEL,
                bindings=(
                    Binding.literal("source", "tool_input"),
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_result", "tool_result"),
                ),
            ),
        ),
        requirements=RailRequirements(extras=("cel",), optional_dependencies=("common-expression-language",)),
        privacy=RailPrivacy(),
    ),
)
