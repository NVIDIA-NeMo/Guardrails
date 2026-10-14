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
    RailActions,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailSpec,
    RailSurface,
)

TOOL_SAFETY_CHECK_OUTPUT = ActionRef(
    name="tool_safety_check_tool_call",
    target="nemoguardrails.library.tool_safety_check.actions:tool_safety_check_tool_call",
)
TOOL_SAFETY_CHECK_INPUT = ActionRef(
    name="tool_safety_check_tool_result",
    target="nemoguardrails.library.tool_safety_check.actions:tool_safety_check_tool_result",
)
RAIL = RailManifest(
    name="tool_safety_check",
    metadata=RailMetadata(
        display_name="Tool Safety Check",
        description="Checks a tool call's arguments or a tool result's content for safety policy violations using an LLM prompt.",
        categories=("tool_call", "tool_result"),
        capabilities=("allow", "block", "moderate"),
        tags=("moderation", "safety", "tool-calling"),
        docs_url="docs/configure-rails/guardrail-catalog/tool-safety-check.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(
            # Every surface here is TOOL_CALL/TOOL_RESULT (IORails-only), so this rail
            # ships no Colang flow definitions at all.
            files=(),
            v1_files=(),
            flow_names=(
                "tool safety check tool call",
                "tool safety check tool result",
            ),
        ),
        actions=RailActions(refs=(TOOL_SAFETY_CHECK_OUTPUT, TOOL_SAFETY_CHECK_INPUT)),
        surfaces=(
            RailSurface(
                name="tool safety check tool call",
                direction=RailDirection.TOOL_CALL,
                action=TOOL_SAFETY_CHECK_OUTPUT,
                bindings=(
                    Binding.model_param("model_name", "model"),
                    Binding.surface_param("variant", "variant"),
                    Binding.surface_param("argument_name", "argument", required=False),
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_definition", "tool_definition"),
                ),
            ),
            RailSurface(
                name="tool safety check tool result",
                direction=RailDirection.TOOL_RESULT,
                action=TOOL_SAFETY_CHECK_INPUT,
                bindings=(
                    Binding.model_param("model_name", "model"),
                    Binding.surface_param("variant", "variant"),
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_result", "tool_result"),
                ),
            ),
        ),
        privacy=RailPrivacy(sends_tool_call_text=True, sends_tool_result_text=True),
    ),
)
