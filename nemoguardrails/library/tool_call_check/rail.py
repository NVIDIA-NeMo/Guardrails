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
    RailActions,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailSpec,
)

CHECK_TOOL_CALL = ActionRef(
    name="check_tool_call",
    target="nemoguardrails.library.tool_call_check.actions:check_tool_call",
)
GET_TOOL_CALL_NAME = ActionRef(
    name="get_tool_call_name",
    target="nemoguardrails.library.tool_call_check.actions:get_tool_call_name",
)
GET_PER_TOOL_OUTPUT_FLOWS = ActionRef(
    name="get_per_tool_output_flows",
    target="nemoguardrails.library.tool_call_check.actions:get_per_tool_output_flows",
)
GET_PER_TOOL_INPUT_FLOWS = ActionRef(
    name="get_per_tool_input_flows",
    target="nemoguardrails.library.tool_call_check.actions:get_per_tool_input_flows",
)

RAIL = RailManifest(
    name="tool_call_check",
    metadata=RailMetadata(
        display_name="Tool Call Check",
        description=(
            "Per-tool LLM-based check for tool calls and tool results. "
            "Evaluates each tool call or result against a named prompt task and "
            "blocks if the verdict is BLOCK."
        ),
        categories=("tool_output", "tool_input"),
        capabilities=("allow", "block"),
        tags=("tool", "check", "per-tool"),
    ),
    spec=RailSpec(
        flows=RailFlows(
            v1_files=("flows.v1.co",),
            flow_names=("check tool call",),
        ),
        actions=RailActions(
            refs=(
                CHECK_TOOL_CALL,
                GET_TOOL_CALL_NAME,
                GET_PER_TOOL_OUTPUT_FLOWS,
                GET_PER_TOOL_INPUT_FLOWS,
            ),
        ),
    ),
)
