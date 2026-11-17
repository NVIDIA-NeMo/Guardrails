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

HIDDENLAYER_SERVICE = "HiddenLayer AI Runtime Security API"

HIDDENLAYER_EVALUATE = ActionRef(
    name="hiddenlayer_evaluate",
    target="nemoguardrails.library.hiddenlayer.actions:hiddenlayer_evaluate",
)
HIDDENLAYER_EVALUATE_TOOL_CALL = ActionRef(
    name="hiddenlayer_evaluate_tool_call",
    target="nemoguardrails.library.hiddenlayer.actions:hiddenlayer_evaluate_tool_call",
)
HIDDENLAYER_EVALUATE_TOOL_RESULT = ActionRef(
    name="hiddenlayer_evaluate_tool_result",
    target="nemoguardrails.library.hiddenlayer.actions:hiddenlayer_evaluate_tool_result",
)

RAIL = RailManifest(
    name="hiddenlayer",
    metadata=RailMetadata(
        display_name="HiddenLayer AI Runtime Security",
        description=(
            "Evaluates user input, bot output, tool calls and tool results with HiddenLayer AI Runtime "
            "Security, which answers each check with the action to apply and the payload to forward."
        ),
        categories=("input", "output", "tool_call", "tool_result"),
        capabilities=("allow", "block", "detect_jailbreak", "detect_pii", "moderate", "transform"),
        tags=("third-party", "api", "security", "tool-calling"),
        docs_url="docs/configure-rails/guardrail-catalog/community/hiddenlayer.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="hiddenlayer",
            spec=ConfigSpecRef(target="nemoguardrails.library.hiddenlayer.rail_config:build_config_spec"),
        ),
        # The tool surfaces run on IORails only, so the flow files define the input and output flows.
        flows=RailFlows(
            flow_names=(
                "hiddenlayer check input",
                "hiddenlayer check output",
                "hiddenlayer check tool call",
                "hiddenlayer check tool result",
            )
        ),
        actions=RailActions(
            refs=(HIDDENLAYER_EVALUATE, HIDDENLAYER_EVALUATE_TOOL_CALL, HIDDENLAYER_EVALUATE_TOOL_RESULT)
        ),
        surfaces=(
            RailSurface(
                name="hiddenlayer check input",
                direction=RailDirection.INPUT,
                action=HIDDENLAYER_EVALUATE,
                bindings=(Binding.literal("surface", "input"), Binding.context("text", "user_message")),
                transform_target=TransformTarget.USER_MESSAGE,
            ),
            RailSurface(
                name="hiddenlayer check output",
                direction=RailDirection.OUTPUT,
                action=HIDDENLAYER_EVALUATE,
                bindings=(Binding.literal("surface", "output"), Binding.context("text", "bot_message")),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
            RailSurface(
                name="hiddenlayer check tool call",
                direction=RailDirection.TOOL_CALL,
                action=HIDDENLAYER_EVALUATE_TOOL_CALL,
                bindings=(
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_definition", "tool_definition"),
                ),
            ),
            RailSurface(
                name="hiddenlayer check tool result",
                direction=RailDirection.TOOL_RESULT,
                action=HIDDENLAYER_EVALUATE_TOOL_RESULT,
                bindings=(
                    Binding.context("tool_call", "tool_call"),
                    Binding.context("tool_result", "tool_result"),
                    Binding.context("tool_definition", "tool_definition"),
                ),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(name="HIDDENLAYER_CLIENT_ID", required=True, description="OAuth2 client ID."),
                EnvVar(name="HIDDENLAYER_CLIENT_SECRET", required=True, description="OAuth2 client secret."),
                EnvVar(
                    name="HIDDENLAYER_BASE_URL",
                    required=False,
                    description="Fallback for api_url. Defaults to https://api.us.hiddenlayer.ai.",
                ),
                EnvVar(name="HIDDENLAYER_PROJECT_ID", required=False, description="Fallback for project_id."),
                EnvVar(
                    name="HIDDENLAYER_REQUESTER_ID",
                    required=False,
                    description="Fallback for requester_id. Defaults to hiddenlayer-nemo.",
                ),
                EnvVar(
                    name="HIDDENLAYER_SEND_SESSION_ID",
                    required=False,
                    description="Fallback for send_session_id. Defaults to sending the conversation ID.",
                ),
            ),
            services=(ServiceRequirement(name=HIDDENLAYER_SERVICE, required=True),),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_tool_call_text=True,
            sends_tool_result_text=True,
            remote_services=(HIDDENLAYER_SERVICE,),
            data_retention=(
                "Evaluated interactions, and the conversation ID they are grouped under, are stored by "
                "HiddenLayer under the tenant's retention policy."
            ),
        ),
    ),
)
