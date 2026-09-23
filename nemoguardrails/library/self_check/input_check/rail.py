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
    ModelRequirement,
    RailActions,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
)

SELF_CHECK_INPUT = ActionRef(
    name="self_check_input",
    target="nemoguardrails.library.self_check.input_check.actions:self_check_input",
)

RAIL = RailManifest(
    name="self_check.input_check",
    metadata=RailMetadata(
        display_name="Self-Check Input",
        description="Checks whether user input should be allowed using an LLM prompt.",
        categories=("input",),
        capabilities=("allow", "block", "moderate"),
        tags=("self-check", "input", "llm-prompt"),
        docs_url="docs/configure-rails/guardrail-catalog/self-check.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("self check input",)),
        actions=RailActions(refs=(SELF_CHECK_INPUT,)),
        surfaces=(
            RailSurface(
                name="self check input",
                direction=RailDirection.INPUT,
                action=SELF_CHECK_INPUT,
                bindings=(
                    Binding.surface_param(action_param="variant", name="variant", required=False),
                    Binding.context("user_message", "user_message"),
                ),
            ),
        ),
        requirements=RailRequirements(models=(ModelRequirement(type="llm", required=True),)),
        privacy=RailPrivacy(sends_user_text=True),
    ),
)
