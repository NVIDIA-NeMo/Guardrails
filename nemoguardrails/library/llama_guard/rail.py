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

LLAMA_GUARD_CHECK_INPUT = ActionRef(
    name="llama_guard_check_input",
    target="nemoguardrails.library.llama_guard.actions:llama_guard_check_input",
)
LLAMA_GUARD_CHECK_OUTPUT = ActionRef(
    name="llama_guard_check_output",
    target="nemoguardrails.library.llama_guard.actions:llama_guard_check_output",
)

RAIL = RailManifest(
    name="llama_guard",
    metadata=RailMetadata(
        display_name="Llama Guard",
        description="Checks user input and bot output with a configured Llama Guard model.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "moderate"),
        tags=("llama-guard", "moderation", "safety"),
        docs_url="docs/configure-rails/guardrail-catalog/community/llama-guard.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("llama guard check input", "llama guard check output")),
        actions=RailActions(refs=(LLAMA_GUARD_CHECK_INPUT, LLAMA_GUARD_CHECK_OUTPUT)),
        surfaces=(
            RailSurface(
                name="llama guard check input",
                direction=RailDirection.INPUT,
                action=LLAMA_GUARD_CHECK_INPUT,
                bindings=(
                    Binding.model("model_name", "llama_guard"),
                    Binding.context("user_message", "user_message"),
                ),
            ),
            RailSurface(
                name="llama guard check output",
                direction=RailDirection.OUTPUT,
                action=LLAMA_GUARD_CHECK_OUTPUT,
                bindings=(
                    Binding.model("model_name", "llama_guard"),
                    Binding.context("user_message", "user_message", required=False),
                    Binding.context("bot_message", "bot_message"),
                ),
            ),
        ),
        requirements=RailRequirements(models=(ModelRequirement(type="llama_guard", required=True),)),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True),
    ),
)
