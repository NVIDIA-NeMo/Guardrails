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

SELF_CHECK_HALLUCINATION = ActionRef(
    name="self_check_hallucination",
    target="nemoguardrails.library.hallucination.actions:self_check_hallucination",
)

RAIL = RailManifest(
    name="hallucination",
    metadata=RailMetadata(
        display_name="Hallucination Detection",
        description="Checks bot output for hallucinations using LLM self-consistency.",
        categories=("output",),
        capabilities=("allow", "block", "fact_check"),
        tags=("self-check", "hallucination", "llm-prompt"),
        docs_url="docs/configure-rails/guardrail-catalog/fact-checking.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("hallucination warning", "self check hallucination")),
        actions=RailActions(refs=(SELF_CHECK_HALLUCINATION,)),
        surfaces=(
            RailSurface(
                name="self check hallucination",
                direction=RailDirection.OUTPUT,
                action=SELF_CHECK_HALLUCINATION,
                bindings=(
                    Binding.context("bot_message", "bot_message"),
                    Binding.context("last_bot_prompt", "_last_bot_prompt"),
                ),
            ),
        ),
        requirements=RailRequirements(models=(ModelRequirement(type="llm", required=True),)),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True),
    ),
)
