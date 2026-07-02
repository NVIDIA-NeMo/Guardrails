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

SELF_CHECK_FACTS = ActionRef(
    name="self_check_facts",
    target="nemoguardrails.library.self_check.facts.actions:self_check_facts",
)

RAIL = RailManifest(
    name="self_check.facts",
    metadata=RailMetadata(
        display_name="Self-Check Facts",
        description="Checks whether bot output is grounded in relevant chunks using an LLM prompt.",
        categories=("output",),
        capabilities=("allow", "block", "fact_check"),
        tags=("self-check", "fact-checking", "rag", "llm-prompt"),
        docs_url="docs/configure-rails/guardrail-catalog/fact-checking.md",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("self check facts",)),
        actions=RailActions(refs=(SELF_CHECK_FACTS,)),
        surfaces=(RailSurface(name="self check facts", direction=RailDirection.OUTPUT, action=SELF_CHECK_FACTS),),
        requirements=RailRequirements(models=(ModelRequirement(type="llm", required=True),)),
        privacy=RailPrivacy(sends_bot_text=True, sends_retrieved_chunks=True),
    ),
)
