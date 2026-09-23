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
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
)

ALIGNSCORE_CHECK_FACTS = ActionRef(
    name="alignscore_check_facts",
    target="nemoguardrails.library.factchecking.align_score.actions:alignscore_check_facts",
)
ALIGNSCORE_REQUEST = ActionRef(
    name="alignscore request",
    target="nemoguardrails.library.factchecking.align_score.request:alignscore_request",
)

RAIL = RailManifest(
    name="factchecking.align_score",
    metadata=RailMetadata(
        display_name="AlignScore Fact Checking",
        description="Checks whether bot output is grounded in retrieved chunks using an AlignScore endpoint.",
        categories=("output",),
        capabilities=("allow", "block", "fact_check"),
        tags=("alignscore", "fact-checking", "rag", "grounding"),
        docs_url="docs/configure-rails/guardrail-catalog/community/alignscore.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("alignscore check facts",)),
        actions=RailActions(refs=(ALIGNSCORE_CHECK_FACTS, ALIGNSCORE_REQUEST)),
        surfaces=(
            RailSurface(
                name="alignscore check facts",
                direction=RailDirection.OUTPUT,
                action=ALIGNSCORE_CHECK_FACTS,
                bindings=(
                    Binding.context("relevant_chunks", "relevant_chunks"),
                    Binding.context("bot_message", "bot_message"),
                ),
            ),
        ),
        requirements=RailRequirements(services=(ServiceRequirement(name="AlignScore HTTP service", required=True),)),
        privacy=RailPrivacy(
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=("AlignScore HTTP service",),
        ),
    ),
)
