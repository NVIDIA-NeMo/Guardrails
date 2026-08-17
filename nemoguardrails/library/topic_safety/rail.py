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

TOPIC_SAFETY_CHECK_INPUT = ActionRef(
    name="topic_safety_check_input",
    target="nemoguardrails.library.topic_safety.actions:topic_safety_check_input",
)
RAIL = RailManifest(
    name="topic_safety",
    metadata=RailMetadata(
        display_name="Topic Safety",
        description="Checks whether user input stays within configured topical guidelines.",
        categories=("input",),
        capabilities=("allow", "block", "classify", "topic_control"),
        tags=("nemoguard", "topic-control"),
        docs_url="docs/configure-rails/guardrail-catalog/topic-control.mdx",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("topic safety check input",)),
        actions=RailActions(
            refs=(TOPIC_SAFETY_CHECK_INPUT,),
        ),
        surfaces=(
            RailSurface(
                name="topic safety check input",
                direction=RailDirection.INPUT,
                action=TOPIC_SAFETY_CHECK_INPUT,
                bindings=(
                    Binding.surface_param("model_name", "model"),
                    Binding.context("user_message", "user_message"),
                ),
            ),
        ),
        requirements=RailRequirements(models=(ModelRequirement(type="topic_control", required=True),)),
        privacy=RailPrivacy(sends_user_text=True),
    ),
)
