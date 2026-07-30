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
    ModelRequirement,
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

CONTENT_SAFETY_CHECK_INPUT = ActionRef(
    name="content_safety_check_input",
    target="nemoguardrails.library.content_safety.actions:content_safety_check_input",
)
CONTENT_SAFETY_CHECK_OUTPUT = ActionRef(
    name="content_safety_check_output",
    target="nemoguardrails.library.content_safety.actions:content_safety_check_output",
)
DETECT_LANGUAGE = ActionRef(
    name="detect_language",
    target="nemoguardrails.library.content_safety.actions:detect_language",
)
RAIL = RailManifest(
    name="content_safety",
    metadata=RailMetadata(
        display_name="Content Safety",
        description="Checks user input and bot output for safety policy violations.",
        categories=("input", "output"),
        capabilities=("allow", "block", "classify", "content_safety", "moderate"),
        tags=("nemoguard", "moderation", "safety"),
        docs_url="docs/configure-rails/guardrail-catalog/content-safety.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="content_safety",
            spec=ConfigSpecRef(target="nemoguardrails.library.content_safety.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=("content safety check input", "content safety check output"),
        ),
        actions=RailActions(
            refs=(
                CONTENT_SAFETY_CHECK_INPUT,
                CONTENT_SAFETY_CHECK_OUTPUT,
                DETECT_LANGUAGE,
            ),
        ),
        surfaces=(
            RailSurface(
                name="content safety check input",
                direction=RailDirection.INPUT,
                action=CONTENT_SAFETY_CHECK_INPUT,
                bindings=(Binding.surface_param("model_name", "model"),),
            ),
            RailSurface(
                name="content safety check output",
                direction=RailDirection.OUTPUT,
                action=CONTENT_SAFETY_CHECK_OUTPUT,
                bindings=(Binding.surface_param("model_name", "model"),),
            ),
        ),
        requirements=RailRequirements(
            models=(ModelRequirement(type="content_safety", required=True),),
            extras=("multilingual",),
        ),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=True),
    ),
)
