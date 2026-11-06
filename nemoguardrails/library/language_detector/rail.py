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

LANGUAGE_DETECTOR_CHECK_INPUT = ActionRef(
    name="language_detector_check_input",
    target="nemoguardrails.library.language_detector.actions:language_detector_check_input",
)
RAIL = RailManifest(
    name="language_detector",
    metadata=RailMetadata(
        display_name="Language Detector",
        description="Blocks user messages written in languages outside the allowed set.",
        categories=("input",),
        capabilities=("allow", "block", "classify"),
        tags=("nemoguard", "language", "detection"),
        docs_url="docs/configure-rails/guardrail-catalog/language-detector.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="language_detector",
            spec=ConfigSpecRef(target="nemoguardrails.library.language_detector.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=("language detector check input",),
        ),
        actions=RailActions(
            refs=(LANGUAGE_DETECTOR_CHECK_INPUT,),
        ),
        surfaces=(
            RailSurface(
                name="language detector check input",
                direction=RailDirection.INPUT,
                action=LANGUAGE_DETECTOR_CHECK_INPUT,
                bindings=(Binding.context("user_message", "user_message"),),
            ),
        ),
        requirements=RailRequirements(optional_dependencies=("fast-langdetect",), extras=("multilingual",)),
        privacy=RailPrivacy(sends_user_text=True, sends_bot_text=False),
    ),
)
