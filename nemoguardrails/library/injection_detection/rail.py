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
    TransformTarget,
)

INJECTION_DETECTION = ActionRef(
    name="injection_detection",
    target="nemoguardrails.library.injection_detection.actions:injection_detection",
)

RAIL = RailManifest(
    name="injection_detection",
    metadata=RailMetadata(
        display_name="Injection Detection",
        description="Detects and mitigates injection patterns in bot output.",
        categories=("output", "tool_output"),
        capabilities=("allow", "block", "classify", "transform"),
        tags=("security", "agentic", "yara"),
        docs_url="docs/configure-rails/guardrail-catalog/agentic-security.md",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="injection_detection",
            spec=ConfigSpecRef(target="nemoguardrails.library.injection_detection.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("injection detection",)),
        actions=RailActions(refs=(INJECTION_DETECTION,)),
        surfaces=(
            RailSurface(
                name="injection detection",
                direction=RailDirection.OUTPUT,
                action=INJECTION_DETECTION,
                bindings=(Binding.context("text", "bot_message"),),
                transform_target=TransformTarget.BOT_MESSAGE,
            ),
        ),
        requirements=RailRequirements(optional_dependencies=("yara-python",)),
        privacy=RailPrivacy(sends_bot_text=True),
    ),
)
