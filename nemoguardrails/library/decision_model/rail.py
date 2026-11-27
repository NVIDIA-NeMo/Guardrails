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
    ServiceRequirement,
)

CHECK = ActionRef(
    name="decision_model_check_output",
    target="nemoguardrails.library.decision_model.actions:decision_model_check_output",
)
RAIL = RailManifest(
    name="decision_model",
    metadata=RailMetadata(
        display_name="Decision Model",
        lifecycle="experimental",
        description="Checks assistant replies with a native decision model.",
        categories=("output",),
        capabilities=("allow", "block", "classify"),
        tags=("decision_model",),
        docs_url="docs/configure-rails/guardrail-catalog/decision-model.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="decision_model",
            spec=ConfigSpecRef(target="nemoguardrails.library.decision_model.rail_config:build_config_spec"),
        ),
        actions=RailActions(refs=(CHECK,)),
        flows=RailFlows(flow_names=("decision model check output",)),
        surfaces=(
            RailSurface(
                name="decision model check output",
                direction=RailDirection.OUTPUT,
                action=CHECK,
                bindings=(
                    Binding.surface_param("rule", "rule"),
                    Binding.model_param("model_name", "model", resource="decision_model"),
                    Binding.context("bot_message", "bot_message"),
                ),
            ),
        ),
        requirements=RailRequirements(services=(ServiceRequirement(name="Configured decision endpoint"),)),
        privacy=RailPrivacy(sends_bot_text=True, remote_services=("Configured decision endpoint",)),
    ),
)
