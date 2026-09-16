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

ATR_DETECTION = ActionRef(
    name="atr_detection",
    target="nemoguardrails.library.atr.actions:atr_detection",
)

RAIL = RailManifest(
    name="atr",
    metadata=RailMetadata(
        display_name="Agent Threat Rules",
        description="Matches user input against the open Agent Threat Rules detection catalog.",
        long_description="Evaluates the user message against Agent Threat Rules, a "
        "community-maintained detection catalog for AI-agent attacks such as prompt "
        "injection, jailbreak, tool poisoning, MCP attacks, and skill compromise. "
        "Rules ship inside the optional `pyatr` package and are evaluated in-process, "
        "so the rail needs no API key, no service endpoint, and no network access, and "
        "the same input yields the same verdict on every run.",
        categories=("input",),
        capabilities=("allow", "block", "classify", "detect_jailbreak"),
        tags=("security", "agentic", "offline", "deterministic"),
        docs_url="docs/configure-rails/guardrail-catalog/agentic-security.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="atr",
            spec=ConfigSpecRef(target="nemoguardrails.library.atr.rail_config:build_config_spec"),
        ),
        flows=RailFlows(flow_names=("atr detection",)),
        actions=RailActions(refs=(ATR_DETECTION,)),
        surfaces=(
            RailSurface(
                name="atr detection",
                direction=RailDirection.INPUT,
                action=ATR_DETECTION,
                bindings=(Binding.context("text", "user_message"),),
            ),
        ),
        requirements=RailRequirements(optional_dependencies=("pyatr",)),
        # Everything is evaluated in-process: no text leaves the machine and no
        # remote service is contacted, so every disclosure flag stays false.
        privacy=RailPrivacy(),
    ),
)
