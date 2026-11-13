# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        display_name="Agent Threat Rules Detection",
        description="Detects agent-specific threats in user messages with pyatr.",
        categories=("input",),
        capabilities=("block", "classify"),
        tags=("security", "agentic", "local"),
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="atr_detection",
            spec=ConfigSpecRef(
                target="nemoguardrails.library.atr.rail_config:build_config_spec"
            ),
        ),
        flows=RailFlows(flow_names=("atr check input",)),
        actions=RailActions(refs=(ATR_DETECTION,)),
        surfaces=(
            RailSurface(
                name="atr check input",
                direction=RailDirection.INPUT,
                action=ATR_DETECTION,
                bindings=(Binding.context("text", "user_message"),),
            ),
        ),
        requirements=RailRequirements(optional_dependencies=("pyatr",)),
        privacy=RailPrivacy(sends_user_text=True),
    ),
)
