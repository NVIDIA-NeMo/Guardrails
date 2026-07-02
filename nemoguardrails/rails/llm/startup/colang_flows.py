# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Colang flow loading, rail flow marking, and rail config validation."""

import importlib.resources as resources
import logging
from collections.abc import Iterator

try:
    from importlib.resources.abc import Traversable
except ImportError:
    from importlib.abc import Traversable

from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v1_0.runtime.flows import _normalize_flow_id
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)

_NEMOGUARDRAILS_PACKAGE = "nemoguardrails"

__all__ = [
    "load_default_colang_1_flows",
    "load_guardrails_library_flows_and_bot_messages",
    "mark_rail_flows_as_system_subflows",
    "validate_rail_flow_names",
]


def _package_resource(*parts: str) -> Traversable:
    resource = resources.files(_NEMOGUARDRAILS_PACKAGE)
    for part in parts:
        resource = resource.joinpath(part)
    return resource


def _iter_colang_resources(resource: Traversable) -> Iterator[Traversable]:
    if resource.is_file():
        if resource.name.endswith(".co"):
            yield resource
        return

    # Sort children by name so the library load order is deterministic and independent
    # of the filesystem's directory-iteration order. ``bot_messages`` are inserted on a
    # first-seen-wins basis, so a non-deterministic order can change which utterances win.
    children = sorted(resource.iterdir(), key=lambda child: child.name)
    for child in children:
        if child.is_file() and child.name.endswith(".co"):
            yield child

    for child in children:
        if child.is_dir():
            yield from _iter_colang_resources(child)


def load_default_colang_1_flows(config: RailsConfig) -> None:
    """Load the built-in Colang 1.0 LLM flows into the rails config."""
    if config.colang_version != "1.0":
        return

    default_flows_resource = _package_resource("rails", "llm", "llm_flows.co")
    default_flows_content = default_flows_resource.read_text(encoding="utf-8")
    default_flows = parse_colang_file(default_flows_resource.name, default_flows_content)["flows"]

    for flow_config in default_flows:
        flow_config["is_system_flow"] = True

    config.flows.extend(default_flows)


def load_guardrails_library_flows_and_bot_messages(config: RailsConfig) -> None:
    """Load Colang 1.0 flows and bot messages from the guardrails library."""
    if config.colang_version != "1.0":
        return

    library_resource = _package_resource("library")
    for colang_resource in _iter_colang_resources(library_resource):
        log.debug(f"Loading file: {colang_resource}")
        content = parse_colang_file(
            colang_resource.name,
            content=colang_resource.read_text(encoding="utf-8"),
            version=config.colang_version,
        )
        if not content:
            continue

        for flow_config in content["flows"]:
            flow_config["is_system_flow"] = True

        config.flows.extend(content["flows"])

        for message_id, utterances in content.get("bot_messages", {}).items():
            if message_id not in config.bot_messages:
                config.bot_messages[message_id] = utterances


def mark_rail_flows_as_system_subflows(config: RailsConfig) -> None:
    """Mark configured input, output, and retrieval rail flows as system subflows."""
    rail_flow_ids = config.rails.input.flows + config.rails.output.flows + config.rails.retrieval.flows

    for flow_config in config.flows:
        if flow_config.get("id") in rail_flow_ids:
            flow_config["is_system_flow"] = True
            flow_config["is_subflow"] = True


def validate_rail_flow_names(config: RailsConfig) -> None:
    """Validate that configured rail flows exist in the config."""
    if config.colang_version == "1.0":
        existing_flows_names = set([flow.get("id") for flow in config.flows])
    else:
        existing_flows_names = set([flow.get("name") for flow in config.flows])

    for flow_name in config.rails.input.flows:
        flow_name = _normalize_flow_id(flow_name)
        if flow_name not in existing_flows_names:
            raise InvalidRailsConfigurationError(f"The provided input rail flow `{flow_name}` does not exist")

    for flow_name in config.rails.output.flows:
        flow_name = _normalize_flow_id(flow_name)
        if flow_name not in existing_flows_names:
            raise InvalidRailsConfigurationError(f"The provided output rail flow `{flow_name}` does not exist")

    for flow_name in config.rails.retrieval.flows:
        if flow_name not in existing_flows_names:
            raise InvalidRailsConfigurationError(f"The provided retrieval rail flow `{flow_name}` does not exist")
