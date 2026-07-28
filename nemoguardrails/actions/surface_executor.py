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

from __future__ import annotations

from typing import Any, Mapping

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.actions.execution import (
    ActionExecutionScope,
    resolve_action_parameters,
)
from nemoguardrails.actions.rail_outcome import RailOutcome, require_rail_outcome
from nemoguardrails.manifests import (
    RailCatalog,
    RailDirection,
    RailSurface,
    parse_configured_surface,
)


def resolve_surface_parameters(
    surface: RailSurface,
    surface_parameters: Mapping[str, str],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}

    for binding in surface.bindings:
        if binding.kind == "literal":
            parameters[binding.action_param] = binding.value
            continue

        source = surface_parameters if binding.kind == "surface_param" else context
        if binding.key in source:
            parameters[binding.action_param] = source[binding.key]
            continue
        if binding.required:
            raise ValueError(f"Missing required {binding.kind} value {binding.key!r} for surface {surface.name!r}")

    return parameters


class RailSurfaceExecutor:
    def __init__(
        self,
        *,
        catalog: RailCatalog,
        dispatcher: ActionDispatcher,
    ) -> None:
        self.catalog = catalog
        self.dispatcher = dispatcher

    async def execute(
        self,
        configured_surface: str,
        *,
        direction: RailDirection,
        scope: ActionExecutionScope,
    ) -> RailOutcome:
        surface_name, surface_parameters = parse_configured_surface(configured_surface)
        surface = self.catalog.surfaces(direction).get((direction, surface_name))
        if surface is None:
            raise LookupError(f"No {direction.value} rail surface named {surface_name!r}")

        action = self.dispatcher.get_action(surface.action.name)
        if action is None:
            raise LookupError(f"Action {surface.action.name!r} is unavailable")

        explicit_parameters = resolve_surface_parameters(
            surface,
            surface_parameters,
            scope.context,
        )
        parameters = resolve_action_parameters(
            action,
            explicit_parameters,
            scope,
            action_name=surface.action.name,
        )
        result, status = await self.dispatcher.execute_action(surface.action.name, parameters)
        if status != "success":
            raise RuntimeError(f"Action {surface.action.name!r} failed")
        return require_rail_outcome(result)


__all__ = [
    "RailSurfaceExecutor",
    "resolve_surface_parameters",
]
