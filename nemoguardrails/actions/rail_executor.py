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

"""Backend-neutral execution of configured rail manifest surfaces."""

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.actions.rail_outcome import RailOutcome, require_rail_outcome
from nemoguardrails.manifests import RailCatalog, RailDirection, RailSurface, parse_configured_surface

__all__ = ["RailInvocation", "resolve_rail_invocation", "execute_rail"]


@dataclass(frozen=True, slots=True)
class RailInvocation:
    """One resolved surface invocation independent of an execution backend."""

    surface: RailSurface
    action_params: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    events: Sequence[dict[str, Any]] | None = None
    config: Any = None
    llm_task_manager: Any = None
    registered_action_params: Mapping[str, Any] = field(default_factory=dict)


def _resolve_bindings(
    surface: RailSurface,
    surface_params: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_surface_params = {
        binding.key for binding in surface.bindings if binding.kind == "surface_param" and binding.key is not None
    }
    unexpected = sorted(set(surface_params) - accepted_surface_params)
    if unexpected:
        raise ValueError(f"Surface {surface.name!r} received unsupported parameters: {unexpected}.")

    resolved: dict[str, Any] = {}
    for binding in surface.bindings:
        if binding.kind == "literal":
            resolved[binding.action_param] = binding.value
            continue

        source = surface_params if binding.kind == "surface_param" else context
        if binding.key in source:
            resolved[binding.action_param] = source[binding.key]
        elif binding.required:
            raise ValueError(f"Surface {surface.name!r} requires {binding.kind} value {binding.key!r}.")

    return resolved


def resolve_rail_invocation(
    catalog: RailCatalog,
    direction: RailDirection | str,
    configured_surface: str,
    *,
    context: Mapping[str, Any] | None = None,
    events: Sequence[dict[str, Any]] | None = None,
    config: Any = None,
    llm_task_manager: Any = None,
    registered_action_params: Mapping[str, Any] | None = None,
) -> RailInvocation:
    """Resolve a configured surface reference and its declared bindings."""
    try:
        manifest_direction = RailDirection(direction)
    except ValueError as error:
        raise ValueError(f"Unsupported rail direction {direction!r}.") from error

    name, surface_params = parse_configured_surface(configured_surface)
    surface = catalog.surfaces(manifest_direction).get((manifest_direction, name))
    if surface is None:
        available = sorted(surface_name for _, surface_name in catalog.surfaces(manifest_direction))
        raise ValueError(
            f"Rail surface {name!r} is not declared for direction {manifest_direction.value!r}; "
            f"available surfaces: {available}."
        )

    invocation_context = dict(context or {})
    return RailInvocation(
        surface=surface,
        action_params=_resolve_bindings(surface, surface_params, invocation_context),
        context=invocation_context,
        events=events,
        config=config,
        llm_task_manager=llm_task_manager,
        registered_action_params=dict(registered_action_params or {}),
    )


def _execution_params(invocation: RailInvocation, action: Any) -> dict[str, Any]:
    parameters = inspect.signature(action).parameters
    params = dict(invocation.action_params)

    for parameter_name in parameters:
        if parameter_name.startswith("__context__"):
            params[parameter_name] = invocation.context.get(parameter_name[11:])

    special_values = {
        "context": dict(invocation.context),
        "events": list(invocation.events) if invocation.events is not None else None,
        "config": invocation.config,
        "llm_task_manager": invocation.llm_task_manager,
    }
    for name, value in special_values.items():
        if name in parameters:
            params[name] = value

    for name, value in invocation.registered_action_params.items():
        if name in parameters:
            params[name] = value

    action_llm = invocation.registered_action_params.get(f"{invocation.surface.action.name}_llm")
    if "llm" in params and action_llm is not None:
        params["llm"] = action_llm

    return params


async def execute_rail(invocation: RailInvocation, dispatcher: ActionDispatcher) -> RailOutcome:
    """Execute a resolved surface action and require a ``RailOutcome`` verdict."""
    action_name = invocation.surface.action.name
    action = dispatcher.get_action(action_name)
    if action is None:
        raise ValueError(f"Rail action {action_name!r} is not registered.")

    result, status = await dispatcher.execute_action(action_name, _execution_params(invocation, action))
    if status != "success":
        raise RuntimeError(f"Rail action {action_name!r} failed with status {status!r}.")
    outcome = require_rail_outcome(result)
    if outcome.is_transform:
        targets = {transform.target for transform in outcome.transforms}
        if invocation.surface.transform_target is None or targets != {invocation.surface.transform_target}:
            raise ValueError(
                f"Rail surface {invocation.surface.name!r} declared transform target "
                f"{invocation.surface.transform_target!r}, but the action returned {sorted(target.value for target in targets)}."
            )
    return outcome
