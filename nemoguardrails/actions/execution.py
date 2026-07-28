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

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

if TYPE_CHECKING:
    from nemoguardrails.actions.action_dispatcher import RegisteredAction
    from nemoguardrails.llm.models.resources import LLMModelResources
    from nemoguardrails.llm.taskmanager import LLMTaskManager
    from nemoguardrails.rails.llm.config import RailsConfig


@dataclass(frozen=True, slots=True)
class ActionExecutionScope:
    config: RailsConfig
    context: Mapping[str, Any]
    events: Sequence[Any]
    llm_task_manager: LLMTaskManager
    model_resources: LLMModelResources
    parameters: Mapping[str, Any] = field(default_factory=dict)


def action_parameter_providers(
    scope: ActionExecutionScope,
    *,
    action_name: str | None = None,
) -> dict[str, Any]:
    specialized_models = dict(scope.model_resources.specialized_models)
    providers: dict[str, Any] = {
        "config": scope.config,
        "context": dict(scope.context),
        "events": list(scope.events),
        "llm_task_manager": scope.llm_task_manager,
        "llm": scope.model_resources.main,
        "llms": specialized_models,
        "model_caches": dict(scope.model_resources.caches),
    }
    providers.update({f"{model_type}_llm": model for model_type, model in specialized_models.items()})
    providers.update(scope.parameters)
    if action_name is not None:
        action_model = specialized_models.get(action_name)
        if action_model is not None:
            providers["llm"] = action_model
    return providers


def resolve_action_parameters(
    action: RegisteredAction,
    explicit_parameters: Mapping[str, Any],
    scope: ActionExecutionScope,
    *,
    action_name: str | None = None,
) -> dict[str, Any]:
    parameters = dict(explicit_parameters)
    providers = action_parameter_providers(scope, action_name=action_name)

    try:
        signature = inspect.signature(cast(Any, action))
    except (TypeError, ValueError) as error:
        raise TypeError(f"Cannot inspect action {action_name or action!r}") from error

    for name, parameter in signature.parameters.items():
        if name in parameters:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in providers and providers[name] is not None:
            parameters[name] = providers[name]
            continue
        if parameter.default is inspect.Parameter.empty:
            raise TypeError(f"Cannot supply required parameter {name!r} to action {action_name or action!r}")

    return parameters


__all__ = [
    "ActionExecutionScope",
    "action_parameter_providers",
    "resolve_action_parameters",
]
