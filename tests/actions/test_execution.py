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

from typing import Optional

import pytest

from nemoguardrails.actions.execution import (
    ActionExecutionScope,
    action_parameter_providers,
    resolve_action_parameters,
)
from nemoguardrails.llm.models.resources import LLMModelResources
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.testing import FakeLLMModel


def _scope(
    *,
    main=None,
    specialized_models=None,
    parameters=None,
) -> ActionExecutionScope:
    config = RailsConfig(models=[])
    return ActionExecutionScope(
        config=config,
        context={"user_message": "hello"},
        events=[{"type": "UserMessage"}],
        llm_task_manager=LLMTaskManager(config),
        model_resources=LLMModelResources(
            main=main,
            specialized_models=specialized_models or {},
            caches={},
        ),
        parameters=parameters or {},
    )


def test_action_parameter_providers_expose_shared_resources_and_model_aliases():
    main = FakeLLMModel(responses=[])
    llama_guard = FakeLLMModel(responses=[])
    scope = _scope(
        main=main,
        specialized_models={"llama_guard": llama_guard},
        parameters={"custom": "value"},
    )

    providers = action_parameter_providers(scope)

    assert providers["config"] is scope.config
    assert providers["context"] == {"user_message": "hello"}
    assert providers["events"] == [{"type": "UserMessage"}]
    assert providers["llm_task_manager"] is scope.llm_task_manager
    assert providers["llm"] is main
    assert providers["llms"] == {"llama_guard": llama_guard}
    assert providers["model_caches"] == {}
    assert providers["llama_guard_llm"] is llama_guard
    assert providers["custom"] == "value"


def test_action_specific_model_replaces_main_provider():
    main = FakeLLMModel(responses=[])
    self_check = FakeLLMModel(responses=[])
    scope = _scope(
        main=main,
        specialized_models={"self_check_input": self_check},
    )

    providers = action_parameter_providers(scope, action_name="self_check_input")

    assert providers["llm"] is self_check


def test_resolve_action_parameters_injects_only_declared_dependencies():
    main = FakeLLMModel(responses=[])
    llama_guard = FakeLLMModel(responses=[])
    scope = _scope(
        main=main,
        specialized_models={"llama_guard": llama_guard},
        parameters={"unused": "ignored"},
    )

    async def check(
        text,
        config,
        context,
        llm,
        llama_guard_llm,
        optional: Optional[str] = None,
        **kwargs,
    ):
        return None

    parameters = resolve_action_parameters(
        check,
        {"text": "bound"},
        scope,
        action_name="check",
    )

    assert parameters == {
        "text": "bound",
        "config": scope.config,
        "context": {"user_message": "hello"},
        "llm": main,
        "llama_guard_llm": llama_guard,
    }


def test_explicit_action_parameters_take_precedence_over_providers():
    scope = _scope(parameters={"config": "registered"})

    async def check(config):
        return None

    parameters = resolve_action_parameters(check, {"config": "explicit"}, scope)

    assert parameters == {"config": "explicit"}


def test_missing_required_action_parameter_fails_resolution():
    scope = _scope()

    async def check(required):
        return None

    with pytest.raises(TypeError, match="required parameter 'required'"):
        resolve_action_parameters(check, {}, scope, action_name="check")
