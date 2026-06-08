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

from typing import Any, cast
from unittest.mock import MagicMock, patch

from nemoguardrails.rails.llm.config import Model, RailsConfig
from nemoguardrails.rails.llm.startup.llm_action_models import (
    load_llm_action_models,
    model_kwargs_from_config,
    sync_update_llm_bindings,
)
from nemoguardrails.types import LLMModel


class RuntimeWithActionParams:
    def __init__(self):
        self.registered_action_params = {}

    def register_action_param(self, name, value):
        self.registered_action_params[name] = value


class FakeRails:
    content_safety_llm: Any

    def __init__(self, config: RailsConfig, llm: Any, runtime: RuntimeWithActionParams):
        self.config = config
        self.llm = llm
        self.runtime = runtime
        self._llm_generation_actions = MagicMock()


def test_model_kwargs_from_config_adds_api_key_from_environment():
    model = Model(
        type="main",
        engine="openai",
        model="gpt-3.5-turbo",
        api_key_env_var="TEST_OPENAI_KEY",
        parameters={"temperature": 0.7},
    )

    with patch.dict("os.environ", {"TEST_OPENAI_KEY": "secret-api-key-from-env"}):
        kwargs = model_kwargs_from_config(model)

    assert kwargs["api_key"] == "secret-api-key-from-env"
    assert kwargs["temperature"] == 0.7


def test_model_kwargs_from_config_omits_missing_api_key_environment_variable():
    model = Model(
        type="main",
        engine="openai",
        model="gpt-3.5-turbo",
        api_key_env_var="MISSING_OPENAI_KEY",
        parameters={"temperature": 0.5},
    )

    with patch.dict("os.environ", {}, clear=True):
        kwargs = model_kwargs_from_config(model)

    assert "api_key" not in kwargs
    assert kwargs["temperature"] == 0.5


def test_model_kwargs_from_config_preserves_direct_api_key_parameter():
    model = Model(
        type="main",
        engine="openai",
        model="gpt-3.5-turbo",
        parameters={"api_key": "direct-key", "temperature": 0.3},
    )

    kwargs = model_kwargs_from_config(model)

    assert kwargs["api_key"] == "direct-key"
    assert kwargs["temperature"] == 0.3


def test_load_llm_action_models_uses_constructor_llm_and_loads_action_models():
    injected_llm = object()
    content_safety_llm = object()
    init_llm = MagicMock(return_value=content_safety_llm)
    rails = FakeRails(
        config=RailsConfig(
            models=[
                Model(type="main", engine="fake", model="main-model"),
                Model(type="content_safety", engine="fake", model="content-safety-model"),
            ]
        ),
        llm=injected_llm,
        runtime=RuntimeWithActionParams(),
    )

    load_llm_action_models(rails, init_llm=init_llm)

    init_llm.assert_called_once_with(
        model_name="content-safety-model",
        provider_name="fake",
        mode="chat",
        kwargs={},
    )
    assert rails.llm is injected_llm
    assert rails.content_safety_llm is content_safety_llm
    assert rails.runtime.registered_action_params["llm"] is injected_llm
    assert rails.runtime.registered_action_params["content_safety_llm"] is content_safety_llm
    assert rails.runtime.registered_action_params["llms"] == {"content_safety": content_safety_llm}


def test_load_llm_action_models_initializes_main_llm_from_config():
    main_llm = object()
    init_llm = MagicMock(return_value=main_llm)
    rails = FakeRails(
        config=RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="fake",
                    model="main-model",
                    parameters={"temperature": 0.2},
                )
            ]
        ),
        llm=None,
        runtime=RuntimeWithActionParams(),
    )

    load_llm_action_models(rails, init_llm=init_llm)

    init_llm.assert_called_once_with(
        model_name="main-model",
        provider_name="fake",
        mode="chat",
        kwargs={"temperature": 0.2},
    )
    assert rails.llm is main_llm
    assert rails.runtime.registered_action_params["llm"] is main_llm
    assert rails.runtime.registered_action_params["llms"] == {}


def test_sync_update_llm_bindings_updates_llm_generation_actions_and_runtime_param():
    initial_llm = cast(LLMModel, object())
    updated_llm = cast(LLMModel, object())
    rails = FakeRails(
        config=RailsConfig(models=[]),
        llm=initial_llm,
        runtime=RuntimeWithActionParams(),
    )

    sync_update_llm_bindings(rails, updated_llm)

    assert rails.llm is updated_llm
    assert rails._llm_generation_actions.llm is updated_llm
    assert rails.runtime.registered_action_params["llm"] is updated_llm
