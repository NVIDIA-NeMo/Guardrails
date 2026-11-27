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

"""Characterize model composition through the configured framework before extraction."""

from unittest.mock import Mock, call

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.frameworks import register_framework, set_default_framework
from nemoguardrails.llm.frameworks import registry as framework_registry
from nemoguardrails.llm.frameworks.default import DefaultFramework
from nemoguardrails.testing.fake_model import FakeLLMModel


@pytest.fixture
def model_framework(monkeypatch):
    monkeypatch.setattr(framework_registry, "_frameworks", {})
    monkeypatch.setattr(framework_registry, "_default_framework", "default")
    framework = DefaultFramework()
    register_framework("composition-contract", framework)
    set_default_framework("composition-contract")
    return framework


@pytest.mark.parametrize("inject_main", [False, True])
def test_framework_models_reach_all_action_views(model_framework, monkeypatch, inject_main):
    main = FakeLLMModel(responses=[])
    safety = FakeLLMModel(responses=[])
    topic = FakeLLMModel(responses=[])
    models = {"configured-main": main, "safety-model": safety, "topic-model": topic}
    provider = Mock(side_effect=lambda model, **kwargs: models[model])
    model_framework.register_provider("composition-provider", provider)
    create_model = Mock(wraps=model_framework.create_model)
    monkeypatch.setattr(model_framework, "create_model", create_model)
    config = RailsConfig.from_content(
        yaml_content="""
models:
  - type: main
    engine: composition-provider
    model: configured-main
    mode: text
    parameters:
      temperature: 0.7
  - type: content_safety
    engine: composition-provider
    model: safety-model
    parameters:
      temperature: 0.1
  - type: topic_control
    engine: composition-provider
    model: topic-model
    mode: text
    parameters:
      max_tokens: 40
"""
    )

    rails = LLMRails(config, llm=main if inject_main else None)

    expected_calls = [
        call(
            model_name="safety-model",
            provider_name="composition-provider",
            model_kwargs={"temperature": 0.1, "mode": "chat"},
        ),
        call(
            model_name="topic-model",
            provider_name="composition-provider",
            model_kwargs={"max_tokens": 40, "mode": "text"},
        ),
    ]
    if not inject_main:
        expected_calls.insert(
            0,
            call(
                model_name="configured-main",
                provider_name="composition-provider",
                model_kwargs={"temperature": 0.7, "mode": "chat"},
            ),
        )
    assert create_model.call_args_list == expected_calls
    assert provider.call_count == len(expected_calls)
    assert rails.llm is main
    params = rails.runtime.registered_action_params
    assert params["llm"] is main
    assert params["llms"] == {"content_safety": safety, "topic_control": topic}
    assert params["content_safety_llm"] is safety
    assert params["topic_control_llm"] is topic
    assert rails.content_safety_llm is safety
    assert rails.topic_control_llm is topic


def test_environment_key_overrides_configured_key_at_registered_provider(model_framework, monkeypatch):
    monkeypatch.setenv("COMPOSITION_TEST_API_KEY", "test-environment-key")
    provider = Mock(return_value=FakeLLMModel(responses=[]))
    model_framework.register_provider("composition-provider", provider)
    config = RailsConfig.from_content(
        yaml_content="""
models:
  - type: main
    engine: composition-provider
    model: configured-main
    api_key_env_var: COMPOSITION_TEST_API_KEY
    parameters:
      api_key: test-configured-key
      temperature: 0.25
"""
    )

    LLMRails(config)

    provider.assert_called_once_with(
        model="configured-main",
        api_key="test-environment-key",
        temperature=0.25,
    )
    assert config.models[0].parameters == {"api_key": "test-configured-key", "temperature": 0.25}


def test_model_caches_belong_to_each_runtime_with_shared_configuration(model_framework):
    provider = Mock(side_effect=lambda **kwargs: FakeLLMModel(responses=[]))
    model_framework.register_provider("composition-provider", provider)
    config = RailsConfig.from_content(
        yaml_content="""
models:
  - type: content_safety
    engine: composition-provider
    model: safety-model
    cache:
      enabled: true
      maxsize: 2
      stats:
        enabled: true
"""
    )

    first = LLMRails(config, llm=FakeLLMModel(responses=[]))
    second = LLMRails(config, llm=FakeLLMModel(responses=[]))
    first_cache = first.runtime.registered_action_params["model_caches"]["content_safety"]
    second_cache = second.runtime.registered_action_params["model_caches"]["content_safety"]

    first_cache.put("prompt", "safe")
    assert first_cache.get("prompt") == "safe"
    assert second_cache.get("prompt") is None
    assert first_cache is not second_cache
    assert first_cache.maxsize == second_cache.maxsize == 2
    assert provider.call_count == 2


def test_preexisting_specialized_alias_remains_in_all_action_views(model_framework):
    preset = FakeLLMModel(responses=[])

    class RailsWithPresetModel(LLMRails):
        content_safety_llm = preset

    provider = Mock(return_value=FakeLLMModel(responses=[]))
    model_framework.register_provider("composition-provider", provider)
    config = RailsConfig.from_content(
        yaml_content="""
models:
  - type: content_safety
    engine: composition-provider
    model: safety-model
"""
    )

    rails = RailsWithPresetModel(config, llm=FakeLLMModel(responses=[]))

    assert rails.content_safety_llm is preset
    params = rails.runtime.registered_action_params
    assert params["content_safety_llm"] is preset
    assert params["llms"] == {"content_safety": preset}
