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

"""Model composition validates configuration before constructing resources."""

from unittest.mock import Mock, call

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.exceptions import InvalidModelConfigurationError
from nemoguardrails.llm.models import resources
from nemoguardrails.rails.llm.config import Model, ModelCacheConfig
from nemoguardrails.testing.fake_model import FakeLLMModel


@pytest.fixture
def model_factory(monkeypatch):
    factory = Mock(side_effect=lambda **kwargs: FakeLLMModel(responses=[]))
    monkeypatch.setattr(resources, "init_llm_model", factory)
    return factory


def test_injected_main_skips_overridden_config_validation_and_construction(model_factory):
    config = RailsConfig(models=[Model(type="main", engine="unused", model="configured-main")])
    config.models[0].model = None
    injected = FakeLLMModel(responses=[])

    plan = resources.resolve_model_resources(config.models, main_model=injected)
    built = resources.build_model_resources(plan)

    assert plan.main_config is None
    assert plan.main_model is injected
    assert built.main_model is injected
    assert built.specialized_models == {}
    model_factory.assert_not_called()


def test_specialized_models_work_without_a_main_model(model_factory):
    config = RailsConfig(models=[Model(type="content_safety", engine="fake", model="safety")])

    plan = resources.resolve_model_resources(config.models)
    built = resources.build_model_resources(plan)

    assert plan.main_config is None
    assert plan.main_model is None
    assert built.main_model is None
    assert list(built.specialized_models) == ["content_safety"]
    assert isinstance(built.specialized_models["content_safety"], FakeLLMModel)
    assert built.caches == {}
    model_factory.assert_called_once_with(model_name="safety", provider_name="fake", mode="chat", kwargs={})


def test_builder_preserves_configured_modes_and_specialized_mapping(model_factory):
    config = RailsConfig(
        models=[
            Model(type="topic_control", engine="fake", model="topic", mode="text", parameters={"max_tokens": 40}),
            Model(type="main", engine="fake", model="main", mode="text"),
            Model(type="content_safety", engine="fake", model="safety", parameters={"temperature": 0.1}),
        ]
    )
    main, topic, safety = [FakeLLMModel(responses=[]) for _ in range(3)]
    model_factory.side_effect = [main, topic, safety]

    plan = resources.resolve_model_resources(config.models)
    model_factory.assert_not_called()
    built = resources.build_model_resources(plan)

    assert built.main_model is main
    assert built.specialized_models == {"topic_control": topic, "content_safety": safety}
    assert model_factory.call_args_list == [
        call(model_name="main", provider_name="fake", mode="text", kwargs={}),
        call(model_name="topic", provider_name="fake", mode="text", kwargs={"max_tokens": 40}),
        call(model_name="safety", provider_name="fake", mode="chat", kwargs={"temperature": 0.1}),
    ]


@pytest.mark.parametrize("model_name", [None, "", "   "])
def test_invalid_later_model_fails_before_any_model_is_constructed(model_factory, model_name):
    config = RailsConfig(
        models=[
            Model(type="main", engine="fake", model="main"),
            Model(type="content_safety", engine="fake", model="safety", parameters={"api_key": "test-private-key"}),
        ]
    )
    config.models[1].model = model_name

    with pytest.raises(InvalidModelConfigurationError) as error:
        resources.build_model_resources(resources.resolve_model_resources(config.models))

    model_factory.assert_not_called()
    assert "test-private-key" not in str(error.value)


@pytest.mark.parametrize("maxsize", [0, -1])
def test_invalid_later_cache_fails_before_any_model_is_constructed(model_factory, maxsize):
    config = RailsConfig(
        models=[
            Model(type="main", engine="fake", model="main"),
            Model(
                type="content_safety",
                engine="fake",
                model="safety",
                cache=ModelCacheConfig(enabled=True, maxsize=maxsize),
            ),
        ]
    )

    with pytest.raises(ValueError, match="Invalid cache maxsize"):
        resources.build_model_resources(resources.resolve_model_resources(config.models))

    model_factory.assert_not_called()


def test_exclusions_keep_jailbreak_cache_without_constructing_its_model(model_factory):
    config = RailsConfig(
        models=[
            Model(type="main", engine="fake", model="main", cache=ModelCacheConfig(enabled=True, maxsize=0)),
            Model(type="embeddings", engine="fake", model="embedding", cache=ModelCacheConfig(enabled=True, maxsize=0)),
            Model(
                type="jailbreak_detection",
                engine="fake",
                model="jailbreak",
                cache=ModelCacheConfig(enabled=True, maxsize=2),
            ),
            Model(
                type="content_safety", engine="fake", model="safety", cache=ModelCacheConfig(enabled=False, maxsize=0)
            ),
        ]
    )
    config.models[1].model = None
    config.models[2].model = None

    plan = resources.resolve_model_resources(config.models)
    built = resources.build_model_resources(plan)

    assert model_factory.call_args_list == [
        call(model_name="main", provider_name="fake", mode="chat", kwargs={}),
        call(model_name="safety", provider_name="fake", mode="chat", kwargs={}),
    ]
    assert list(built.specialized_models) == ["content_safety"]
    assert list(built.caches) == ["jailbreak_detection"]
    cache = built.caches["jailbreak_detection"]
    cache.put("prompt", "safe")
    assert cache.get("prompt") == "safe"
    assert cache.maxsize == 2


@pytest.mark.parametrize("configured_key", [None, "test-configured-key"])
def test_environment_disappearing_after_config_parse_preserves_parameters(monkeypatch, model_factory, configured_key):
    monkeypatch.setenv("RESOURCE_TEST_API_KEY", "test-environment-key")
    parameters = {"temperature": 0.25}
    if configured_key is not None:
        parameters["api_key"] = configured_key
    config = RailsConfig(
        models=[
            Model(
                type="main", engine="fake", model="main", api_key_env_var="RESOURCE_TEST_API_KEY", parameters=parameters
            )
        ]
    )
    monkeypatch.delenv("RESOURCE_TEST_API_KEY")

    resources.build_model_resources(resources.resolve_model_resources(config.models))

    assert model_factory.call_args.kwargs["kwargs"] == parameters
    assert config.models[0].parameters == parameters
    assert model_factory.call_args.kwargs["kwargs"] is not config.models[0].parameters


def test_factory_parameter_changes_do_not_mutate_configuration(model_factory):
    config = RailsConfig(models=[Model(type="main", engine="fake", model="main", parameters={"temperature": 0.25})])

    def create_model(**kwargs):
        kwargs["kwargs"]["temperature"] = 0.9
        kwargs["kwargs"]["provider_default"] = True
        return FakeLLMModel(responses=[])

    model_factory.side_effect = create_model
    resources.build_model_resources(resources.resolve_model_resources(config.models))

    assert config.models[0].parameters == {"temperature": 0.25}


def test_duplicate_specialized_types_preserve_first_model_and_last_enabled_cache(model_factory):
    config = RailsConfig(
        models=[
            Model(type="content_safety", engine="fake", model="first", cache=ModelCacheConfig(enabled=True, maxsize=2)),
            Model(
                type="content_safety", engine="fake", model="second", cache=ModelCacheConfig(enabled=True, maxsize=3)
            ),
            Model(
                type="content_safety", engine="fake", model="third", cache=ModelCacheConfig(enabled=False, maxsize=4)
            ),
        ]
    )
    first = FakeLLMModel(responses=[])
    model_factory.side_effect = lambda **kwargs: (
        first if kwargs["model_name"] == "first" else FakeLLMModel(responses=[])
    )

    built = resources.build_model_resources(resources.resolve_model_resources(config.models))

    assert built.specialized_models == {"content_safety": first}
    assert built.caches["content_safety"].maxsize == 3
    assert model_factory.call_args_list == [
        call(model_name=name, provider_name="fake", mode="chat", kwargs={}) for name in ("first", "second", "third")
    ]
