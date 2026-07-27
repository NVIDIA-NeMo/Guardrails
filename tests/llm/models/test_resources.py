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

import logging
from unittest.mock import Mock

import pytest

from nemoguardrails.llm.models.initializer import ModelInitializationError
from nemoguardrails.llm.models.resources import (
    build_llm_model_resources,
    prepare_model_kwargs,
)
from nemoguardrails.rails.llm.config import CacheStatsConfig, Model, ModelCacheConfig, RailsConfig
from nemoguardrails.testing import FakeLLMModel


def _config(*models: Model) -> RailsConfig:
    return RailsConfig(models=list(models))


def test_builds_main_specialized_models_action_parameters_and_caches():
    main = FakeLLMModel(responses=[])
    content_safety = FakeLLMModel(responses=[])
    initializer = Mock(side_effect=[main, content_safety])
    config = _config(
        Model(type="main", engine="fake", model="main-model", mode="text"),
        Model(
            type="content_safety",
            engine="fake",
            model="safety-model",
            mode="text",
            cache=ModelCacheConfig(
                enabled=True,
                maxsize=17,
                stats=CacheStatsConfig(enabled=True, log_interval=30.0),
            ),
        ),
        Model(
            type="jailbreak_detection",
            engine="fake",
            model="jailbreak-model",
            cache=ModelCacheConfig(enabled=True, maxsize=23),
        ),
        Model(type="embeddings", engine="fake", model="embedding-model"),
    )

    resources = build_llm_model_resources(config.models, initializer=initializer)

    assert resources.main is main
    assert resources.by_type == {"content_safety": content_safety}
    assert set(resources.caches) == {"content_safety", "jailbreak_detection"}
    assert resources.caches["content_safety"].maxsize == 17
    assert resources.caches["content_safety"].track_stats is True
    assert resources.caches["content_safety"].stats_logging_interval == 30.0
    assert resources.caches["jailbreak_detection"].maxsize == 23
    assert initializer.call_count == 2
    assert initializer.call_args_list[0].kwargs == {
        "model_name": "main-model",
        "provider_name": "fake",
        "mode": "chat",
        "kwargs": {},
    }
    assert initializer.call_args_list[1].kwargs == {
        "model_name": "safety-model",
        "provider_name": "fake",
        "mode": "text",
        "kwargs": {},
    }
    assert resources.action_parameters() == {
        "llm": main,
        "llms": {"content_safety": content_safety},
        "content_safety_llm": content_safety,
        "model_caches": dict(resources.caches),
    }


def test_injected_main_takes_precedence_and_is_decorated(caplog):
    injected = FakeLLMModel(responses=[])
    decorated_main = FakeLLMModel(responses=[])
    specialized = FakeLLMModel(responses=[])
    initializer = Mock(return_value=specialized)
    decorator = Mock(side_effect=[decorated_main, specialized])
    config = _config(
        Model(type="main", engine="fake", model="configured-main"),
        Model(type="topic_control", engine="fake", model="topic-model", parameters={"temperature": 0.1}),
    )

    with caplog.at_level(logging.WARNING):
        resources = build_llm_model_resources(
            config.models,
            main=injected,
            initializer=initializer,
            decorator=decorator,
        )

    assert resources.main is decorated_main
    assert resources.by_type == {"topic_control": specialized}
    initializer.assert_called_once_with(
        model_name="topic-model",
        provider_name="fake",
        mode="chat",
        kwargs={"temperature": 0.1},
    )
    assert decorator.call_args_list[0].args == (injected, {})
    assert decorator.call_args_list[1].args == (specialized, {"temperature": 0.1})
    assert "Both an LLM was provided via constructor" in caplog.text


def test_prepare_model_kwargs_reads_environment_without_mutating_config(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "secret")
    model = Model(
        type="main",
        engine="fake",
        model="main-model",
        api_key_env_var="MODEL_API_KEY",
        parameters={"temperature": 0.2},
    )

    kwargs = prepare_model_kwargs(model)

    assert kwargs == {"temperature": 0.2, "api_key": "secret"}
    assert model.parameters == {"temperature": 0.2}


def test_prepare_model_kwargs_ignores_missing_environment_variable(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    model = Model(
        type="main",
        engine="fake",
        model="main-model",
        api_key_env_var="MODEL_API_KEY",
        parameters={"temperature": 0.2},
    )

    assert prepare_model_kwargs(model) == {"temperature": 0.2}


def test_invalid_cache_capacity_fails_before_returning_resources():
    config = _config(
        Model(
            type="content_safety",
            engine="fake",
            model="safety-model",
            cache=ModelCacheConfig(enabled=True, maxsize=0),
        )
    )

    with pytest.raises(ValueError, match="Invalid cache maxsize"):
        build_llm_model_resources(
            config.models,
            initializer=Mock(return_value=FakeLLMModel(responses=[])),
        )


def test_model_initialization_error_is_preserved(caplog):
    error = ModelInitializationError("provider failed")
    config = _config(Model(type="main", engine="fake", model="main-model"))

    with caplog.at_level(logging.ERROR), pytest.raises(ModelInitializationError) as exc_info:
        build_llm_model_resources(config.models, initializer=Mock(side_effect=error))

    assert exc_info.value is error
    assert "Failed to initialize model: provider failed" in caplog.text
