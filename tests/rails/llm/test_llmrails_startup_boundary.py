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

from types import ModuleType, SimpleNamespace

from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.rails.llm import llmrails
from nemoguardrails.rails.llm.config import RailsConfig, TracingConfig
from nemoguardrails.rails.llm.startup.config_preparation import PreparedLLMRailsConfig
from nemoguardrails.rails.llm.startup.embedding_search import DEFAULT_EMBEDDING_ENGINE, DEFAULT_EMBEDDING_MODEL
from nemoguardrails.rails.llm.startup.tracing import create_startup_tracing_adapters


class RuntimeWithRegistries:
    def __init__(self):
        self.llm_task_manager = object()
        self.registered_action_params = {}

    def register_action_param(self, name, value):
        self.registered_action_params[name] = value


def _prepared_config(config: RailsConfig) -> PreparedLLMRailsConfig:
    setattr(config, "_prepared_for_startup_test", True)
    return PreparedLLMRailsConfig(
        config=config,
        default_embedding_model="prepared-model",
        default_embedding_engine="prepared-engine",
        default_embedding_params={"prepared": True},
    )


def test_constructor_startup_order_from_config_preparation_through_kb(monkeypatch):
    sequence = []

    def prepare_llmrails_config(**kwargs):
        sequence.append("prepare config")
        assert kwargs["default_embedding_model"] == DEFAULT_EMBEDDING_MODEL
        assert kwargs["default_embedding_engine"] == DEFAULT_EMBEDDING_ENGINE
        assert kwargs["default_embedding_params"] == {}
        return _prepared_config(kwargs["config"])

    def load_config_py_modules(config):
        sequence.append("load config.py modules")
        assert getattr(config, "_prepared_for_startup_test") is True
        return [ModuleType("config")]

    def runtime_for_colang_version(config, verbose):
        sequence.append("runtime creation")
        assert getattr(config, "_prepared_for_startup_test") is True
        assert verbose is True
        return RuntimeWithRegistries()

    def run_config_py_init_hooks(rails, config_modules):
        sequence.append("config.py init hooks")
        assert len(config_modules) == 1
        assert getattr(rails.config, "_prepared_for_startup_test") is True
        assert isinstance(rails.runtime, RuntimeWithRegistries)

    def create_log_adapters(tracing_config):
        sequence.append("tracing adapter creation")
        assert tracing_config.enabled is True
        return ["adapter"]

    def validate_config(config):
        sequence.append("validation")

    def init_llms(rails):
        sequence.append("LLM/model-cache initialization")

    def register_llm_generation_actions(rails, verbose):
        sequence.append("generation-action registration")
        assert verbose is True
        rails.llm_generation_actions = object()

    def init_knowledge_base(rails):
        sequence.append("KB initialization")
        rails.kb = None

    monkeypatch.setattr(llmrails, "prepare_llmrails_config", prepare_llmrails_config)
    monkeypatch.setattr(llmrails, "load_config_py_modules", load_config_py_modules)
    monkeypatch.setattr(llmrails, "runtime_for_colang_version", runtime_for_colang_version)
    monkeypatch.setattr(llmrails, "run_config_py_init_hooks", run_config_py_init_hooks)
    monkeypatch.setattr("nemoguardrails.tracing.create_log_adapters", create_log_adapters)
    monkeypatch.setattr(llmrails, "validate_llmrails_config", validate_config)
    monkeypatch.setattr(llmrails.LLMRails, "_init_llms", init_llms)
    monkeypatch.setattr(llmrails, "register_llm_generation_actions", register_llm_generation_actions)
    monkeypatch.setattr(llmrails, "init_knowledge_base", init_knowledge_base)

    config = RailsConfig(models=[], tracing=TracingConfig(enabled=True))

    rails = llmrails.LLMRails(config, verbose=True)

    assert sequence == [
        "prepare config",
        "load config.py modules",
        "runtime creation",
        "config.py init hooks",
        "tracing adapter creation",
        "validation",
        "LLM/model-cache initialization",
        "generation-action registration",
        "KB initialization",
    ]
    assert rails._log_adapters == ["adapter"]
    assert rails.embedding_search.default_model == "prepared-model"
    assert rails.embedding_search.default_engine == "prepared-engine"
    assert rails.embedding_search.default_params == {"prepared": True}
    assert rails.explain_info is None


def test_config_py_hook_sees_prepared_state_and_provider_registration_reaches_later_startup(monkeypatch):
    provider_seen_by_generation_actions = None
    provider_seen_by_kb = None

    class CustomProvider(EmbeddingsIndex):
        pass

    config_module = ModuleType("config")

    def init(app):
        assert getattr(app.config, "_prepared_for_startup_test") is True
        assert isinstance(app.runtime, RuntimeWithRegistries)
        assert app.embedding_search.default_model == "prepared-model"
        assert app.embedding_search.default_engine == "prepared-engine"
        assert app.embedding_search.default_params == {"prepared": True}
        assert app.embedding_search.providers == {}
        app.register_embedding_search_provider("custom", CustomProvider)

    setattr(config_module, "init", init)

    def register_llm_generation_actions(rails, verbose):
        nonlocal provider_seen_by_generation_actions
        del verbose
        provider_seen_by_generation_actions = rails.embedding_search.providers["custom"]
        rails.llm_generation_actions = SimpleNamespace()

    def init_knowledge_base(rails):
        nonlocal provider_seen_by_kb
        provider_seen_by_kb = rails.embedding_search.providers["custom"]
        rails.kb = None

    monkeypatch.setattr(llmrails, "prepare_llmrails_config", lambda **kwargs: _prepared_config(kwargs["config"]))
    monkeypatch.setattr(llmrails, "load_config_py_modules", lambda config: [config_module])
    monkeypatch.setattr(llmrails, "runtime_for_colang_version", lambda config, verbose: RuntimeWithRegistries())
    monkeypatch.setattr(llmrails, "validate_llmrails_config", lambda config: None)
    monkeypatch.setattr(llmrails.LLMRails, "_init_llms", lambda rails: None)
    monkeypatch.setattr(llmrails, "register_llm_generation_actions", register_llm_generation_actions)
    monkeypatch.setattr(llmrails, "init_knowledge_base", init_knowledge_base)

    config = RailsConfig(models=[])
    object.__setattr__(config, "tracing", None)

    rails = llmrails.LLMRails(config)

    assert rails.embedding_search.providers["custom"] is CustomProvider
    assert provider_seen_by_generation_actions is CustomProvider
    assert provider_seen_by_kb is CustomProvider
    assert rails._log_adapters is None


def test_tracing_adapters_are_created_after_config_py_init_hooks(monkeypatch):
    events = []
    config_module = ModuleType("config")

    def init(app):
        app.tracing_marker = "hook-ran"
        events.append("hook")

    setattr(config_module, "init", init)

    def create_log_adapters(tracing_config):
        assert tracing_config.enabled is True
        assert events == ["hook"]
        events.append("tracing")
        return [{"tracing": tracing_config}]

    monkeypatch.setattr(llmrails, "prepare_llmrails_config", lambda **kwargs: _prepared_config(kwargs["config"]))
    monkeypatch.setattr(llmrails, "load_config_py_modules", lambda config: [config_module])
    monkeypatch.setattr(llmrails, "runtime_for_colang_version", lambda config, verbose: RuntimeWithRegistries())
    monkeypatch.setattr("nemoguardrails.tracing.create_log_adapters", create_log_adapters)
    monkeypatch.setattr(llmrails, "validate_llmrails_config", lambda config: None)
    monkeypatch.setattr(llmrails.LLMRails, "_init_llms", lambda rails: None)
    monkeypatch.setattr(
        llmrails,
        "register_llm_generation_actions",
        lambda rails, verbose: setattr(rails, "llm_generation_actions", object()),
    )
    monkeypatch.setattr(llmrails, "init_knowledge_base", lambda rails: setattr(rails, "kb", None))

    config = RailsConfig(models=[], tracing=TracingConfig(enabled=True))

    rails = llmrails.LLMRails(config)

    assert events == ["hook", "tracing"]
    assert getattr(rails, "tracing_marker") == "hook-ran"
    assert rails._log_adapters == [{"tracing": config.tracing}]


def test_startup_tracing_helper_returns_none_without_tracing_config(monkeypatch):
    calls = []

    def create_log_adapters(tracing_config):
        calls.append(tracing_config)
        return ["adapter"]

    monkeypatch.setattr("nemoguardrails.tracing.create_log_adapters", create_log_adapters)

    config = RailsConfig(models=[])
    object.__setattr__(config, "tracing", None)

    assert create_startup_tracing_adapters(config) is None
    assert calls == []


def test_startup_tracing_helper_creates_adapters_when_tracing_config_is_present(monkeypatch):
    calls = []

    def create_log_adapters(tracing_config):
        calls.append(tracing_config)
        return ["adapter"]

    monkeypatch.setattr("nemoguardrails.tracing.create_log_adapters", create_log_adapters)

    config = RailsConfig(models=[], tracing=TracingConfig(enabled=True))

    assert create_startup_tracing_adapters(config) == ["adapter"]
    assert calls == [config.tracing]
