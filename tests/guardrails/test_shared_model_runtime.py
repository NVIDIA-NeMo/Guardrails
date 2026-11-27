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

"""Public IORails paths consume canonical resources without legacy model engines."""

from unittest.mock import Mock

import pytest

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails.model_runtime import ModelRuntime
from nemoguardrails.llm.frameworks import registry
from nemoguardrails.llm.frameworks.default import DefaultFramework
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction


@pytest.fixture
def selected_provider(monkeypatch):
    framework = DefaultFramework()
    created = {}

    def create(model, **kwargs):
        responses = ['{"User Safety": "safe"}'] if model == "safety" else ["Hello", "Again"]
        instance = FakeLLMModel(responses=responses)
        created[model] = instance
        return instance

    factory = Mock(side_effect=create)
    framework.register_provider("custom", factory)
    monkeypatch.setattr(registry, "_frameworks", {"default": framework})
    monkeypatch.setattr(registry, "_default_framework", "default")
    return factory, created


@pytest.fixture(autouse=True)
def prohibit_legacy_model_construction(monkeypatch):
    from nemoguardrails.guardrails.model_engine import ModelEngine

    def fail(*args, **kwargs):
        pytest.fail("The public IORails path constructed a legacy ModelEngine")

    monkeypatch.setattr(ModelEngine, "__init__", fail)


@pytest.mark.asyncio
async def test_injected_main_without_configured_main_supports_public_generation_and_restart():
    model = FakeLLMModel(responses=["Hello", "Again"])
    rails = Guardrails(RailsConfig(models=[]), llm=model, require_iorails=True)
    assert rails.use_iorails_engine
    assert rails.llm is model
    try:
        assert (await rails.generate_async(messages=[{"role": "user", "content": "Hi"}]))["content"] == "Hello"
        await rails.shutdown()
        assert (await rails.generate_async(messages=[{"role": "user", "content": "Hi"}]))["content"] == "Again"
    finally:
        await rails.shutdown()


@pytest.mark.asyncio
async def test_injected_main_streams_through_public_api():
    model = FakeLLMModel(responses=["Hello world"])
    rails = Guardrails(RailsConfig(models=[]), llm=model, require_iorails=True)
    try:
        chunks = [chunk async for chunk in rails.stream_async(messages=[{"role": "user", "content": "Hi"}])]
        assert "Hello world" in "".join(chunk for chunk in chunks if isinstance(chunk, str))
        assert model.inference_count == 1
    finally:
        await rails.shutdown()


@pytest.mark.asyncio
async def test_injected_main_satisfies_main_backed_rail_admission():
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"input": {"flows": ["self check input"]}},
            "prompts": [
                {
                    "task": "self_check_input",
                    "content": "Is this unsafe? {{ user_input }}",
                    "output_parser": "is_content_safe",
                }
            ],
        }
    )
    model = FakeLLMModel(responses=["no", "Hello"])
    rails = Guardrails(config, llm=model, require_iorails=True)
    try:
        result = await rails.generate_async(messages=[{"role": "user", "content": "Hi"}])
        assert result["content"] == "Hello"
        assert model.inference_count == 2
    finally:
        await rails.shutdown()


@pytest.mark.parametrize("use_iorails", [False, True], ids=["llmrails", "iorails"])
@pytest.mark.asyncio
async def test_first_configured_main_and_safety_models_run_with_rail_cache(selected_provider, use_iorails):
    _, models = selected_provider
    config = RailsConfig.from_content(
        config={
            "models": [
                {"type": "main", "engine": "custom", "model": "main"},
                {
                    "type": "content_safety",
                    "engine": "custom",
                    "model": "safety",
                    "cache": {"enabled": True, "maxsize": 4},
                },
                {"type": "main", "engine": "custom", "model": "later-main"},
                {"type": "content_safety", "engine": "custom", "model": "later-safety"},
            ],
            "rails": {"input": {"flows": ["content safety check input $model=content_safety"]}},
            "prompts": [
                {
                    "task": "content_safety_check_input $model=content_safety",
                    "content": "Check {{ user_input }}",
                    "output_parser": "nemoguard_parse_prompt_safety",
                }
            ],
        }
    )
    rails = Guardrails(config, use_iorails=use_iorails, require_iorails=use_iorails)
    try:
        assert rails.use_iorails_engine is use_iorails
        for expected in ("Hello", "Again"):
            result = await rails.generate_async(messages=[{"role": "user", "content": "Hi"}])
            assert result["content"] == expected
        assert models["safety"].inference_count == 1
        assert models["main"].inference_count == 2
        assert models["later-safety"].inference_count == 0
        assert rails.llm is models["main"]
    finally:
        await rails.shutdown()


def test_injected_main_skips_invalid_configured_transport(selected_provider):
    factory, _ = selected_provider
    config = RailsConfig.from_content(
        config={
            "models": [{"type": "main", "engine": "custom", "model": "ignored", "parameters": {"max_attempts": -1}}]
        }
    )
    model = FakeLLMModel(responses=[])
    rails = Guardrails(config, llm=model, require_iorails=True)
    assert rails.llm is model
    factory.assert_not_called()


def test_legacy_transport_aliases_do_not_mutate_configuration():
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "test-model",
                    "parameters": {"base_url": "http://model.test", "max_attempts": 1, "timeout_connect": 2},
                }
            ]
        }
    )
    original = dict(config.models[0].parameters)
    runtime = ModelRuntime(config.models)
    client = runtime.main_model._client
    assert client._max_retries == 0
    assert runtime.main_model.provider_url == "http://model.test/v1"
    assert config.models[0].parameters == original


@pytest.mark.asyncio
async def test_injected_tool_response_is_validated_and_preserved():
    call = ToolCall(id="call-1", type="function", function=ToolCallFunction(name="lookup", arguments={"term": "hello"}))
    model = FakeLLMModel(llm_responses=[LLMResponse(content="", tool_calls=[call], finish_reason="tool_calls")])
    rails = Guardrails(RailsConfig(models=[]), llm=model, require_iorails=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
            },
        }
    ]
    try:
        result = await rails.generate_async(
            messages=[{"role": "user", "content": "Search"}], options={"llm_params": {"tools": tools}}
        )
        assert result.tool_calls[0]["function"]["name"] == "lookup"
    finally:
        await rails.shutdown()


@pytest.mark.asyncio
async def test_legacy_transport_parameters_stay_out_of_request_body(httpx_mock):
    import json

    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "test-model",
                    "parameters": {
                        "base_url": "https://model.test",
                        "api_key": "test-key",
                        "max_attempts": 1,
                        "timeout_connect": 2,
                        "default_headers": {"X-Tenant": 42},
                        "default_query": {"tag": ["a", "b"], "flag": True},
                        "temperature": 0.5,
                    },
                }
            ]
        }
    )
    httpx_mock.add_response(
        json={"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]}
    )
    rails = Guardrails(config, require_iorails=True)
    try:
        assert (await rails.generate_async(messages=[{"role": "user", "content": "Hi"}]))["content"] == "hello"
        request = httpx_mock.get_request()
        assert request.headers["x-tenant"] == "42"
        assert request.url.params.get_list("tag") == ["a", "b"]
        assert request.url.params["flag"] == "true"
        body = json.loads(request.content)
        assert body == {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.5}
    finally:
        await rails.shutdown()


@pytest.mark.parametrize("query", ["not-a-mapping", {"nested": {"key": "value"}}, {"nested": [{"key": "value"}]}])
def test_invalid_later_query_fails_before_provider_construction(selected_provider, query):
    factory, _ = selected_provider
    config = RailsConfig.from_content(
        config={
            "models": [
                {"type": "main", "engine": "custom", "model": "main"},
                {
                    "type": "content_safety",
                    "engine": "openai",
                    "model": "safety",
                    "parameters": {"default_query": query},
                },
            ]
        }
    )
    with pytest.raises(ValueError, match="default_query"):
        Guardrails(config, require_iorails=True)
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("engine", "error"),
    [("azure", "azure_endpoint"), ("unknown-compatible", "No default base_url")],
)
def test_invalid_selected_provider_fails_before_main_factory(selected_provider, engine, error):
    factory, _ = selected_provider
    config = RailsConfig.from_content(
        config={
            "models": [
                {"type": "main", "engine": "custom", "model": "main"},
                {"type": "content_safety", "engine": engine, "model": "safety"},
            ]
        }
    )
    with pytest.raises(ValueError, match=error):
        Guardrails(config, require_iorails=True)
    factory.assert_not_called()


def test_injected_main_survives_repeated_synchronous_generation():
    model = FakeLLMModel(responses=["Hello", "Again"])
    rails = Guardrails(RailsConfig(models=[]), llm=model, require_iorails=True)
    for expected in ("Hello", "Again"):
        assert rails.generate(messages=[{"role": "user", "content": "Hi"}])["content"] == expected
    assert rails.llm is model
    assert model.inference_count == 2


def test_injected_main_satisfies_synchronous_main_backed_check():
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"input": {"flows": ["self check input"]}},
            "prompts": [
                {
                    "task": "self_check_input",
                    "content": "Is this unsafe? {{ user_input }}",
                    "output_parser": "is_content_safe",
                }
            ],
        }
    )
    model = FakeLLMModel(responses=["no", "no"])
    rails = Guardrails(config, llm=model, require_iorails=True)
    for _ in range(2):
        result = rails.check(messages=[{"role": "user", "content": "Hi"}])
        assert result.status.value == "passed"
    assert model.inference_count == 2


def test_testchat_can_force_public_iorails_with_injected_completions(monkeypatch):
    from nemoguardrails.testing import chat_harness

    def force_iorails(config, **kwargs):
        return Guardrails(config, require_iorails=True, **kwargs)

    monkeypatch.setattr(chat_harness, "LLMRails", force_iorails)
    chat = chat_harness.TestChat(RailsConfig(models=[]), llm_completions=["Hello", "Again"])
    chat.user("Hi")
    chat.bot("Hello")
    chat.user("Hi again")
    chat.bot("Again")
    assert chat.app.use_iorails_engine
    assert chat.app.llm is chat.llm


@pytest.mark.parametrize("provider", ["custom", "openai"])
def test_registered_provider_preserves_constructor_parameters(monkeypatch, provider):
    framework = DefaultFramework()
    factory = Mock(return_value=FakeLLMModel(responses=[]))
    framework.register_provider(provider, factory)
    monkeypatch.setattr(registry, "_frameworks", {"default": framework})
    monkeypatch.setattr(registry, "_default_framework", "default")
    parameters = {
        "base_url": "https://custom.test/rpc",
        "api_key": "test-only-key",
        "timeout_connect": 7,
        "connect_timeout": 11,
        "max_attempts": 4,
        "max_retries": 5,
        "temperature": 0.25,
        "default_headers": {"X-Numeric": 42},
        "default_query": {"provider_options": {"version": "custom"}},
    }
    config = RailsConfig.from_content(
        config={"models": [{"type": "main", "engine": provider, "model": "custom-model", "parameters": parameters}]}
    )
    runtime = ModelRuntime(config.models)
    factory.assert_called_once_with(model="custom-model", **parameters)
    assert runtime.main_model is factory.return_value
    assert config.models[0].parameters == parameters


def test_custom_framework_preserves_constructor_parameters(monkeypatch):
    factory = Mock(return_value=FakeLLMModel(responses=[]))

    class CustomFramework(DefaultFramework):
        def create_model(self, model_name, provider_name, model_kwargs=None):
            return factory(model=model_name, provider=provider_name, **(model_kwargs or {}))

    framework = CustomFramework()
    monkeypatch.setattr(registry, "_frameworks", {"custom-framework": framework})
    monkeypatch.setattr(registry, "_default_framework", "custom-framework")
    parameters = {
        "base_url": "https://custom.test/rpc",
        "api_key": "test-only-key",
        "timeout_connect": 7,
        "temperature": 0.25,
    }
    config = RailsConfig.from_content(
        config={"models": [{"type": "main", "engine": "openai", "model": "custom-model", "parameters": parameters}]}
    )
    runtime = ModelRuntime(config.models)
    factory.assert_called_once_with(model="custom-model", provider="openai", **parameters, mode="chat")
    assert runtime.main_model is factory.return_value
    assert config.models[0].parameters == parameters
