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

"""Managed model resources release only their own retryable client transports."""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.frameworks import register_framework, set_default_framework
from nemoguardrails.llm.frameworks import registry as framework_registry
from nemoguardrails.llm.frameworks.default import DefaultFramework
from nemoguardrails.llm.models.resources import build_model_resources, resolve_model_resources
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.testing.fake_model import FakeLLMModel


@pytest.fixture
def framework(monkeypatch):
    monkeypatch.setattr(framework_registry, "_frameworks", {})
    monkeypatch.setattr(framework_registry, "_default_framework", "default")
    framework = DefaultFramework()
    register_framework("default", framework)
    set_default_framework("default")
    return framework


def configured_models(*, distinct_clients=False):
    return RailsConfig(
        models=[
            Model(type="main", engine="openai", model="main"),
            Model(
                type="content_safety",
                engine="openai",
                model="safety",
                parameters={"base_url": "https://safety.example/v1"} if distinct_clients else {},
            ),
        ]
    )


def build(config, **kwargs):
    return build_model_resources(resolve_model_resources(config.models), managed=True, **kwargs)


@pytest.mark.asyncio
async def test_runtime_pools_are_independent_and_share_within_a_runtime(framework):
    config = configured_models()
    first = build(config)
    second = build(config)
    first_client = first.main_model._client
    second_client = second.main_model._client
    assert first.specialized_models["content_safety"]._client is first_client
    assert second.specialized_models["content_safety"]._client is second_client
    assert first_client is not second_client
    assert first_client._http_client is None
    assert second_client._http_client is None
    assert framework._pool.clients == {}

    first_transport = first_client._client
    second_transport = second_client._client
    await first.aclose()
    assert first_transport.is_closed
    assert not second_transport.is_closed
    await second.aclose()
    assert second_transport.is_closed


@pytest.mark.asyncio
async def test_injected_main_and_registered_provider_products_are_borrowed(framework):
    main = FakeLLMModel(responses=[])
    main.aclose = AsyncMock()
    specialized = FakeLLMModel(responses=[])
    specialized.aclose = AsyncMock()
    provider = Mock(return_value=specialized)
    framework.register_provider("custom", provider)
    config = RailsConfig(models=[Model(type="content_safety", engine="custom", model="safety")])
    resources = build_model_resources(resolve_model_resources(config.models, main_model=main), managed=True)

    await resources.aclose()

    assert resources.main_model is main
    assert resources.specialized_models["content_safety"] is specialized
    main.aclose.assert_not_called()
    specialized.aclose.assert_not_called()
    assert framework._providers["custom"] is provider
    assert framework.create_model("another", "custom") is specialized


@pytest.mark.asyncio
async def test_custom_framework_retains_existing_ownership(monkeypatch):
    model = FakeLLMModel(responses=[])
    model.aclose = AsyncMock()

    class CustomFramework:
        def create_model(self, model_name, provider_name, model_kwargs=None):
            return model

        def register_provider(self, name, provider_cls):
            pass

        def get_provider_names(self):
            return ["custom"]

        async def reset(self):
            raise AssertionError("global reset must not run")

    monkeypatch.setattr(framework_registry, "_frameworks", {})
    monkeypatch.setattr(framework_registry, "_default_framework", "default")
    register_framework("custom", CustomFramework())
    set_default_framework("custom")
    resources = build(RailsConfig(models=[Model(type="main", engine="custom", model="main")]))
    assert resources.main_model is model
    await resources.aclose()
    model.aclose.assert_not_called()


def test_later_constructor_failure_allocates_no_transport(framework, monkeypatch):
    create_transport = Mock(side_effect=AssertionError("transport allocated during construction"))
    monkeypatch.setattr(OpenAICompatibleClient, "_new_http_client", create_transport)
    config = configured_models()
    config.models[1].engine = "unknown-provider"

    with pytest.raises(ValueError, match="No default base_url"):
        build(config)

    create_transport.assert_not_called()
    assert framework._pool.clients == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("close failed"), asyncio.CancelledError()])
async def test_unsuccessful_cleanup_is_retryable_without_blocking_other_clients(framework, monkeypatch, failure):
    resources = build(configured_models(distinct_clients=True))
    failed_client = resources.main_model._client
    successful_client = resources.specialized_models["content_safety"]._client
    failed_transport = failed_client._client
    successful_transport = successful_client._client
    real_close = failed_transport.aclose
    close = AsyncMock(side_effect=failure)
    monkeypatch.setattr(failed_transport, "aclose", close)

    with pytest.raises(type(failure)):
        await resources.aclose()

    assert failed_client._http_client is failed_transport
    assert successful_client._http_client is None
    assert successful_transport.is_closed
    close.side_effect = real_close
    await resources.aclose()
    assert failed_transport.is_closed
    assert failed_client._http_client is None
    assert close.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("transport close failed"), asyncio.CancelledError()])
async def test_restart_surfaces_incomplete_httpx_transport_cleanup(framework, monkeypatch, failure):
    resources = build(configured_models())
    adapter = resources.main_model._client
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    close_transport = AsyncMock(side_effect=failure)
    monkeypatch.setattr(transport, "aclose", close_transport)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(adapter, "_new_http_client", lambda: client)
    assert adapter._client is client

    with pytest.raises(type(failure)) as caught:
        await resources.aclose()
    assert caught.value is failure
    assert client.is_closed

    with pytest.raises(RuntimeError, match="transport cleanup is incomplete"):
        await resources.start()
    with pytest.raises(RuntimeError, match="transport cleanup is incomplete"):
        await resources.aclose()
    close_transport.assert_awaited_once()
    assert adapter._http_client is client


@pytest.mark.asyncio
async def test_simultaneous_cleanup_closes_transport_once(framework, monkeypatch):
    resources = build(configured_models())
    transport = resources.main_model._client._client
    close = AsyncMock(wraps=transport.aclose)
    monkeypatch.setattr(transport, "aclose", close)

    await asyncio.gather(resources.aclose(), resources.aclose())

    close.assert_awaited_once()
    assert transport.is_closed


@pytest.mark.asyncio
async def test_lazy_adapter_does_not_close_borrowed_http_client():
    transport = httpx.AsyncClient()
    client = OpenAICompatibleClient(base_url="https://model.example/v1", http_client=transport, lazy_client=True)
    try:
        await client.close()
        assert not transport.is_closed
        assert client._client is transport
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_llmrails_shutdown_leaves_the_framework_client_available(framework):
    config = configured_models()
    first = Guardrails(config, use_iorails=False)
    second = Guardrails(config, use_iorails=False)
    try:
        await first.startup()
        await second.startup()
        client = framework.create_model("main", "openai")._client
        assert first.llm._client is second.llm._client is client

        await first.shutdown()
        assert not client.is_closed()
        await second.shutdown()
        assert not client.is_closed()
    finally:
        await framework.aclose()
    assert client.is_closed()


@pytest.mark.asyncio
async def test_client_defaults_are_scoped_and_do_not_reach_custom_factories(framework):
    custom = Mock(return_value=FakeLLMModel(responses=[]))
    framework.register_provider("custom", custom)
    config = configured_models()
    config.models.append(Model(type="custom_check", engine="custom", model="custom"))
    config.models[1].parameters = {"timeout": 42, "connect_timeout": 7, "max_retries": 4}
    resources = build(config, client_defaults={"timeout": 30, "connect_timeout": 5, "max_retries": 2})
    main = resources.main_model._client
    specialized = resources.specialized_models["content_safety"]._client
    assert main._timeout.read == 30
    assert main._timeout.connect == 5
    assert main._max_retries == 2
    assert specialized._timeout.read == 42
    assert specialized._timeout.connect == 7
    assert specialized._max_retries == 4
    custom.assert_called_once_with(model="custom")
    assert framework._pool._client_defaults == {}
    await resources.aclose()


@pytest.mark.asyncio
async def test_framework_subclass_keeps_custom_construction_and_ownership(monkeypatch):
    model = FakeLLMModel(responses=[])

    model.aclose = AsyncMock()

    class CustomFramework(DefaultFramework):
        def create_model(self, model_name, provider_name, model_kwargs=None):
            return model

    monkeypatch.setattr(framework_registry, "_frameworks", {})
    monkeypatch.setattr(framework_registry, "_default_framework", "default")
    register_framework("default", CustomFramework())
    resources = build(configured_models())
    assert resources.main_model is model
    assert resources.specialized_models["content_safety"] is model
    await resources.aclose()
    model.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_start_retries_interrupted_cleanup_before_reusing_resources(framework, monkeypatch):
    resources = build(configured_models())
    client = resources.main_model._client
    transport = client._client
    close = AsyncMock(side_effect=RuntimeError("close failed"))
    original_close = transport.aclose
    monkeypatch.setattr(transport, "aclose", close)
    with pytest.raises(RuntimeError, match="close failed"):
        await resources.aclose()
    with pytest.raises(RuntimeError, match="close failed"):
        await resources.start()
    assert client._http_client is transport

    close.side_effect = original_close
    await resources.start()
    assert transport.is_closed
    assert client._http_client is None
    assert close.await_count == 3
