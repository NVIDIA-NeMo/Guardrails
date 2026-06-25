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

import warnings

import httpx
import pytest

from nemoguardrails.llm.clients.constants import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
)
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.models.openai_chat import OpenAIChatModel


def _make_client(**kwargs):
    return OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key="sk-test", **kwargs)


class TestTimeout:
    @pytest.mark.asyncio
    async def test_defaults(self):
        async with _make_client() as client:
            assert client._client.timeout.read == DEFAULT_TIMEOUT.read
            assert client._client.timeout.connect == DEFAULT_TIMEOUT.connect

    @pytest.mark.asyncio
    async def test_custom(self):
        async with _make_client(timeout=120.0, connect_timeout=10.0) as client:
            assert client._client.timeout.read == 120.0
            assert client._client.timeout.connect == 10.0

    @pytest.mark.asyncio
    async def test_http_client_timeout_inferred(self):
        custom = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0))
        try:
            client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client=custom)
            assert client._client is custom
        finally:
            await custom.aclose()


class TestConnectionPool:
    @pytest.mark.asyncio
    async def test_limits(self):
        async with _make_client() as client:
            pool = client._client._transport._pool
            assert pool._max_connections == DEFAULT_CONNECTION_LIMITS.max_connections
            assert pool._max_keepalive_connections == DEFAULT_CONNECTION_LIMITS.max_keepalive_connections


class TestMaxRetries:
    @pytest.mark.asyncio
    async def test_default(self):
        async with _make_client() as client:
            assert client._max_retries == DEFAULT_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_custom(self):
        async with _make_client(max_retries=5) as client:
            assert client._max_retries == 5


class TestCustomHeaders:
    @pytest.mark.asyncio
    async def test_merged_into_request(self):
        async with _make_client(custom_headers={"X-Custom": "value"}) as client:
            headers = client._build_headers()
            assert headers["X-Custom"] == "value"
            assert headers["Authorization"] == "Bearer sk-test"
            assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_override_defaults(self):
        async with OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            custom_headers={"Content-Type": "text/plain"},
        ) as client:
            headers = client._build_headers()
            assert headers["Content-Type"] == "text/plain"


class TestCustomQuery:
    @pytest.mark.asyncio
    async def test_stored(self):
        async with _make_client(custom_query={"api-version": "2024-02-01"}) as client:
            assert client._custom_query == {"api-version": "2024-02-01"}


class TestHttpClientInjection:
    @pytest.mark.asyncio
    async def test_uses_injected_client(self):
        custom = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        try:
            client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key="sk-test", http_client=custom)
            assert client._client is custom
        finally:
            await custom.aclose()

    @pytest.mark.asyncio
    async def test_close_does_not_close_injected_client(self):
        custom = httpx.AsyncClient()
        client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client=custom)
        await client.close()
        assert not custom.is_closed
        await custom.aclose()

    @pytest.mark.asyncio
    async def test_close_closes_owned_client(self):
        client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key="sk")
        owned = client._client
        await client.close()
        assert owned.is_closed

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="httpx.AsyncClient"):
            OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client="not a client")


class TestPoolKeyAcceptsUnhashableQueryValues:
    @pytest.mark.asyncio
    async def test_list_query_value_does_not_crash(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o",
                "openai",
                {"api_key": "sk", "default_query": {"tags": ["a", "b"]}},
            )
            assert model.model_name == "gpt-4o"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_nested_dict_query_value_does_not_crash(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o",
                "openai",
                {"api_key": "sk", "default_query": {"meta": {"region": "us"}}},
            )
            assert model.model_name == "gpt-4o"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_same_query_pools_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "default_query": {"tags": ["a", "b"]}})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "default_query": {"tags": ["a", "b"]}})
            assert m1._client is m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_query_different_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "default_query": {"tags": ["a", "b"]}})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "default_query": {"tags": ["a", "c"]}})
            assert m1._client is not m2._client
        finally:
            await fw.reset()


class TestPlaintextHttpWarning:
    def test_warns_on_http_with_api_key(self):
        with pytest.warns(UserWarning, match="plaintext HTTP"):
            client = OpenAICompatibleClient(base_url="http://api.example.com/v1", api_key="sk-test")
        assert client._api_key == "sk-test"

    def test_no_warning_on_https_with_api_key(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            OpenAICompatibleClient(base_url="https://api.example.com/v1", api_key="sk-test")

    def test_no_warning_on_http_without_api_key(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            OpenAICompatibleClient(base_url="http://api.example.com/v1")

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
            "http://my-server.local/v1",
            "http://nemo.local:11434/v1",
        ],
    )
    def test_no_warning_for_local_hosts(self, base_url):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            OpenAICompatibleClient(base_url=base_url, api_key="sk-test")


class TestDefaultFramework:
    @pytest.mark.asyncio
    async def test_creates_chat_model(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model("gpt-4o", "openai", {"api_key": "sk-test"})

            assert isinstance(model, OpenAIChatModel)
            assert model.model_name == "gpt-4o"
            assert model.provider_url == "https://api.openai.com/v1"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_creates_nim(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model("llama", "nim", {"api_key": "nvapi-test"})

            assert isinstance(model, OpenAIChatModel)
            assert model.provider_url == "https://integrate.api.nvidia.com/v1"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_creates_azure(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o-mini",
                "azure",
                {
                    "base_url": "https://my-resource.openai.azure.com/",
                    "azure_deployment": "my-deployment",
                    "api_version": "2024-02-15-preview",
                    "api_key": "test-azure-key",
                },
            )

            assert isinstance(model, OpenAIChatModel)
            assert model.provider_url == "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
            assert model._client._api_key is None
            assert model._client._custom_headers == {"api-key": "test-azure-key"}
            assert model._client._custom_query == {"api-version": "2024-02-15-preview"}
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_openai_alias(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o-mini",
                "azure_openai",
                {
                    "base_url": "https://my-resource.openai.azure.com",
                    "azure_deployment": "deploy-2",
                    "api_version": "2024-12-01-preview",
                    "api_key": "k",
                },
            )

            assert model.provider_url == "https://my-resource.openai.azure.com/openai/deployments/deploy-2"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_resolves_api_key_from_env(self, monkeypatch):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key-value")
        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o-mini",
                "azure",
                {
                    "base_url": "https://my-resource.openai.azure.com/",
                    "azure_deployment": "d",
                    "api_version": "2024-02-15-preview",
                },
            )
            assert model._client._custom_headers["api-key"] == "env-key-value"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_user_headers_and_query_preserved(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o-mini",
                "azure",
                {
                    "base_url": "https://my-resource.openai.azure.com/",
                    "azure_deployment": "d",
                    "api_version": "2024-02-15-preview",
                    "api_key": "k",
                    "default_headers": {"X-Custom": "value"},
                    "default_query": {"trace-id": "abc"},
                },
            )
            assert model._client._custom_headers == {"X-Custom": "value", "api-key": "k"}
            assert model._client._custom_query == {"trace-id": "abc", "api-version": "2024-02-15-preview"}
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_default_query_api_version_conflict_raises(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match=r"conflicting Azure API versions"):
                fw.create_model(
                    "gpt-4o-mini",
                    "azure",
                    {
                        "base_url": "https://my-resource.openai.azure.com/",
                        "azure_deployment": "d",
                        "api_version": "2024-02-15-preview",
                        "api_key": "k",
                        "default_query": {"api-version": "2023-05-15"},
                    },
                )
        finally:
            await fw.reset()

    @pytest.mark.parametrize("header_name", ["api-key", "Api-Key", "API-KEY"])
    @pytest.mark.asyncio
    async def test_azure_default_headers_api_key_satisfies_auth(self, monkeypatch, header_name):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o-mini",
                "azure",
                {
                    "base_url": "https://my-resource.openai.azure.com/",
                    "azure_deployment": "d",
                    "api_version": "2024-02-15-preview",
                    "default_headers": {header_name: "header-key"},
                },
            )
            assert model._client._custom_headers == {header_name: "header-key"}
        finally:
            await fw.reset()

    @pytest.mark.parametrize("header_name", ["api-key", "Api-Key", "API-KEY"])
    @pytest.mark.asyncio
    async def test_azure_api_key_and_default_headers_api_key_conflict_raises(self, header_name):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match="conflicting Azure API keys"):
                fw.create_model(
                    "gpt-4o-mini",
                    "azure",
                    {
                        "base_url": "https://my-resource.openai.azure.com/",
                        "azure_deployment": "d",
                        "api_version": "2024-02-15-preview",
                        "api_key": "param-key",
                        "default_headers": {header_name: "header-key"},
                    },
                )
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_missing_endpoint_raises(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match=r"requires parameters\.azure_endpoint"):
                fw.create_model(
                    "gpt",
                    "azure",
                    {"azure_deployment": "d", "api_version": "v", "api_key": "k"},
                )
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_endpoint_alias(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model(
                "gpt-4o-mini",
                "azure",
                {
                    "azure_endpoint": "https://my-resource.openai.azure.com/",
                    "azure_deployment": "my-deployment",
                    "api_version": "2024-02-15-preview",
                    "api_key": "k",
                },
            )
            assert model.provider_url == "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
            assert model._client._custom_headers == {"api-key": "k"}
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_endpoint_and_base_url_both_set_raises(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match=r"either parameters\.azure_endpoint or parameters\.base_url"):
                fw.create_model(
                    "gpt",
                    "azure",
                    {
                        "azure_endpoint": "https://x.openai.azure.com/",
                        "base_url": "https://x.openai.azure.com/",
                        "azure_deployment": "d",
                        "api_version": "v",
                        "api_key": "k",
                    },
                )
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_missing_deployment_raises(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match=r"requires parameters\.azure_deployment"):
                fw.create_model(
                    "gpt",
                    "azure",
                    {"base_url": "https://x", "api_version": "v", "api_key": "k"},
                )
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_missing_api_version_raises(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match=r"requires parameters\.api_version"):
                fw.create_model(
                    "gpt",
                    "azure",
                    {"base_url": "https://x", "azure_deployment": "d", "api_key": "k"},
                )
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_missing_api_key_raises(self, monkeypatch):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match="requires an API key"):
                fw.create_model(
                    "gpt",
                    "azure",
                    {"base_url": "https://x", "azure_deployment": "d", "api_version": "v"},
                )
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_azure_listed_in_provider_names(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        names = fw.get_provider_names()
        assert "azure" in names
        assert "azure_openai" in names

    @pytest.mark.asyncio
    async def test_pools_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-test"})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk-test"})

            assert m1._client is m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_keys_different_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-one"})
            m2 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-two"})

            assert m1._client is not m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_timeout_different_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "timeout": 30.0})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "timeout": 5.0})

            assert m1._client is not m2._client
            assert m1._client._client.timeout.read == 30.0
            assert m2._client._client.timeout.read == 5.0
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_headers_different_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "default_headers": {"X-A": "1"}})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "default_headers": {"X-B": "2"}})

            assert m1._client is not m2._client
            assert m1._client._custom_headers == {"X-A": "1"}
            assert m2._client._custom_headers == {"X-B": "2"}
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_same_full_config_pooled(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            cfg = {"api_key": "sk", "timeout": 30.0, "default_headers": {"X-A": "1"}}
            m1 = fw.create_model("gpt-4o", "openai", cfg.copy())
            m2 = fw.create_model("gpt-4o-mini", "openai", cfg.copy())

            assert m1._client is m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_reset_closes_all_pooled_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-a"})
        m2 = fw.create_model("llama", "nim", {"api_key": "nv-a"})
        m3 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk-b"})

        clients = [m1._client._client, m2._client._client, m3._client._client]
        assert all(not c.is_closed for c in clients)

        await fw.reset()

        assert all(c.is_closed for c in clients)
        assert fw._clients == {}

    @pytest.mark.asyncio
    async def test_reset_clears_registered_providers(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        fw.register_provider("custom", lambda **kw: object())
        assert "custom" in fw._providers

        await fw.reset()

        assert fw._providers == {}

    @pytest.mark.asyncio
    async def test_reset_allows_recreation_with_fresh_clients(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
        first_client = m1._client._client
        await fw.reset()

        m2 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
        assert m2._client._client is not first_client
        assert not m2._client._client.is_closed

    @pytest.mark.asyncio
    async def test_reset_does_not_close_injected_clients(self):
        import httpx

        from nemoguardrails.llm.frameworks.default import DefaultFramework

        injected = httpx.AsyncClient()
        client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client=injected)
        fw = DefaultFramework()
        fw._clients[("injected",)] = client

        await fw.reset()

        assert not injected.is_closed
        await injected.aclose()

    @pytest.mark.asyncio
    async def test_aclose_closes_pools_only_keeps_providers(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-a"})
        client = m1._client._client
        fw.register_provider("custom", lambda **kw: object())

        await fw.aclose()

        assert client.is_closed
        assert fw._clients == {}
        assert "custom" in fw._providers

    @pytest.mark.asyncio
    async def test_aclose_can_be_called_repeatedly(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        fw.create_model("gpt-4o", "openai", {"api_key": "sk-a"})
        await fw.aclose()
        await fw.aclose()

    @pytest.mark.asyncio
    async def test_aclose_then_create_model_rebuilds_pool(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
            first_client = m1._client._client
            await fw.aclose()

            m2 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
            assert m2._client._client is not first_client
            assert not m2._client._client.is_closed
        finally:
            await fw.aclose()

    def test_clear_providers_drops_registry_only(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        fw.register_provider("custom_a", lambda **kw: object())
        fw.register_provider("custom_b", lambda **kw: object())
        assert "custom_a" in fw._providers
        assert "custom_b" in fw._providers

        fw.clear_providers()

        assert fw._providers == {}

    @pytest.mark.asyncio
    async def test_clear_providers_does_not_touch_pool(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
            client = m1._client._client
            fw.register_provider("custom", lambda **kw: object())

            fw.clear_providers()

            assert not client.is_closed
            assert fw._clients != {}
        finally:
            await fw.aclose()

    @pytest.mark.asyncio
    async def test_reset_calls_both_aclose_and_clear_providers(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
        client = m1._client._client
        fw.register_provider("custom", lambda **kw: object())

        await fw.reset()

        assert client.is_closed
        assert fw._clients == {}
        assert fw._providers == {}

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match="No default base_url"):
                fw.create_model("model", "unknown_provider", {})
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_unknown_provider_error_lists_both_fix_paths(self):
        """Error message must name both the OpenAI-compatible fix and the
        LangChain fix so users don't have to search the docs to know what the
        runtime is asking of them."""
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError) as excinfo:
                fw.create_model("claude-3-5-sonnet-latest", "anthropic", {})
            message = str(excinfo.value)
            assert "anthropic" in message
            assert "parameters.base_url" in message
            assert "NEMOGUARDRAILS_LLM_FRAMEWORK=langchain" in message
            assert "langchain-<provider>" in message
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model("my-model", "custom", {"base_url": "https://my.api.com/v1"})

            assert model.provider_url == "https://my.api.com/v1"
        finally:
            await fw.reset()

    def test_get_provider_names(self):
        from nemoguardrails.llm.frameworks.default import DefaultFramework

        fw = DefaultFramework()
        names = fw.get_provider_names()
        assert "openai" in names
        assert "nim" in names
        assert "ollama" in names
