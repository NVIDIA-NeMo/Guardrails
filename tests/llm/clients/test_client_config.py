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

import httpx
import pytest

from nemoguardrails.llm.clients.constants import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
)
from nemoguardrails.llm.clients.openai_chat_model import OpenAIChatModel
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient


def _make_client(**kwargs):
    return OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key="sk-test", **kwargs)


class TestTimeout:
    def test_defaults(self):
        client = _make_client()
        assert client._client.timeout.read == DEFAULT_TIMEOUT.read
        assert client._client.timeout.connect == DEFAULT_TIMEOUT.connect

    def test_custom(self):
        client = _make_client(timeout=120.0, connect_timeout=10.0)
        assert client._client.timeout.read == 120.0
        assert client._client.timeout.connect == 10.0

    def test_http_client_timeout_inferred(self):
        custom = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0))
        client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client=custom)
        assert client._client is custom


class TestConnectionPool:
    def test_limits(self):
        client = _make_client()
        pool = client._client._transport._pool
        assert pool._max_connections == DEFAULT_CONNECTION_LIMITS.max_connections
        assert pool._max_keepalive_connections == DEFAULT_CONNECTION_LIMITS.max_keepalive_connections


class TestMaxRetries:
    def test_default(self):
        client = _make_client()
        assert client._max_retries == DEFAULT_MAX_RETRIES

    def test_custom(self):
        client = _make_client(max_retries=5)
        assert client._max_retries == 5


class TestCustomHeaders:
    def test_merged_into_request(self):
        client = _make_client(custom_headers={"X-Custom": "value"})
        headers = client._build_headers()
        assert headers["X-Custom"] == "value"
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_override_defaults(self):
        client = OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            custom_headers={"Content-Type": "text/plain"},
        )
        headers = client._build_headers()
        assert headers["Content-Type"] == "text/plain"


class TestCustomQuery:
    def test_stored(self):
        client = _make_client(custom_query={"api-version": "2024-02-01"})
        assert client._custom_query == {"api-version": "2024-02-01"}


class TestHttpClientInjection:
    def test_uses_injected_client(self):
        custom = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key="sk-test", http_client=custom)
        assert client._client is custom

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


class TestDefaultFramework:
    def test_creates_chat_model(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        model = fw.create_model("gpt-4o", "openai", {"api_key": "sk-test"})

        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == "gpt-4o"
        assert model.provider_url == "https://api.openai.com/v1"

    def test_creates_nim(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        model = fw.create_model("llama", "nim", {"api_key": "nvapi-test"})

        assert isinstance(model, OpenAIChatModel)
        assert model.provider_url == "https://integrate.api.nvidia.com/v1"

    def test_pools_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-test"})
        m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk-test"})

        assert m1._client is m2._client

    def test_different_keys_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-one"})
        m2 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-two"})

        assert m1._client is not m2._client

    def test_different_timeout_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "timeout": 30.0})
        m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "timeout": 5.0})

        assert m1._client is not m2._client
        assert m1._client._client.timeout.read == 30.0
        assert m2._client._client.timeout.read == 5.0

    def test_different_headers_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "default_headers": {"X-A": "1"}})
        m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "default_headers": {"X-B": "2"}})

        assert m1._client is not m2._client
        assert m1._client._custom_headers == {"X-A": "1"}
        assert m2._client._custom_headers == {"X-B": "2"}

    def test_same_full_config_pooled(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        cfg = {"api_key": "sk", "timeout": 30.0, "default_headers": {"X-A": "1"}}
        m1 = fw.create_model("gpt-4o", "openai", cfg.copy())
        m2 = fw.create_model("gpt-4o-mini", "openai", cfg.copy())

        assert m1._client is m2._client

    @pytest.mark.asyncio
    async def test_reset_closes_all_pooled_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        fw.register_provider("custom", lambda **kw: object())
        assert "custom" in fw._providers

        await fw.reset()

        assert fw._providers == {}

    @pytest.mark.asyncio
    async def test_reset_allows_recreation_with_fresh_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

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

        from nemoguardrails.llm.default_framework import DefaultFramework

        injected = httpx.AsyncClient()
        client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client=injected)
        fw = DefaultFramework()
        fw._clients[("injected",)] = client

        await fw.reset()

        assert not injected.is_closed
        await injected.aclose()

    def test_unknown_provider_raises(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        with pytest.raises(ValueError, match="No default base_url"):
            fw.create_model("model", "unknown_provider", {})

    def test_custom_base_url(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        model = fw.create_model("my-model", "custom", {"base_url": "https://my.api.com/v1"})

        assert model.provider_url == "https://my.api.com/v1"

    def test_get_provider_names(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        names = fw.get_provider_names()
        assert "openai" in names
        assert "nim" in names
        assert "ollama" in names
