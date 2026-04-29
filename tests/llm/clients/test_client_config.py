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

import asyncio
import warnings

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
    @pytest.mark.asyncio
    async def test_defaults(self):
        async with _make_client() as client:
            httpx_client = client._get_client()
            assert httpx_client.timeout.read == DEFAULT_TIMEOUT.read
            assert httpx_client.timeout.connect == DEFAULT_TIMEOUT.connect

    @pytest.mark.asyncio
    async def test_custom(self):
        async with _make_client(timeout=120.0, connect_timeout=10.0) as client:
            httpx_client = client._get_client()
            assert httpx_client.timeout.read == 120.0
            assert httpx_client.timeout.connect == 10.0

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
            pool = client._get_client()._transport._pool
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
        owned = client._get_client()
        await client.close()
        assert owned.is_closed

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="httpx.AsyncClient"):
            OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client="not a client")


class TestPoolKeyAcceptsUnhashableQueryValues:
    @pytest.mark.asyncio
    async def test_list_query_value_does_not_crash(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "default_query": {"tags": ["a", "b"]}})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "default_query": {"tags": ["a", "b"]}})
            assert m1._client is m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_query_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model("llama", "nim", {"api_key": "nvapi-test"})

            assert isinstance(model, OpenAIChatModel)
            assert model.provider_url == "https://integrate.api.nvidia.com/v1"
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_pools_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-test"})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk-test"})

            assert m1._client is m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_keys_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-one"})
            m2 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-two"})

            assert m1._client is not m2._client
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_timeout_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk", "timeout": 30.0})
            m2 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk", "timeout": 5.0})

            assert m1._client is not m2._client
            assert m1._client._get_client().timeout.read == 30.0
            assert m2._client._get_client().timeout.read == 5.0
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_different_headers_different_clients(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

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
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        m1 = fw.create_model("gpt-4o", "openai", {"api_key": "sk-a"})
        m2 = fw.create_model("llama", "nim", {"api_key": "nv-a"})
        m3 = fw.create_model("gpt-4o-mini", "openai", {"api_key": "sk-b"})

        clients = [m1._client._get_client(), m2._client._get_client(), m3._client._get_client()]
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
        first_client = m1._client._get_client()
        await fw.reset()

        m2 = fw.create_model("gpt-4o", "openai", {"api_key": "sk"})
        assert m2._client._get_client() is not first_client
        assert not m2._client._get_client().is_closed

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

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            with pytest.raises(ValueError, match="No default base_url"):
                fw.create_model("model", "unknown_provider", {})
        finally:
            await fw.reset()

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        try:
            model = fw.create_model("my-model", "custom", {"base_url": "https://my.api.com/v1"})

            assert model.provider_url == "https://my.api.com/v1"
        finally:
            await fw.reset()

    def test_get_provider_names(self):
        from nemoguardrails.llm.default_framework import DefaultFramework

        fw = DefaultFramework()
        names = fw.get_provider_names()
        assert "openai" in names
        assert "nim" in names
        assert "ollama" in names


class TestLoopScopedHttpxClient:
    """Regression coverage for the 'Event loop is closed' bug.

    The framework cached one OpenAICompatibleClient per provider URL, and
    that client held a single httpx.AsyncClient that bound to the first
    event loop on which it issued a request. Reusing the same rails across
    asyncio.run() boundaries (or pytest-asyncio function-scoped loops) then
    raised RuntimeError: Event loop is closed.

    With per-loop lazy httpx clients in BaseClient, each loop gets its own
    httpx.AsyncClient on first use; the WeakKeyDictionary purges entries
    when their loop is garbage collected.
    """

    def test_reused_client_across_two_asyncio_runs(self):
        client = _make_client()

        async def fetch_underlying():
            return client._get_client()

        first = asyncio.run(fetch_underlying())
        second = asyncio.run(fetch_underlying())

        assert first is not second, "different loops must yield different httpx clients"
        assert isinstance(first, httpx.AsyncClient)
        assert isinstance(second, httpx.AsyncClient)

    def test_reused_client_across_function_scope_loops(self):
        client = _make_client()
        seen = []

        async def fetch_underlying():
            return client._get_client()

        for _ in range(3):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                seen.append(loop.run_until_complete(fetch_underlying()))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        assert len(set(id(c) for c in seen)) == 3, "each loop must get its own httpx client"

    def test_loop_gc_drops_client_entry(self):
        import gc

        client = _make_client()

        async def fetch_underlying():
            return client._get_client()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(fetch_underlying())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        assert len(client._clients_by_loop) == 1

        del loop
        gc.collect()
        assert len(client._clients_by_loop) == 0, "WeakKeyDictionary must drop entry after loop GC"

    def test_injected_client_not_per_loop(self):
        injected = httpx.AsyncClient()
        try:
            client = OpenAICompatibleClient(base_url="https://api.openai.com/v1", http_client=injected)

            async def fetch_underlying():
                return client._get_client()

            first = asyncio.run(fetch_underlying())
            second = asyncio.run(fetch_underlying())

            assert first is injected
            assert second is injected
            assert len(client._clients_by_loop) == 0
        finally:
            asyncio.run(injected.aclose())

    @pytest.mark.asyncio
    async def test_close_only_targets_current_loop_client(self):
        client = _make_client()
        current = client._get_client()
        assert client._clients_by_loop[asyncio.get_running_loop()] is current

        await client.close()

        assert current.is_closed
        assert asyncio.get_running_loop() not in client._clients_by_loop

    def test_close_in_loop_with_no_client_is_noop(self):
        """close() popping must not raise when the running loop has no entry.

        After the setup loop ends and is GC'd, the WeakKeyDictionary is empty.
        Calling close() on a fresh loop must be a clean no-op: no exception,
        no spurious creation of a client just to close it.
        """

        async def setup():
            client = _make_client()
            client._get_client()
            return client

        client = asyncio.run(setup())

        async def close_in_new_loop():
            await client.close()
            return len(client._clients_by_loop)

        remaining = asyncio.run(close_in_new_loop())
        assert remaining == 0

    def test_full_chat_completion_across_two_asyncio_runs(self):
        """End-to-end: same client survives `asyncio.run` -> close loop -> `asyncio.run` again.

        Mirrors scenario A of the bug repro. Without the per-loop client this
        raises RuntimeError: Event loop is closed on the second call.
        """

        def handler(request):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": "ok", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                request=request,
            )

        from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient as _Client

        client = _Client(base_url="https://api.openai.com/v1", api_key="sk-test")

        original_factory = client._client_factory

        def factory_with_mock_transport():
            real = original_factory()
            real._transport = httpx.MockTransport(handler)
            return real

        client._client_factory = factory_with_mock_transport

        async def call_once():
            return await client.chat_completion("gpt-4o-mini", [{"role": "user", "content": "hi"}])

        first = asyncio.run(call_once())
        second = asyncio.run(call_once())

        assert first["choices"][0]["message"]["content"] == "ok"
        assert second["choices"][0]["message"]["content"] == "ok"

    def test_streaming_chat_completion_across_two_asyncio_runs(self):
        """Streaming path also rebuilds the httpx client per loop.

        _apost_stream uses _get_client() the same way as _apost, but only
        end-to-end coverage proves the SSE pipeline survives a loop teardown.
        """

        def handler(request):
            body = (
                'data: {"id":"s","model":"x","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
                'data: {"id":"s","model":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                content=body.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient as _Client

        client = _Client(base_url="https://api.openai.com/v1", api_key="sk-test")

        original_factory = client._client_factory

        def factory_with_mock_transport():
            real = original_factory()
            real._transport = httpx.MockTransport(handler)
            return real

        client._client_factory = factory_with_mock_transport

        async def stream_once():
            chunks = []
            async for chunk in client.stream_chat_completion("gpt-4o-mini", [{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        first = asyncio.run(stream_once())
        second = asyncio.run(stream_once())

        assert len(first) >= 1
        assert len(second) >= 1
        assert any(chunk["choices"][0]["delta"].get("content") == "hi" for chunk in first)
        assert any(chunk["choices"][0]["delta"].get("content") == "hi" for chunk in second)
