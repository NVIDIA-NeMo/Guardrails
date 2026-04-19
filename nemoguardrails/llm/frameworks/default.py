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

import json
import logging
import os
from typing import Any, Dict, List, Optional

from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.models.openai_chat import OpenAIChatModel
from nemoguardrails.types import LLMModel

log = logging.getLogger(__name__)

_DEFAULT_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
    "nvidia_ai_endpoints": "https://integrate.api.nvidia.com/v1",
    "ollama": "http://localhost:11434/v1",
}

_API_KEY_ENV_VARS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "nim": "NVIDIA_API_KEY",
    "nvidia_ai_endpoints": "NVIDIA_API_KEY",
}


def _resolve_base_url(provider_name: str) -> str:
    url = _DEFAULT_BASE_URLS.get(provider_name)
    if url:
        return url
    raise ValueError(
        f"No default base_url for provider '{provider_name}'. "
        "Set it explicitly in model parameters: parameters.base_url"
    )


def _resolve_api_key(provider_name: str) -> Optional[str]:
    env_var = _API_KEY_ENV_VARS.get(provider_name)
    if env_var:
        return os.environ.get(env_var)
    return None


class DefaultFramework:
    def __init__(self):
        self._providers: Dict[str, Any] = {}
        self._clients: Dict[tuple, OpenAICompatibleClient] = {}

    def _get_or_create_client(
        self,
        base_url: str,
        api_key: Optional[str],
        timeout: Optional[float],
        connect_timeout: Optional[float],
        max_retries: Optional[int],
        default_headers: Optional[Dict[str, str]],
        default_query: Optional[Dict[str, Any]],
    ) -> OpenAICompatibleClient:
        key = (
            base_url,
            api_key or "",
            timeout,
            connect_timeout,
            max_retries,
            json.dumps(default_headers or {}, sort_keys=True, default=str),
            json.dumps(default_query or {}, sort_keys=True, default=str),
        )
        if key not in self._clients:
            client_kwargs: Dict[str, Any] = {}
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            if connect_timeout is not None:
                client_kwargs["connect_timeout"] = connect_timeout
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
            if default_headers is not None:
                client_kwargs["custom_headers"] = default_headers
            if default_query is not None:
                client_kwargs["custom_query"] = default_query
            self._clients[key] = OpenAICompatibleClient(base_url=base_url, api_key=api_key, **client_kwargs)
        return self._clients[key]

    def create_model(
        self,
        model_name: str,
        provider_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMModel:
        kwargs = dict(model_kwargs) if model_kwargs else {}
        kwargs.pop("mode", None)

        if provider_name in self._providers:
            return self._providers[provider_name](model=model_name, **kwargs)

        base_url = kwargs.pop("base_url", None) or _resolve_base_url(provider_name)
        api_key = kwargs.pop("api_key", None) or _resolve_api_key(provider_name)
        timeout = kwargs.pop("timeout", None)
        connect_timeout = kwargs.pop("connect_timeout", None)
        max_retries = kwargs.pop("max_retries", None)
        default_headers = kwargs.pop("default_headers", None)
        default_query = kwargs.pop("default_query", None)

        client = self._get_or_create_client(
            base_url, api_key, timeout, connect_timeout, max_retries, default_headers, default_query
        )

        return OpenAIChatModel(client=client, model=model_name, provider_name=provider_name, **kwargs)

    def register_provider(self, name: str, provider_cls: Any) -> None:
        self._providers[name] = provider_cls

    def get_provider_names(self) -> List[str]:
        return sorted({*_DEFAULT_BASE_URLS, *self._providers})

    async def aclose(self) -> None:
        """Close all pooled HTTP clients and drop them from the pool.

        Connection-pool teardown only. Registered providers are kept.
        Mirrors ``httpx.AsyncClient.aclose()`` semantics: an async resource
        cleanup hook that releases sockets and TLS sessions back to the OS.

        Re-creating models after ``aclose`` works as expected: the next call
        to ``create_model`` for a given config rebuilds the client.

        If any ``client.close()`` fails, all remaining closes are still
        attempted; the first error is re-raised after the pool is cleared.
        """
        errors = []
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception as exc:
                errors.append(exc)
                log.warning("Error closing pooled client: %s", exc)
        self._clients.clear()
        if errors:
            raise errors[0]

    def clear_providers(self) -> None:
        """Drop all providers registered via ``register_provider``.

        Registry teardown only. Pooled HTTP clients are not affected; if
        the registered provider classes constructed clients via the
        framework, those clients survive in the pool until ``aclose()``.
        """
        self._providers.clear()

    async def reset(self) -> None:
        """Test-only convenience: tear down both pools and providers.

        Equivalent to ``await fw.aclose(); fw.clear_providers()``. In
        production code, prefer the granular methods so connection refresh
        doesn't accidentally drop registered providers.
        """
        try:
            await self.aclose()
        finally:
            self.clear_providers()
