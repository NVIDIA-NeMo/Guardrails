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
from nemoguardrails.llm.constants import AZURE_PROVIDERS
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
    "azure": "AZURE_OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}

_UNSET: Any = object()


def _resolve_base_url(provider_name: str) -> str:
    url = _DEFAULT_BASE_URLS.get(provider_name)
    if url:
        return url
    raise ValueError(
        f"No default base_url for provider '{provider_name}'. "
        "If your endpoint is OpenAI-compatible, set parameters.base_url. "
        "Otherwise, set NEMOGUARDRAILS_LLM_FRAMEWORK=langchain and install "
        "the matching langchain-<provider> package (see migration guide)."
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

        if provider_name in AZURE_PROVIDERS:
            self._prepare_azure_kwargs(provider_name, kwargs)

        base_url = kwargs.pop("base_url", None) or _resolve_base_url(provider_name)

        # Sentinel-based pop so the Azure preset can set kwargs["api_key"] = None
        # to suppress Authorization: Bearer without falling back to the env-var
        # resolver. A missing key still resolves through _resolve_api_key.
        api_key = kwargs.pop("api_key", _UNSET)
        if api_key is _UNSET:
            api_key = _resolve_api_key(provider_name)

        timeout = kwargs.pop("timeout", None)
        connect_timeout = kwargs.pop("connect_timeout", None)
        max_retries = kwargs.pop("max_retries", None)
        default_headers = kwargs.pop("default_headers", None)
        default_query = kwargs.pop("default_query", None)

        client = self._get_or_create_client(
            base_url, api_key, timeout, connect_timeout, max_retries, default_headers, default_query
        )

        return OpenAIChatModel(client=client, model=model_name, provider_name=provider_name, **kwargs)

    def _prepare_azure_kwargs(self, provider_name: str, kwargs: Dict[str, Any]) -> None:
        """Reshape kwargs in place for the Azure preset.

        Validates Azure-specific inputs (``azure_endpoint`` or ``base_url``,
        ``azure_deployment``, ``api_version``, ``api_key``). Composes the
        deployment URL, sets ``api-version`` in ``default_query``, and writes
        the ``api-key`` header so the standard create_model path can build
        the client without an Azure-specific branch.

        Sets ``kwargs["api_key"] = None`` so the standard path does not emit
        the ``Authorization: Bearer`` header. Azure authenticates via the
        ``api-key`` header carried in ``default_headers``.

        The resource endpoint can be supplied as ``azure_endpoint`` (preferred,
        matches the OpenAI Python SDK) or ``base_url`` (compatibility alias for
        v0.21 LangChain configs). Both accept the same value (a resource-only
        URL such as ``https://my-resource.openai.azure.com/``); the deployment
        path is composed by this preset. Setting both raises an error.
        """
        azure_endpoint = kwargs.pop("azure_endpoint", None)
        base_url = kwargs.pop("base_url", None)
        if azure_endpoint and base_url:
            raise ValueError(
                f"Provider '{provider_name}' accepts either parameters.azure_endpoint "
                "or parameters.base_url, not both. Use azure_endpoint (preferred); "
                "base_url is a v0.21-compatibility alias for the same value."
            )
        resource_endpoint = azure_endpoint or base_url
        if not resource_endpoint:
            raise ValueError(
                f"Provider '{provider_name}' requires parameters.azure_endpoint "
                "(your Azure OpenAI resource endpoint, "
                "e.g. 'https://my-resource.openai.azure.com/'). "
                "parameters.base_url is also accepted as a v0.21-compatibility alias."
            )

        azure_deployment = kwargs.pop("azure_deployment", None)
        if not azure_deployment:
            raise ValueError(
                f"Provider '{provider_name}' requires parameters.azure_deployment "
                "(the deployment name configured in your Azure OpenAI resource)."
            )

        api_version = kwargs.pop("api_version", None)
        if not api_version:
            raise ValueError(
                f"Provider '{provider_name}' requires parameters.api_version "
                "(the Azure OpenAI API version, e.g. '2024-02-15-preview')."
            )

        default_query = dict(kwargs.pop("default_query", None) or {})
        if "api-version" in default_query and default_query["api-version"] != api_version:
            raise ValueError(
                f"Provider '{provider_name}' received conflicting Azure API versions. "
                "parameters.api_version must match default_query['api-version']."
            )
        default_query["api-version"] = api_version

        default_headers = dict(kwargs.pop("default_headers", None) or {})
        api_key = kwargs.pop("api_key", _UNSET)
        api_key_header_name = next((name for name in default_headers if name.lower() == "api-key"), None)
        if api_key is not _UNSET and api_key_header_name is not None:
            raise ValueError(
                f"Provider '{provider_name}' received conflicting Azure API keys. "
                "Set either parameters.api_key or default_headers['api-key'], not both."
            )
        if api_key_header_name is None:
            if api_key is _UNSET:
                api_key = _resolve_api_key(provider_name)
            if not api_key:
                raise ValueError(
                    f"Provider '{provider_name}' requires an API key. "
                    "Set AZURE_OPENAI_API_KEY in the environment, or set "
                    "api_key_env_var (or parameters.api_key) on the model entry."
                )
            default_headers["api-key"] = api_key

        kwargs["api_key"] = None
        kwargs["base_url"] = f"{resource_endpoint.rstrip('/')}/openai/deployments/{azure_deployment}"
        kwargs["default_query"] = default_query
        kwargs["default_headers"] = default_headers

    def register_provider(self, name: str, provider_cls: Any) -> None:
        self._providers[name] = provider_cls

    def get_provider_names(self) -> List[str]:
        return sorted({*_DEFAULT_BASE_URLS, *AZURE_PROVIDERS, *self._providers})

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
