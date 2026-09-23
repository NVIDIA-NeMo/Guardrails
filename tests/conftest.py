# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import os
from importlib.metadata import version
from unittest.mock import Mock, patch

import pytest
from aiohttp import ClientResponse
from packaging.version import Version

from tests.testing.embeddings import (
    DeterministicEmbeddingSearchProvider,
)

try:
    import langchain_core  # noqa: F401
except ImportError:
    collect_ignore_glob = ["integrations/langchain/*.py", "integrations/langchain/**/*.py"]

REASONING_TRACE_MOCK_PATH = "nemoguardrails.actions.llm.generation.get_and_clear_reasoning_trace_contextvar"
_FASTEMBED_ENGINES = {"FastEmbed"}
_FASTEMBED_MODELS = {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}
_REAL_EMBEDDING_ENV_VARS = (
    "LIVE_TEST",
    "LIVE_TEST_MODE",
    "TEST_LIVE_MODE",
)


@pytest.fixture(scope="session", autouse=True)
def patch_aioresponses_client_response():
    from aioresponses import core

    if "stream_writer" not in inspect.signature(ClientResponse).parameters:
        yield
        return

    if Version(version("aioresponses")) > Version("0.7.9"):
        pytest.fail(
            "aioresponses has advanced beyond 0.7.9; verify aiohttp compatibility and remove patch_aioresponses_client_response."
        )

    class CompatibleClientResponse(ClientResponse):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("stream_writer", Mock(output_size=0))
            super().__init__(*args, **kwargs)

    with patch.object(core, "ClientResponse", CompatibleClientResponse):
        yield


@pytest.fixture(autouse=True)
def reset_reasoning_trace_var():
    """Reset reasoning_trace_var before each test to prevent state leakage."""
    from nemoguardrails.context import reasoning_trace_var

    reasoning_trace_var.set(None)
    yield
    reasoning_trace_var.set(None)


@pytest.fixture(autouse=True)
def reset_tool_calls_var():
    from nemoguardrails.context import tool_calls_var

    tool_calls_var.set(None)
    yield
    tool_calls_var.set(None)


@pytest.fixture(autouse=True)
def reset_explain_info_var():
    from nemoguardrails.context import explain_info_var

    explain_info_var.set(None)
    yield
    explain_info_var.set(None)


@pytest.fixture(autouse=True)
def use_deterministic_embeddings_for_default_fastembed(monkeypatch, request):
    if request.node.get_closest_marker("real_embeddings") or any(
        os.environ.get(name) for name in _REAL_EMBEDDING_ENV_VARS
    ):
        return

    from nemoguardrails.rails.llm.llmrails import LLMRails

    original_get_embedding_search_provider = LLMRails._get_embeddings_search_provider_instance

    def patched_get_embedding_search_provider(self, esp_config=None):
        if esp_config is None or _uses_default_fastembed(
            esp_config,
            self.default_embedding_engine,
            self.default_embedding_model,
        ):
            parameters = getattr(esp_config, "parameters", {}) if esp_config is not None else {}
            return DeterministicEmbeddingSearchProvider(**parameters)

        return original_get_embedding_search_provider(self, esp_config)

    monkeypatch.setattr(
        LLMRails,
        "_get_embeddings_search_provider_instance",
        patched_get_embedding_search_provider,
    )


@pytest.fixture
def langchain_framework():
    from nemoguardrails.llm.frameworks import _reset_frameworks, set_default_framework

    _reset_frameworks()
    set_default_framework("langchain")
    yield
    _reset_frameworks()


def pytest_configure(config):
    patch("prompt_toolkit.PromptSession", autospec=True).start()


def _uses_default_fastembed(provider_config, default_engine, default_model):
    if provider_config.name != "default":
        return False

    parameters = provider_config.parameters
    engine = parameters.get("embedding_engine", default_engine)
    model = parameters.get("embedding_model", default_model)
    return engine in _FASTEMBED_ENGINES and model in _FASTEMBED_MODELS
