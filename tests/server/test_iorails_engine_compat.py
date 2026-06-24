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

"""Server compatibility when NEMO_GUARDRAILS_IORAILS_ENGINE aliases LLMRails to Guardrails.

Under that env var, ``nemoguardrails.server.api`` builds a ``Guardrails`` wrapper that
selects the stateless IORails engine for IORails-compatible configs. These tests
reproduce the reported 500: ``_get_rails`` assigns ``events_history_cache``, which used
to raise ``NotImplementedError`` on an IORails-backed wrapper. The alias is simulated by
patching ``api.LLMRails`` to ``Guardrails`` and ``RailsConfig.from_path`` to return an
IORails-compatible content-safety config, so the test does not depend on the global
import-time env var.
"""

import pytest
from fastapi.testclient import TestClient

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.server import api
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG


@pytest.fixture
def iorails_compatible_config():
    """An IORails-compatible Colang 1.0 content-safety config loaded once for the test."""
    return RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)


@pytest.fixture(autouse=True)
def reset_server_state():
    """Clear the per-config rails caches and force multi-config mode around each test."""
    original_single_config_mode = api.app.single_config_mode
    api.app.single_config_mode = False
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    yield
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    api.app.single_config_mode = original_single_config_mode


@pytest.fixture
def iorails_alias(monkeypatch, iorails_compatible_config):
    """Simulate NEMO_GUARDRAILS_IORAILS_ENGINE: alias LLMRails->Guardrails and serve the IORails config."""
    monkeypatch.setattr(api, "LLMRails", Guardrails)
    monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: iorails_compatible_config))
    yield


@pytest.mark.asyncio
async def test_get_rails_does_not_raise_under_iorails_alias(iorails_alias):
    """_get_rails returns an IORails-backed Guardrails with an inert events_history_cache instead of raising.

    This is the exact crash site from the report: the events_history_cache assignment at
    the end of _get_rails. Under IORails it must round-trip inertly (read -> {}) rather
    than raise NotImplementedError.
    """
    rails = await api._get_rails(["content_safety"])

    assert isinstance(rails, Guardrails)
    assert isinstance(rails.rails_engine, IORails)
    assert rails.events_history_cache == {}


def test_chat_completion_returns_200_under_iorails_alias(iorails_alias, monkeypatch):
    """POST /v1/chat/completions returns 200 when the server runs on the IORails engine.

    Generation is stubbed at the IORails boundary (start + generate_async) so the test
    exercises the server plumbing and the _get_rails cache assignment without any network
    or provider credentials; the response shape is asserted to be OpenAI-compatible.
    """

    async def _fake_start(self):
        self._running = True

    async def _fake_generate_async(self, messages, **kwargs):
        return {"role": "assistant", "content": "hello from iorails"}

    monkeypatch.setattr(IORails, "start", _fake_start)
    monkeypatch.setattr(IORails, "generate_async", _fake_generate_async)

    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "content_safety"},
        },
    )

    assert response.status_code == 200
    res = response.json()
    assert res["object"] == "chat.completion"
    assert res["choices"][0]["message"]["role"] == "assistant"
    assert res["choices"][0]["message"]["content"] == "hello from iorails"
