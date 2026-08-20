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

"""A reasoning model's reasoning must survive the trip through the HTTP server, on either engine.

One mocked model reply carrying both content and reasoning is driven through the real ASGI
app twice, once per engine, and every assertion is on the JSON the client receives. The
engines reach the same wire shape by unrelated routes -- LLMRails via ``BotThinking`` events,
IORails via ``LLMResponse.reasoning`` -- so the equivalence cases pin agreement that nothing
else in the codebase enforces.

The mock sits at each engine's model boundary rather than at the transport, because the two
engines do not share an HTTP client: LLMRails calls through ``httpx`` and IORails through
``aiohttp``, so no single transport-level mock covers both.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

pytest.importorskip("openai", reason="openai is required for server tests")
from httpx import ASGITransport, AsyncClient

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails import LLMRails
from nemoguardrails.server import api
from nemoguardrails.types import LLMResponse
from tests.guardrails.async_helpers import mock_rail_model
from tests.utils import FakeLLMModel

# A single main model and no rails: the narrowest config both engines accept, so any
# difference in the response is attributable to reasoning handling alone.
SINGLE_MODEL_CONFIG = """
models:
  - type: main
    engine: openai
    model: test-model
"""

CONTENT = "Hello! I'm doing well, thanks."
REASONING = "The user greeted me, so a short warm reply is enough."

# The same reply expressed the two ways a reasoning model can deliver it: a provider-native
# reasoning field, and inline <think> tags the engines must strip out of the content.
NATIVE_REASONING = LLMResponse(content=CONTENT, reasoning=REASONING)
INLINE_THINK_TAGS = LLMResponse(content=f"<think>{REASONING}</think>{CONTENT}")
NO_REASONING = LLMResponse(content=CONTENT)

ENGINES = ["llmrails", "iorails"]


def _build_rails(engine: str, llm_response: LLMResponse):
    """Build one engine wired to a model that always returns *llm_response*."""
    config = RailsConfig.from_content(yaml_content=SINGLE_MODEL_CONFIG)
    if engine == "llmrails":
        return LLMRails(config, llm=FakeLLMModel(llm_responses=[llm_response]))

    # IORails rejects an `llm` argument outright (`IORails.unsupported_reason`), so the double
    # goes on the engine registry instead. `require_iorails` keeps a silent fallback to
    # LLMRails from turning the IORails half of these tests into a second LLMRails run.
    guardrails = Guardrails(config, use_iorails=True, require_iorails=True)
    mock_rail_model(guardrails.rails_engine.engine_registry, AsyncMock(return_value=llm_response))
    return guardrails


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


@pytest.fixture(autouse=True)
def provider_api_keys(monkeypatch):
    """Satisfy config-time credential lookup without reaching a provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")


@pytest_asyncio.fixture
async def assistant_message():
    """Return a coroutine that POSTs one chat completion on the named engine and yields choices[0].message.

    The app is driven in-process on the running event loop rather than through ``TestClient``,
    which would run it in a portal loop of its own. IORails starts a worker queue and provider
    sessions on first generate, and those must be stopped from the loop that created them --
    a portal loop is already closed by teardown, so stopping from there raises
    ``RuntimeError: Event loop is closed``.
    """
    started = []

    async def _post(engine: str, llm_response: LLMResponse) -> dict:
        rails = _build_rails(engine, llm_response)
        started.append(rails)
        with patch.object(api, "_get_rails", AsyncMock(return_value=rails)):
            transport = ASGITransport(app=api.app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "stream": False,
                        "messages": [{"role": "user", "content": "Hello! How are you"}],
                        "guardrails": {"config_id": "reasoning"},
                    },
                )
        assert response.status_code == 200, response.text
        return response.json()["choices"][0]["message"]

    yield _post

    for rails in started:
        if isinstance(rails, Guardrails):
            await rails.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_native_reasoning_reaches_the_client(engine, assistant_message):
    """A provider-native reasoning field is delivered as message.reasoning_content."""
    message = await assistant_message(engine, NATIVE_REASONING)
    assert message.get("reasoning_content") == REASONING


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_inline_think_tags_reach_the_client(engine, assistant_message):
    """Reasoning arriving as inline <think> tags is delivered as message.reasoning_content."""
    message = await assistant_message(engine, INLINE_THINK_TAGS)
    assert message.get("reasoning_content") == REASONING


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("llm_response", [NATIVE_REASONING, INLINE_THINK_TAGS], ids=["native", "inline"])
async def test_reasoning_is_not_merged_into_content(engine, llm_response, assistant_message):
    """The answer is returned clean: reasoning rides in its own field, never inside content."""
    message = await assistant_message(engine, llm_response)
    assert message["content"] == CONTENT
    assert "<think>" not in message["content"]
    assert REASONING not in message["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ENGINES)
async def test_no_reasoning_omits_the_field(engine, assistant_message):
    """A model that returns no reasoning produces no reasoning_content key at all."""
    message = await assistant_message(engine, NO_REASONING)
    assert "reasoning_content" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_response",
    [NATIVE_REASONING, INLINE_THINK_TAGS, NO_REASONING],
    ids=["native", "inline", "none"],
)
async def test_engines_return_identical_messages(llm_response, assistant_message):
    """Both engines put the same model reply on the wire, so neither can drift from the other."""
    assert await assistant_message("llmrails", llm_response) == await assistant_message("iorails", llm_response)
