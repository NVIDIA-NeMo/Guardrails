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

import asyncio

import pytest
from fastapi.testclient import TestClient

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails import guardrails as guardrails_module
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server import api
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG

REASONING_TRACE = "The user asked for a capital city."
LLM_ANSWER = "Paris."
THINK_PREFIXED_ANSWER = f"<think>{REASONING_TRACE}</think>\n{LLM_ANSWER}"


class _StubLLMRails:
    """Stand-in for a real LLMRails: not an IORails instance, and keeps reasoning in the structured field.

    The shipped configs use the `nim` engine, so constructing a real LLMRails would need
    provider credentials; only the engine dispatch matters here. The signature accepts both
    the server's `LLMRails(config=..., verbose=...)` call and `Guardrails`' positional
    `LLMRails(config, llm, verbose)` fallback.
    """

    def __init__(self, config, llm=None, verbose=False):
        self.config = config
        self.events_history_cache = {}

    async def generate_async(self, messages=None, options=None, state=None, **kwargs):
        return GenerationResponse(
            response=[{"role": "assistant", "content": LLM_ANSWER}],
            reasoning_content=REASONING_TRACE,
        )


def _post_chat_completion():
    """POST a single-turn request to /v1/chat/completions against the content_safety config."""
    client = TestClient(api.app, raise_server_exceptions=False)
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "guardrails": {"config_id": "content_safety"},
        },
    )


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


@pytest.fixture
def llmrails_engine(monkeypatch, iorails_compatible_config):
    """Serve the same config through a non-Guardrails engine: the contrast case for `iorails_alias`."""
    monkeypatch.setattr(api, "LLMRails", _StubLLMRails)
    monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: iorails_compatible_config))
    yield


def test_chat_completion_inlines_reasoning_as_think_tags_under_iorails_alias(iorails_alias, monkeypatch):
    """Reasoning split into `reasoning_content` by IORails reaches the client as a <think> prefix on the content."""

    async def _fake_start(self):
        self._running = True

    async def _fake_generate_async(self, messages, **kwargs):
        return GenerationResponse(
            response=[{"role": "assistant", "content": LLM_ANSWER}],
            reasoning_content=REASONING_TRACE,
        )

    monkeypatch.setattr(IORails, "start", _fake_start)
    monkeypatch.setattr(IORails, "generate_async", _fake_generate_async)

    response = _post_chat_completion()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == THINK_PREFIXED_ANSWER


def test_chat_completion_leaves_reasoning_alone_on_the_llmrails_engine(llmrails_engine):
    """A non-Guardrails engine keeps the message content clean, so the fold stays scoped to IORails."""
    response = _post_chat_completion()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == LLM_ANSWER


def test_inline_reasoning_clears_the_field_once_folded_into_content():
    """`reasoning_content` is cleared after its trace is folded into the assistant message."""
    res = GenerationResponse(
        response=[{"role": "assistant", "content": LLM_ANSWER}],
        reasoning_content=REASONING_TRACE,
    )

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] == THINK_PREFIXED_ANSWER
    assert folded.reasoning_content is None


def test_inline_reasoning_keeps_the_trace_when_a_tool_call_has_no_content():
    """A tool-call-only message has no content to prefix, so the reasoning stays in its field rather than being dropped."""
    res = GenerationResponse(
        response=[{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}],
        reasoning_content=REASONING_TRACE,
    )

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] is None
    assert folded.reasoning_content == REASONING_TRACE


def test_inline_reasoning_leaves_content_alone_when_there_is_no_trace():
    """A response without reasoning passes through with its content untouched."""
    res = GenerationResponse(response=[{"role": "assistant", "content": LLM_ANSWER}])

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] == LLM_ANSWER
    assert folded.reasoning_content is None


def test_inline_reasoning_keeps_the_trace_when_the_response_is_a_bare_string():
    """A string-shaped `response` has no message dict to fold into, so the trace is kept."""
    res = GenerationResponse(response=LLM_ANSWER, reasoning_content=REASONING_TRACE)

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response == LLM_ANSWER
    assert folded.reasoning_content == REASONING_TRACE


def test_inline_reasoning_keeps_the_trace_when_no_message_is_from_the_assistant():
    """Only assistant messages carry reasoning, so a response without one keeps the trace in its field."""
    res = GenerationResponse(
        response=[{"role": "user", "content": "What is the capital of France?"}],
        reasoning_content=REASONING_TRACE,
    )

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] == "What is the capital of France?"
    assert folded.reasoning_content == REASONING_TRACE


def test_inline_reasoning_folds_only_into_the_assistant_message():
    """With several messages present, the <think> prefix lands on the assistant turn alone."""
    res = GenerationResponse(
        response=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": LLM_ANSWER},
        ],
        reasoning_content=REASONING_TRACE,
    )

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] == "What is the capital of France?"
    assert folded.response[1]["content"] == THINK_PREFIXED_ANSWER
    assert folded.reasoning_content is None


@pytest.fixture
def guardrails_over_llmrails(monkeypatch, iorails_compatible_config):
    """Alias LLMRails->Guardrails but force the wrapper onto its LLMRails fallback engine.

    `Guardrails` falls back to LLMRails whenever `IORails.unsupported_reason` returns a reason,
    so the server can hold a `Guardrails` that is not IORails-backed.
    """
    monkeypatch.setattr(api, "LLMRails", Guardrails)
    monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: iorails_compatible_config))
    monkeypatch.setattr(IORails, "unsupported_reason", classmethod(lambda cls, config, llm=None: "forced fallback"))
    monkeypatch.setattr(guardrails_module, "LLMRails", _StubLLMRails)
    yield


def test_chat_completion_leaves_reasoning_alone_when_guardrails_wraps_llmrails(guardrails_over_llmrails):
    """A Guardrails wrapper on its LLMRails fallback engine keeps clean content: the fold is IORails-only."""
    response = _post_chat_completion()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == LLM_ANSWER


def test_guardrails_over_llmrails_is_not_iorails_backed(guardrails_over_llmrails):
    """The fallback fixture really produces a non-IORails engine, so the contrast test above is meaningful."""

    async def _resolve():
        return await api._get_rails(["content_safety"])

    rails = asyncio.run(_resolve())

    assert isinstance(rails, Guardrails)
    assert not isinstance(rails.rails_engine, IORails)


def test_inline_reasoning_does_not_duplicate_an_existing_think_block():
    """Content that already carries an inline <think> block is left alone rather than given a second one."""
    inline = f"<think>inline trace</think>\n{LLM_ANSWER}"
    res = GenerationResponse(response=[{"role": "assistant", "content": inline}], reasoning_content=REASONING_TRACE)

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] == inline
    assert folded.response[0]["content"].count("<think>") == 1
    assert folded.reasoning_content == REASONING_TRACE
