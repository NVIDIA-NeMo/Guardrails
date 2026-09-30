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
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails import guardrails as guardrails_module
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server import api
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG
from tests.guardrails.test_tool_rails_iorails import TOOL_CONFIG, WEATHER_TOOL
from tests.guardrails.tool_helpers import make_tool_conversation

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


def test_inline_reasoning_folds_when_think_is_mentioned_mid_content():
    """Content that merely mentions <think> outside a leading block still receives its reasoning prefix."""
    quoted = f"{LLM_ANSWER} Reasoning models wrap traces in <think> tags."
    res = GenerationResponse(
        response=[{"role": "assistant", "content": quoted}],
        reasoning_content=REASONING_TRACE,
    )

    folded = api._inline_reasoning_as_think_tags(res)

    assert folded.response[0]["content"] == f"<think>{REASONING_TRACE}</think>\n{quoted}"
    assert folded.reasoning_content is None


@pytest.fixture
def tool_rails_alias(monkeypatch):
    """Alias LLMRails->Guardrails and serve a config with both tool rails, with IORails start stubbed."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    tool_rails_config = RailsConfig.from_content(config=TOOL_CONFIG)

    async def _fake_start(self):
        self._running = True

    monkeypatch.setattr(api, "LLMRails", Guardrails)
    monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: tool_rails_config))
    monkeypatch.setattr(IORails, "start", _fake_start)
    yield


@pytest.fixture
def stubbed_main_model(monkeypatch):
    """Stub the main model at its engine, so the IORails rails in front of it run for real."""
    main_model = AsyncMock(return_value=LLMResponse(content="It is sunny."))
    monkeypatch.setattr(ModelEngine, "chat_completion", main_model)
    return main_model


def _post_tool_conversation(messages: list):
    """POST *messages* to /v1/chat/completions against the tool-rails config."""
    client = TestClient(api.app, raise_server_exceptions=False)
    return client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": messages, "guardrails": {"config_id": "tools"}},
    )


def test_chat_completion_accepts_a_spec_shaped_tool_result_under_iorails_alias(tool_rails_alias, stubbed_main_model):
    """A tool message without `name`, as the OpenAI spec shapes it, passes the tool-result rail and reaches the model."""
    response = _post_tool_conversation(make_tool_conversation(result_name=None))

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "It is sunny."
    stubbed_main_model.assert_awaited_once()


def test_chat_completion_refuses_an_unlinked_tool_result_under_iorails_alias(tool_rails_alias, stubbed_main_model):
    """A tool message whose call id links to no prior call is refused before the model is called."""
    response = _post_tool_conversation(make_tool_conversation(result_call_id="call_999", result_name=None))

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == REFUSAL_MESSAGE
    stubbed_main_model.assert_not_awaited()


def _post_check(messages: list, rail_types: list, *, tools: "list | None" = None, context: "dict | None" = None):
    """POST *messages* to /v1/checks against the tool-rails config."""
    guardrails = {"config_id": "tools", "rail_types": rail_types}
    if context is not None:
        guardrails["context"] = context
    body = {"model": "test-model", "messages": messages, "guardrails": guardrails}
    if tools is not None:
        body["tools"] = tools
    # Each request of an un-entered TestClient runs on a fresh event loop, which an IORails
    # engine cached by an earlier request cannot serve, so each request builds its own.
    api.llm_rails_instances.clear()
    client = TestClient(api.app, raise_server_exceptions=False)
    return client.post("/v1/checks", json=body)


def _tool_call_turn(name: str = "get_weather", arguments: str = '{"city": "Paris"}') -> list:
    """A user question, then an assistant turn calling *name* with id ``call_1``."""
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}],
        },
    ]


def test_check_blocks_a_disallowed_tool_call_under_iorails_alias(tool_rails_alias, stubbed_main_model):
    """/v1/checks refuses an undeclared call with the rail, the reason and a structured violation, without the model."""
    response = _post_check(_tool_call_turn("delete_files", "{}"), ["tool_call"], tools=[WEATHER_TOOL])

    assert response.status_code == 200
    assert response.json() == {
        "status": "blocked",
        "content": REFUSAL_MESSAGE,
        "rail": "tool call validation",
        "reason": "tool call 'delete_files' is not an allowed tool",
        "tool_violations": [
            {
                "kind": "tool_call",
                "violation_type": "tool_not_allowed",
                "reason": "tool call 'delete_files' is not an allowed tool",
                "tool_call_id": "call_1",
                "tool_name": "delete_files",
                "index": 0,
            }
        ],
    }
    stubbed_main_model.assert_not_awaited()


def test_check_honours_the_request_tools_under_iorails_alias(tool_rails_alias, stubbed_main_model):
    """The request's top-level tools are the allowlist: the same call is refused without them and passes with them."""
    without_tools = _post_check(_tool_call_turn(), ["tool_call"])
    with_tools = _post_check(_tool_call_turn(), ["tool_call"], tools=[WEATHER_TOOL])

    assert (without_tools.json()["status"], with_tools.json()["status"]) == ("blocked", "passed")
    stubbed_main_model.assert_not_awaited()


def test_check_round_trip_under_iorails_alias(tool_rails_alias, stubbed_main_model):
    """A harness checks its model's tool call, then the spec-shaped tool message it appends, and both pass."""
    messages = _tool_call_turn()

    call_check = _post_check(messages, ["tool_call"], tools=[WEATHER_TOOL])
    messages.append({"role": "tool", "tool_call_id": "call_1", "content": "18C"})
    result_check = _post_check(messages, ["tool_result"])

    assert (call_check.json()["status"], result_check.json()["status"]) == ("passed", "passed")
    assert "tool_violations" not in result_check.json()
    stubbed_main_model.assert_not_awaited()


@pytest.mark.parametrize("rail_type", ["tool_call", "tool_result"])
def test_tool_checks_accept_guardrails_context_under_iorails_alias(tool_rails_alias, stubbed_main_model, rail_type):
    """The context message the server prepends does not break either tool check."""
    messages = [*_tool_call_turn(), {"role": "tool", "tool_call_id": "call_1", "content": "18C"}]

    response = _post_check(messages, [rail_type], tools=[WEATHER_TOOL], context={"user_id": "u1"})

    assert response.status_code == 200
    assert response.json()["status"] == "passed"


def test_result_violation_index_counts_request_messages_under_iorails_alias(tool_rails_alias, stubbed_main_model):
    """With context prepended, an unlinked result's index is still its position in the request's messages."""
    messages = [*_tool_call_turn(), {"role": "tool", "tool_call_id": "call_999", "content": "18C"}]

    response = _post_check(messages, ["tool_result"], context={"user_id": "u1"})

    assert [violation["index"] for violation in response.json()["tool_violations"]] == [2]


class _StubLLMRailsWithCheck(_StubLLMRails):
    """The LLMRails stand-in with LLMRails' own check_async, which refuses tool rail types before any model runs."""

    check_async = LLMRails.check_async


@pytest.fixture
def tool_rails_on_llmrails(monkeypatch):
    """Serve the tool-rails config through a non-Guardrails engine: the contrast case for `tool_rails_alias`."""
    tool_rails_config = RailsConfig.from_content(config=TOOL_CONFIG)
    monkeypatch.setattr(api, "LLMRails", _StubLLMRailsWithCheck)
    monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(lambda full_path: tool_rails_config))
    yield


def test_check_tool_rail_type_on_llmrails_returns_422(tool_rails_on_llmrails):
    """LLMRails cannot run tool rail types, so /v1/checks answers 422 rather than passing or failing with a 500."""
    response = _post_check(_tool_call_turn(), ["tool_call"], tools=[WEATHER_TOOL])

    assert response.status_code == 422
    assert "IORails" in response.json()["error"]["message"]
