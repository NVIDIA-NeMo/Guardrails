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

"""Unit tests for iorails module."""

import asyncio
import gc
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from nemoguardrails import Guardrails
from nemoguardrails.guardrails.guardrails_types import RailDirection, RailResult
from nemoguardrails.guardrails.iorails import (
    INTERNAL_ERROR_MESSAGE,
    REFUSAL_MESSAGE,
    IORails,
    _GeneratedTurn,
    _rewritten_bot_message,
    _rewritten_user_message,
)
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.types import LLMResponse, LLMResponseChunk, ToolCall, ToolCallFunction
from tests.guardrails.rail_stubs import bot_message_rewrite, rail_failure, user_message_rewrite
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG, NEMOGUARDS_CONFIG


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def rails_config():
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_sync(rails_config):
    return IORails(rails_config)


@pytest_asyncio.fixture
async def iorails(rails_config):
    iorails = IORails(rails_config)
    try:
        yield iorails
    finally:
        await iorails.stop()


class TestIORailsInit:
    """Test IORails wires up EngineRegistry and RailsManager from config."""

    def test_creates_engine_registry(self, iorails_sync):
        """EngineRegistry is created during init."""
        assert iorails_sync.engine_registry is not None

    def test_creates_rails_manager(self, iorails_sync):
        """RailsManager is created during init."""
        assert iorails_sync.rails_manager is not None

    def test_rails_manager_uses_engine_registry(self, iorails_sync):
        """RailsManager receives the same EngineRegistry instance."""
        assert iorails_sync.rails_manager.engine_registry is iorails_sync.engine_registry


DEFAULT_HEADERS_CONFIG = {
    "models": [
        {
            "type": "main",
            "engine": "nim",
            "model": "meta/llama-3.3-70b-instruct",
            "parameters": {"default_headers": {"X-Main-Route": "main-pool"}},
        },
        {
            "type": "content_safety",
            "engine": "nim",
            "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
            "parameters": {"default_headers": {"X-Safety-Route": "safety-pool"}},
        },
    ],
}


@pytest_asyncio.fixture
async def iorails_with_header_config():
    """Build an IORails whose main and content_safety models carry distinct
    parameters.default_headers, for end-to-end per-model header assertions."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=DEFAULT_HEADERS_CONFIG))
    try:
        yield iorails
    finally:
        await iorails.stop()


class TestConfigDefaultHeadersEndToEnd:
    """End-to-end: IORails built from config routes each model's
    parameters.default_headers onto that model's outbound request, and the
    main LLM and a second LLM carry only their own headers."""

    @staticmethod
    def _mock_client():
        """Build a mock aiohttp client whose post() records its call args."""
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False
        return mock_client

    async def _headers_for_model(self, iorails, model_type):
        """Route a model_call for model_type through a mock client and return the sent headers."""
        engine = iorails.engine_registry._get_engine(model_type, ModelEngine)
        engine._client = self._mock_client()
        engine._running = True
        await iorails.engine_registry.model_call(model_type, [{"role": "user", "content": "Hi"}])
        return engine._client.post.call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_main_llm_carries_only_its_config_headers(self, iorails_with_header_config):
        """The main model request carries its own default_header plus base headers, and not the second model's."""
        headers = await self._headers_for_model(iorails_with_header_config, "main")
        assert headers["X-Main-Route"] == "main-pool"
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"
        assert "X-Safety-Route" not in headers

    @pytest.mark.asyncio
    async def test_second_llm_carries_only_its_config_headers(self, iorails_with_header_config):
        """The content_safety model request carries its own default_header plus base headers, and not the main model's."""
        headers = await self._headers_for_model(iorails_with_header_config, "content_safety")
        assert headers["X-Safety-Route"] == "safety-pool"
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"
        assert "X-Main-Route" not in headers

    @pytest.mark.asyncio
    async def test_main_and_second_llm_get_distinct_headers(self, iorails_with_header_config):
        """The two models' outbound header sets differ, proving per-model routing."""
        main_headers = await self._headers_for_model(iorails_with_header_config, "main")
        safety_headers = await self._headers_for_model(iorails_with_header_config, "content_safety")
        assert main_headers != safety_headers
        assert main_headers.get("X-Main-Route") == "main-pool"
        assert safety_headers.get("X-Safety-Route") == "safety-pool"


class TestConfigDefaultHeadersOverHTTP:
    """True end-to-end: run generate_async over a real loopback HTTP server and
    assert on the headers the aiohttp client actually put on the wire."""

    @pytest.mark.asyncio
    async def test_config_default_headers_sent_on_the_wire(self):
        """A configured default_header reaches the provider request, alongside the
        api-key Authorization, and never leaks into the JSON body."""
        captured: dict = {}

        async def handler(request):
            captured["headers"] = dict(request.headers)
            captured["body"] = await request.json()
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            config = RailsConfig.from_content(
                config={
                    "models": [
                        {
                            "type": "main",
                            "engine": "openai",
                            "model": "test-model",
                            "parameters": {
                                "base_url": str(server.make_url("/")),
                                "default_headers": {"X-Tenant": "acme"},
                            },
                        }
                    ]
                }
            )
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                iorails = IORails(config)
            async with iorails:
                result = await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])
        finally:
            await server.close()

        assert result == {"role": "assistant", "content": "ok"}
        assert captured["headers"]["X-Tenant"] == "acme"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert "X-Tenant" not in captured["body"]


class TestConfigDefaultQueryOverHTTP:
    """True end-to-end: run generate_async over a real loopback HTTP server and
    assert on the query string the aiohttp client actually put on the wire."""

    @staticmethod
    async def _capture_request(default_query: dict) -> dict:
        """Run one generate_async against a loopback server and return what the server received."""
        captured: dict = {}

        async def handler(request):
            captured["path"] = request.path
            captured["query"] = list(request.rel_url.query.items())
            captured["body"] = await request.json()
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        try:
            config = RailsConfig.from_content(
                config={
                    "models": [
                        {
                            "type": "main",
                            "engine": "openai",
                            "model": "test-model",
                            "parameters": {
                                "base_url": str(server.make_url("/")),
                                "default_query": default_query,
                            },
                        }
                    ]
                }
            )
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                iorails = IORails(config)
            async with iorails:
                captured["result"] = await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])
        finally:
            await server.close()
        return captured

    @pytest.mark.asyncio
    async def test_config_default_query_sent_on_the_wire(self):
        """Configured query parameters reach the provider request URL, leaving the path and the JSON body alone."""
        captured = await self._capture_request({"tenant": "acme-corp", "api-version": "2024-10-21"})

        assert captured["result"] == {"role": "assistant", "content": "ok"}
        assert captured["query"] == [("tenant", "acme-corp"), ("api-version", "2024-10-21")]
        assert captured["path"] == "/v1/chat/completions"
        assert "tenant" not in captured["body"]
        assert "default_query" not in captured["body"]

    @pytest.mark.asyncio
    async def test_list_and_bool_query_values_survive_the_wire(self):
        """A list value repeats the key on the wire and a boolean arrives lowercased."""
        captured = await self._capture_request({"tag": ["red", "blue"], "beta": True})

        assert captured["query"] == [("tag", "red"), ("tag", "blue"), ("beta", "true")]


class TestGenerateAsync:
    """Test the generate_async input-check → LLM → output-check pipeline."""

    @pytest.mark.asyncio
    async def test_safe_input_and_output(self, iorails):
        """Returns LLM response when both input and output rails pass."""
        messages = [{"role": "user", "content": "hi"}]
        llm_response = "Hello from LLM"

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=llm_response))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        result = await iorails.generate_async(messages=messages)

        assert result == {"role": "assistant", "content": llm_response}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages, enabled=True)
        iorails.engine_registry.model_call.assert_called_once_with("main", messages)
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, llm_response, enabled=True)

    @pytest.mark.asyncio
    async def test_safe_input_and_output_with_generation_options(self, iorails):
        """Passing options returns a GenerationResponse wrapping the LLM response."""
        messages = [{"role": "user", "content": "hi"}]
        llm_response = "Hello from LLM"

        llm_params = {"temperature": 0.01, "max_completion_tokens": 1000}
        options = GenerationOptions(llm_params=llm_params)

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=llm_response))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        result = await iorails.generate_async(messages=messages, options=options)

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": llm_response}]
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages, enabled=True)
        iorails.engine_registry.model_call.assert_called_once_with("main", messages, **llm_params)
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, llm_response, enabled=True)

    @pytest.mark.asyncio
    async def test_safe_input_and_output_call_sequence(self, iorails):
        """Pipeline executes in order: input check → generate → output check."""
        call_order = []

        async def mock_input_safe(messages, *, enabled=True):
            call_order.append("input")
            return RailResult.allow()

        async def mock_generate(model_type, messages):
            call_order.append("generate")
            return LLMResponse(content="response")

        async def mock_output_safe(messages, response, *, enabled=True):
            call_order.append("output")
            return RailResult.allow()

        iorails.rails_manager.is_input_safe = mock_input_safe
        iorails.engine_registry.model_call = mock_generate
        iorails.rails_manager.is_output_safe = mock_output_safe

        await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])
        assert call_order == ["input", "generate", "output"]

    @pytest.mark.asyncio
    async def test_unsafe_input(self, iorails):
        """Returns refusal and skips LLM + output check when input is unsafe."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.block(reason="blocked"))
        iorails.engine_registry.model_call = AsyncMock()
        iorails.rails_manager.is_output_safe = AsyncMock()

        messages = [{"role": "user", "content": "bad input"}]
        result = await iorails.generate_async(messages=messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages, enabled=True)
        iorails.engine_registry.model_call.assert_not_called()
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsafe_output(self, iorails):
        """Returns refusal when output check fails, even though LLM was called."""
        messages = [{"role": "user", "content": "hi"}]
        llm_response = "Unsafe response from the LLM!"

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content=llm_response))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.block(reason="blocked"))

        result = await iorails.generate_async(messages=messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        iorails.rails_manager.is_input_safe.assert_called_once_with(messages, enabled=True)
        iorails.engine_registry.model_call.assert_called_once_with("main", messages)
        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, llm_response, enabled=True)

    @pytest.mark.asyncio
    async def test_unsafe_output_records_block_metric_when_metrics_enabled(self, iorails):
        """When metrics are enabled, an output-rails block records an OUTPUT block metric."""
        iorails._metrics_enabled = True
        messages = [{"role": "user", "content": "hi"}]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="unsafe"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.block(reason="blocked"))

        with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_blocked:
            result = await iorails.generate_async(messages=messages)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_blocked.assert_called_once_with(RailDirection.OUTPUT)

    @pytest.mark.asyncio
    async def test_failed_input_rail_returns_the_internal_error(self, iorails):
        """A rail that raised renders the internal error, not the refusal a real block gets."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=rail_failure("f5 guardrails scan input"))
        iorails.engine_registry.model_call = AsyncMock()
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": INTERNAL_ERROR_MESSAGE}
        iorails.engine_registry.model_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_input_rail_returns_the_internal_error_with_generation_options(self, iorails):
        """The structured return shape carries the same sentence as the bare one."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=rail_failure("f5 guardrails scan input"))
        iorails.engine_registry.model_call = AsyncMock()
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(messages=[{"role": "user", "content": "hi"}], options=GenerationOptions())

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": INTERNAL_ERROR_MESSAGE}]

    @pytest.mark.asyncio
    async def test_failed_output_rail_returns_the_internal_error(self, iorails):
        """An output rail that raised is reported the same way an input one is."""
        messages = [{"role": "user", "content": "hi"}]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="an answer"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=rail_failure("f5 guardrails scan output"))

        result = await iorails.generate_async(messages=messages)

        assert result == {"role": "assistant", "content": INTERNAL_ERROR_MESSAGE}

    @pytest.mark.asyncio
    async def test_failed_tool_result_rail_returns_the_internal_error(self, iorails):
        """A tool-result rail that raised short-circuits before generation with the internal error."""
        iorails.rails_manager.are_tool_results_safe = AsyncMock(return_value=rail_failure("tool result validation"))
        iorails.rails_manager.is_input_safe = AsyncMock()
        iorails.engine_registry.model_call = AsyncMock()

        result = await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])

        assert result == {"role": "assistant", "content": INTERNAL_ERROR_MESSAGE}
        iorails.engine_registry.model_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_tool_call_rail_returns_the_internal_error(self, iorails):
        """A tool-call rail that raised discards the model's tool calls with the internal error."""
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
            )
        ]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="", tool_calls=tool_calls))
        iorails.rails_manager.are_tool_calls_safe = AsyncMock(return_value=rail_failure("tool call validation"))
        iorails.rails_manager.is_output_safe = AsyncMock()

        result = await iorails.generate_async(messages=[{"role": "user", "content": "weather?"}])

        assert result == {"role": "assistant", "content": INTERNAL_ERROR_MESSAGE}
        iorails.rails_manager.is_output_safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_dict_options_forwarded(self, iorails):
        """Dict options are converted to GenerationOptions and forwarded."""
        messages = [{"role": "user", "content": "hi"}]
        llm_params = {"temperature": 0.01, "max_completion_tokens": 1000}

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ok"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        await iorails.generate_async(messages=messages, options={"llm_params": llm_params})

        iorails.engine_registry.model_call.assert_called_once_with("main", messages, **llm_params)

    @pytest.mark.asyncio
    async def test_a_turn_with_neither_a_response_nor_a_verdict_raises(self, iorails):
        """A generation bug must surface as an error, not as a plausible-looking guardrail block.

        Both generation paths name the blocking verdict on the one branch that drops the
        response, so this state is unreachable today; the guard is what keeps a future path
        from reporting a missing response to the caller as a refusal.
        """
        messages = [{"role": "user", "content": "hi"}]
        iorails._speculative_generation = False
        iorails._do_generate_sequential = AsyncMock(return_value=_GeneratedTurn(response=None, messages=messages))

        with pytest.raises(RuntimeError, match="without naming a blocking rail"):
            await iorails.generate_async(messages=messages)

    @pytest.mark.asyncio
    async def test_generate_async_propagates_exception(self, iorails):
        """Exceptions from the LLM call propagate to the caller."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("LLM internal error"))

        with pytest.raises(RuntimeError, match="LLM internal error"):
            await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])


class TestToolCalling:
    """Test tool-call forwarding (request body) and return (assistant message)."""

    @pytest.mark.asyncio
    async def test_tools_in_llm_params_forwarded_to_model(self, iorails):
        """Tool definitions in options.llm_params are forwarded to model_call unchanged."""
        messages = [{"role": "user", "content": "weather?"}]
        tool = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        options = GenerationOptions(llm_params={"tools": [tool], "tool_choice": "auto", "parallel_tool_calls": True})

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ok"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        await iorails.generate_async(messages=messages, options=options)

        iorails.engine_registry.model_call.assert_called_once_with(
            "main", messages, tools=[tool], tool_choice="auto", parallel_tool_calls=True
        )

    @pytest.mark.asyncio
    async def test_tool_calls_returned_on_assistant_message(self, iorails):
        """Tool calls from the model are serialized OpenAI-native (JSON-string arguments)."""
        messages = [{"role": "user", "content": "weather?"}]
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(name="get_weather", arguments={"city": "Paris"}),
            )
        ]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="", tool_calls=tool_calls))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        result = await iorails.generate_async(messages=messages)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert result["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
        ]

    @pytest.mark.asyncio
    async def test_output_rails_skipped_for_tool_call_only_response(self, iorails):
        """A tool-call-only response (no text) skips the content output rails."""
        messages = [{"role": "user", "content": "weather?"}]
        tool_calls = [ToolCall(id="c1", type="function", function=ToolCallFunction(name="f", arguments={}))]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="", tool_calls=tool_calls))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        result = await iorails.generate_async(messages=messages)

        iorails.rails_manager.is_output_safe.assert_not_called()
        assert result["tool_calls"][0]["function"]["name"] == "f"

    @pytest.mark.asyncio
    async def test_text_with_tool_calls_runs_output_rails_and_returns_both(self, iorails):
        """When a response has text and tool calls, output rails run on the text and both are returned."""
        messages = [{"role": "user", "content": "hi"}]
        tool_calls = [ToolCall(id="c1", type="function", function=ToolCallFunction(name="f", arguments={}))]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="some text", tool_calls=tool_calls)
        )
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        result = await iorails.generate_async(messages=messages)

        iorails.rails_manager.is_output_safe.assert_called_once_with(messages, "some text", enabled=True)
        assert result["content"] == "some text"
        assert result["tool_calls"][0]["function"]["name"] == "f"


class TestIORailsLifecycle:
    """Test IORails start/stop lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, iorails):
        """start() delegates to engine_registry.start(), stop() to engine_registry.stop()."""
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()

        assert not iorails._running
        await iorails.start()
        assert iorails._running
        iorails.engine_registry.start.assert_called_once()

        await iorails.stop()
        assert not iorails._running
        iorails.engine_registry.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, iorails):
        """Calling start() twice only starts engine_registry once."""
        iorails.engine_registry.start = AsyncMock()

        await iorails.start()
        await iorails.start()

        iorails.engine_registry.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, iorails):
        """Calling stop() twice only stops engine_registry once."""
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()

        await iorails.start()
        await iorails.stop()
        await iorails.stop()

        iorails.engine_registry.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, iorails):
        """stop() without a prior start() does not raise."""
        iorails.engine_registry.stop = AsyncMock()

        await iorails.stop()
        assert not iorails._running
        iorails.engine_registry.stop.assert_not_called()


class TestIORailsStartErrors:
    """Test IORails start() error propagation."""

    @pytest.mark.asyncio
    async def test_start_propagates_engine_registry_error(self, iorails):
        """start() re-raises when engine_registry.start() fails and leaves _running=False."""
        iorails.engine_registry.start = AsyncMock(side_effect=RuntimeError("engine failed"))

        with pytest.raises(RuntimeError, match="engine failed"):
            await iorails.start()

        assert not iorails._running

    @pytest.mark.asyncio
    async def test_start_propagates_model_engine_error(self, iorails):
        """A ModelEngine failure propagates through EngineRegistry up to IORails.start()."""
        # Inject the failure at the ModelEngine level, leaving EngineRegistry real
        engine = iorails.engine_registry._get_engine("content_safety", ModelEngine)
        engine.start = AsyncMock(side_effect=RuntimeError("NIM endpoint unreachable"))

        # Mock the other engines so they don't make real HTTP connections
        for engine_type, engine in iorails.engine_registry._engines.items():
            if engine_type != "content_safety":
                engine.start = AsyncMock()
                engine.stop = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            await iorails.start()

        assert not iorails._running
        assert not iorails.engine_registry._running

        # With fail-fast, only engines that started before the failing one are rolled back.
        # Engines after it in iteration order never had start() called.
        engine_names = list(iorails.engine_registry._engines.keys())
        failed_idx = engine_names.index("content_safety")
        for i, name in enumerate(engine_names):
            eng = iorails.engine_registry._engines[name]
            if i < failed_idx:
                eng.stop.assert_called_once()
            elif name != "content_safety":
                eng.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_failure_allows_retry(self, iorails):
        """After a failed start(), a subsequent start() is not blocked by the idempotency guard."""
        iorails.engine_registry.start = AsyncMock(side_effect=RuntimeError("first fail"))

        with pytest.raises(RuntimeError):
            await iorails.start()
        assert not iorails._running

        # Call start() without an exception being thrown, now this will work
        iorails.engine_registry.start = AsyncMock()
        await iorails.start()
        assert iorails._running
        iorails.engine_registry.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_rolls_back_engine_registry_when_queue_start_raises(self, iorails):
        """If _generate_async_queue.start() raises, engine_registry is rolled back and _running stays False."""
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()
        iorails._generate_async_queue.start = AsyncMock(side_effect=RuntimeError("queue start failed"))

        with pytest.raises(RuntimeError, match="queue start failed"):
            await iorails.start()

        iorails.engine_registry.start.assert_called_once()
        iorails.engine_registry.stop.assert_called_once()
        assert not iorails._running

    @pytest.mark.asyncio
    async def test_start_failed_queue_start_invokes_rollback_in_order(self, iorails):
        """After registry.start succeeds and queue.start fails, registry.stop is the next call."""
        call_order: list[str] = []

        async def registry_start():
            call_order.append("registry.start")

        async def registry_stop():
            call_order.append("registry.stop")

        async def queue_start():
            call_order.append("queue.start")
            raise RuntimeError("queue failed")

        iorails.engine_registry.start = registry_start
        iorails.engine_registry.stop = registry_stop
        iorails._generate_async_queue.start = queue_start

        with pytest.raises(RuntimeError, match="queue failed"):
            await iorails.start()

        # Pre-fix: ["registry.start", "queue.start"] — no rollback, _running=True.
        # Post-fix: rollback runs after the failed queue start, _running=False.
        assert call_order == ["registry.start", "queue.start", "registry.stop"]
        assert not iorails._running

    @pytest.mark.asyncio
    async def test_start_propagates_queue_error_when_registry_rollback_also_raises(self, iorails):
        """Original queue.start error propagates even if the registry.stop rollback raises.
        Cleanup-path failures must not mask the actionable root cause.
        """
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock(side_effect=RuntimeError("registry rollback boom"))
        iorails._generate_async_queue.start = AsyncMock(side_effect=RuntimeError("queue failed"))

        # The QUEUE failure (the root cause) is what propagates, not the rollback failure.
        with pytest.raises(RuntimeError, match="queue failed"):
            await iorails.start()

        iorails.engine_registry.stop.assert_called_once()
        assert not iorails._running

    @pytest.mark.asyncio
    async def test_start_rolls_back_queue_and_registry_when_gauge_registration_raises(self, iorails):
        """If register_nonstream_saturation_gauges raises after the queue is up, BOTH the queue
        and the registry are rolled back so a retry of start() comes from a clean state.
        Without this, _running stays False while the queue is alive — stop() then no-ops and
        leaks worker tasks (the gap Greptile flagged).
        """
        iorails._metrics_enabled = True
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()
        iorails._generate_async_queue.start = AsyncMock()
        iorails._generate_async_queue.stop = AsyncMock()

        with patch(
            "nemoguardrails.guardrails.iorails.register_nonstream_saturation_gauges",
            side_effect=RuntimeError("gauge registration failed"),
        ):
            with pytest.raises(RuntimeError, match="gauge registration failed"):
                await iorails.start()

        iorails.engine_registry.start.assert_called_once()
        iorails._generate_async_queue.start.assert_called_once()
        iorails._generate_async_queue.stop.assert_called_once()
        iorails.engine_registry.stop.assert_called_once()
        assert not iorails._running
        assert not iorails._gauges_registered

    @pytest.mark.asyncio
    async def test_start_failed_gauge_registration_invokes_rollback_in_order(self, iorails):
        """Gauge-registration failure rolls back queue first, then registry."""
        iorails._metrics_enabled = True
        call_order: list[str] = []

        async def registry_start():
            call_order.append("registry.start")

        async def registry_stop():
            call_order.append("registry.stop")

        async def queue_start():
            call_order.append("queue.start")

        async def queue_stop():
            call_order.append("queue.stop")

        def gauge_register(*args, **kwargs):
            call_order.append("gauge.register")
            raise RuntimeError("gauge failed")

        iorails.engine_registry.start = registry_start
        iorails.engine_registry.stop = registry_stop
        iorails._generate_async_queue.start = queue_start
        iorails._generate_async_queue.stop = queue_stop

        with patch(
            "nemoguardrails.guardrails.iorails.register_nonstream_saturation_gauges",
            side_effect=gauge_register,
        ):
            with pytest.raises(RuntimeError, match="gauge failed"):
                await iorails.start()

        # Queue is rolled back before registry — reverse-order shutdown of resources
        # acquired in ``registry.start → queue.start → gauge.register`` order.
        assert call_order == ["registry.start", "queue.start", "gauge.register", "queue.stop", "registry.stop"]
        assert not iorails._running

    @pytest.mark.asyncio
    async def test_start_propagates_gauge_error_when_queue_rollback_raises(self, iorails):
        """Original gauge error propagates even if queue.stop rollback raises.
        Cleanup-path failures must not mask the actionable root cause.
        """
        iorails._metrics_enabled = True
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()
        iorails._generate_async_queue.start = AsyncMock()
        iorails._generate_async_queue.stop = AsyncMock(side_effect=RuntimeError("queue rollback boom"))

        with patch(
            "nemoguardrails.guardrails.iorails.register_nonstream_saturation_gauges",
            side_effect=RuntimeError("gauge failed"),
        ):
            # The GAUGE failure (the root cause) is what propagates, not the rollback failure.
            with pytest.raises(RuntimeError, match="gauge failed"):
                await iorails.start()

        # Both rollback steps were attempted even though queue.stop raised.
        iorails._generate_async_queue.stop.assert_called_once()
        iorails.engine_registry.stop.assert_called_once()
        assert not iorails._running

    @pytest.mark.asyncio
    async def test_start_skips_gauge_registration_when_metrics_disabled(self, iorails):
        """With metrics disabled, register_nonstream_saturation_gauges is never called —
        so a hypothetical bug in that helper can't break IORails.start().
        """
        iorails._metrics_enabled = False
        iorails.engine_registry.start = AsyncMock()
        iorails._generate_async_queue.start = AsyncMock()

        with patch(
            "nemoguardrails.guardrails.iorails.register_nonstream_saturation_gauges",
            side_effect=AssertionError("must not be called when metrics disabled"),
        ) as gauge_mock:
            await iorails.start()

        gauge_mock.assert_not_called()
        assert iorails._running
        assert not iorails._gauges_registered

    @pytest.mark.asyncio
    async def test_gauges_registered_only_once_across_restarts(self, iorails):
        """``_gauges_registered`` is sticky across stop/start cycles — gauges register once
        for the lifetime of the IORails instance.  OTEL has no public unregister API for
        observable instruments; the soft-disable via ``is_running=lambda: self._running``
        is what makes a stopped IORails emit no observations on the same gauges.
        """
        iorails._metrics_enabled = True
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()
        iorails._generate_async_queue.start = AsyncMock()
        iorails._generate_async_queue.stop = AsyncMock()

        with patch(
            "nemoguardrails.guardrails.iorails.register_nonstream_saturation_gauges",
        ) as gauge_mock:
            await iorails.start()
            assert iorails._gauges_registered
            await iorails.stop()
            await iorails.start()

        gauge_mock.assert_called_once()
        assert iorails._running


class TestIORailsStopErrors:
    """Test IORails stop() error propagation."""

    @pytest.mark.asyncio
    async def test_stop_propagates_engine_registry_error(self, iorails):
        """stop() re-raises when engine_registry.stop() fails, but sets _running=False."""
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        await iorails.start()
        assert iorails._running

        with pytest.raises(RuntimeError, match="stop failed"):
            await iorails.stop()

        # _running should be False due to the finally clause
        assert not iorails._running

    @pytest.mark.asyncio
    async def test_stop_runs_engine_registry_when_queue_stop_raises(self, iorails):
        """engine_registry.stop() must run even if _generate_async_queue.stop() raises."""
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()
        iorails._generate_async_queue.stop = AsyncMock(side_effect=RuntimeError("queue stop failed"))

        await iorails.start()
        assert iorails._running

        with pytest.raises(RuntimeError, match="queue stop failed"):
            await iorails.stop()

        iorails.engine_registry.stop.assert_called_once()
        assert not iorails._running


class TestIORailsContextManager:
    """Test IORails async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_start_and_stop(self, iorails):
        """async with calls start() on enter and stop() on exit."""
        iorails.engine_registry.start = AsyncMock()
        iorails.engine_registry.stop = AsyncMock()

        async with iorails as engine:
            assert engine is iorails
            assert iorails._running
            iorails.engine_registry.start.assert_called_once()

        assert not iorails._running
        iorails.engine_registry.stop.assert_called_once()


class TestAutoStart:
    """Test that generate_async and stream_async auto-start IORails.

    Verify IORails.start() is called internally at the start of generate_async() and
    inside stream_async()'s _wrapped_iterator.
    """

    @pytest_asyncio.fixture
    async def iorails_input_only(self):
        """IORails with no output rails (needed for stream_async without StreamingNotSupportedError).

        Yields an *unstarted* IORails (no ``async with``): ``TestAutoStart`` tests
        assert ``not iorails._running`` before invoking ``stream_async`` to verify
        the auto-start contract, so the fixture must not pre-call ``start()``.
        """
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            input_only_config = {
                **NEMOGUARDS_CONFIG,
                "rails": {**NEMOGUARDS_CONFIG["rails"], "output": {"flows": []}},
            }
            iorails = IORails(RailsConfig.from_content(config=input_only_config))
        yield iorails
        await iorails.stop()

    @pytest.mark.asyncio
    async def test_generate_async_calls_start(self, iorails):
        """generate_async() calls start() automatically before running the pipeline."""
        iorails.engine_registry.start = AsyncMock()
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ok"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        assert not iorails._running
        await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])

        iorails.engine_registry.start.assert_called_once()
        assert iorails._running

    @pytest.mark.asyncio
    async def test_generate_async_start_is_idempotent(self, iorails):
        """Two generate_async() calls only trigger start() once."""
        iorails.engine_registry.start = AsyncMock()
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ok"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])
        await iorails.generate_async(messages=[{"role": "user", "content": "hi"}])

        iorails.engine_registry.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_async_calls_start(self, iorails_input_only):
        """stream_async() calls start() automatically before streaming."""

        async def mock_stream(model_type, messages, **kwargs):
            yield LLMResponseChunk(delta_content="hello")

        iorails_input_only.engine_registry.start = AsyncMock()
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails_input_only.engine_registry.stream_model_call = mock_stream

        assert not iorails_input_only._running
        chunks = [
            chunk async for chunk in iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}])
        ]

        iorails_input_only.engine_registry.start.assert_called_once()
        assert iorails_input_only._running
        assert chunks == ["hello"]

    @pytest.mark.asyncio
    async def test_stream_async_start_is_idempotent(self, iorails_input_only):
        """Two stream_async() calls only trigger start() once."""

        async def mock_stream(model_type, messages, **kwargs):
            yield LLMResponseChunk(delta_content="hi")

        iorails_input_only.engine_registry.start = AsyncMock()
        iorails_input_only.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails_input_only.engine_registry.stream_model_call = mock_stream

        _ = [chunk async for chunk in iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}])]
        _ = [chunk async for chunk in iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}])]

        iorails_input_only.engine_registry.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_async_propagates_start_failure(self, iorails_input_only):
        """start() failure inside stream_async propagates to the caller."""
        iorails_input_only.engine_registry.start = AsyncMock(side_effect=RuntimeError("engine unavailable"))

        with pytest.raises(RuntimeError, match="engine unavailable"):
            _ = [chunk async for chunk in iorails_input_only.stream_async(messages=[{"role": "user", "content": "hi"}])]


class TestGenerate:
    """Test the synchronous generate() method."""

    def test_generate_delegates_to_generate_async(self, iorails_sync):
        """generate() creates a temp IORails, starts it, calls generate_async, and stops it."""
        iorails = iorails_sync
        messages = [{"role": "user", "content": "hi"}]
        expected = {"role": "assistant", "content": "Hello from LLM"}

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="Hello from LLM"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        # Patch IORails so the temp instance inside generate() uses our mocked iorails
        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails):
            result = iorails.generate(messages=messages)

        assert result == expected

    def test_generate_passes_kwargs(self, iorails_sync):
        """generate() forwards kwargs to generate_async."""
        iorails = iorails_sync
        messages = [{"role": "user", "content": "hi"}]
        options = GenerationOptions(llm_params={"temperature": 0.5})

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="response"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails):
            iorails.generate(messages=messages, options=options)

        iorails.engine_registry.model_call.assert_called_once_with("main", messages, temperature=0.5)

    def test_generate_marks_temp_engine_as_internal(self, iorails_sync):
        """generate() suppresses usage reporting for its temporary bridge engine."""
        iorails = iorails_sync
        messages = [{"role": "user", "content": "hi"}]

        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="response"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails) as mock_iorails:
            iorails.generate(messages=messages)

        mock_iorails.assert_called_once()
        assert mock_iorails.call_args.kwargs == {"_report_usage": False}

    def test_generate_raises_when_called_from_async_loop(self, iorails_sync):
        """generate() raises RuntimeError when called inside a running event loop."""

        async def call_generate():
            iorails_sync.generate(messages=[{"role": "user", "content": "hi"}])

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")

            with pytest.raises(RuntimeError, match="cannot be called from a running event loop"):
                asyncio.run(call_generate())

            gc.collect()

        assert not [warning for warning in recorded_warnings if issubclass(warning.category, RuntimeWarning)]


class TestRefusalMessage:
    """Test the REFUSAL_MESSAGE module constant."""

    def test_refusal_message_is_string(self):
        """REFUSAL_MESSAGE is a non-empty string."""
        assert isinstance(REFUSAL_MESSAGE, str)
        assert len(REFUSAL_MESSAGE) > 0

    def test_the_internal_error_is_not_the_refusal(self):
        """Aliasing the two would silently undo the distinction the whole fix rests on."""
        assert INTERNAL_ERROR_MESSAGE != REFUSAL_MESSAGE


class TestIORailsConfigToCallURL:
    """End-to-end: from a RailsConfig with a user-supplied base_url to the URL the engine POSTs.

    Covers regression for issue #1861 / PR #1862: a base_url with or without a trailing
    "/v1" must produce a single /v1/chat/completions in the final HTTP request.
    """

    @pytest.mark.parametrize(
        "base_url_input",
        [
            "https://custom.example.com",
            "https://custom.example.com/v1",
        ],
    )
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_main_model_url_not_doubled(self, base_url_input):
        """A user base_url with or without /v1 yields a single /v1 in the POST URL through the full IORails pipeline."""
        config = RailsConfig.from_content(
            config={
                "models": [
                    {
                        "type": "main",
                        "engine": "nim",
                        "model": "meta/llama-3.3-70b-instruct",
                        "parameters": {"base_url": base_url_input},
                    },
                ],
            }
        )
        iorails = IORails(config)

        # No rails are configured, so RailsManager short-circuits both checks to
        # is_safe=True and the only outbound HTTP call is the main-model call.
        # Pre-seed the main engine's HTTP client + _running=True so BaseEngine.start()
        # short-circuits via its "if running: return" guard — no real aiohttp session
        # is ever created.
        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

        mock_client = AsyncMock()
        mock_client.post = MagicMock(return_value=mock_response)
        mock_client.closed = False

        main_engine = iorails.engine_registry._engines["main"]
        main_engine._client = mock_client
        main_engine._running = True

        try:
            await iorails.generate_async(messages=[{"role": "user", "content": "Hi"}])
        finally:
            await iorails.stop()

        url = mock_client.post.call_args[0][0]
        assert url == "https://custom.example.com/v1/chat/completions"
        assert "/v1/v1/" not in url


def _make_mock_http_client(content: str):
    """Build an AsyncMock RetryClient whose .post() returns a chat-completions response with the given content."""
    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"choices": [{"message": {"content": content}}]})
    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.closed = False
    return mock_client


class TestGuardrailsConfigToCallURLs:
    """End-to-end through the top-level Guardrails class: every model's POST URL
    must have a single /v1, including rail-engine models.

    Uses CONTENT_SAFETY_CONFIG (input + output content-safety rails + main model),
    overrides both models with /v1-suffixed base_urls, and verifies that
    guardrails.generate_async() composes correct URLs for every outbound HTTP call.
    """

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_main_and_content_safety_urls_not_doubled(self):
        """Both main and content_safety models POST to single-/v1 URLs through guardrails.generate_async()."""
        # SAFE_OUTPUT_JSON works for both input and output content-safety parsers:
        # the input parser keys on "User Safety", the output parser on "Response Safety".
        safe_output_json = '{"User Safety": "safe", "Response Safety": "safe"}'

        config_dict = {
            **CONTENT_SAFETY_CONFIG,
            "models": [
                {**CONTENT_SAFETY_CONFIG["models"][0], "parameters": {"base_url": "https://main.example.com/v1"}},
                {**CONTENT_SAFETY_CONFIG["models"][1], "parameters": {"base_url": "https://safety.example.com/v1"}},
            ],
        }
        config = RailsConfig.from_content(config=config_dict)
        guardrails = Guardrails(config=config)

        # Sanity: CONTENT_SAFETY_CONFIG flows are IORails-eligible, so routing goes to IORails.
        assert isinstance(guardrails.rails_engine, IORails)
        iorails = guardrails.rails_engine

        main_engine = iorails.engine_registry._get_engine("main", ModelEngine)
        safety_engine = iorails.engine_registry._get_engine("content_safety", ModelEngine)

        # Construction-time normalization: both engines' base_url have /v1 stripped.
        assert main_engine.base_url == "https://main.example.com"
        assert safety_engine.base_url == "https://safety.example.com"

        # Pre-seed each engine with a mock HTTP client + _running=True so
        # BaseEngine.start() short-circuits and no real aiohttp session is created.
        main_client = _make_mock_http_client("Hello from main")
        safety_client = _make_mock_http_client(safe_output_json)
        main_engine._client = main_client
        main_engine._running = True
        safety_engine._client = safety_client
        safety_engine._running = True

        try:
            await guardrails.generate_async(messages=[{"role": "user", "content": "Hi"}])
        finally:
            await iorails.stop()

        # Main model POSTs once to a single-/v1 URL.
        main_url = main_client.post.call_args[0][0]
        assert main_url == "https://main.example.com/v1/chat/completions"
        assert "/v1/v1/" not in main_url

        # Content-safety model POSTs once for the input rail and once for the output rail.
        # Every call must hit a single-/v1 URL.
        assert safety_client.post.call_count == 2
        for call_args in safety_client.post.call_args_list:
            url = call_args[0][0]
            assert url == "https://safety.example.com/v1/chat/completions"
            assert "/v1/v1/" not in url


REWRITE_USER_TEXT = "my ssn is 123-45-6789"
REWRITE_MASKED_USER = "my ssn is <SSN>"
REWRITE_MESSAGES = [{"role": "user", "content": REWRITE_USER_TEXT}]


@pytest.mark.asyncio
class TestGenerateWithRewritingRails:
    """A rail's rewrite reaches the model and the caller, rather than being computed and dropped."""

    async def test_an_input_rewrite_is_what_the_model_reads(self, iorails):
        """Masking that never reaches the model would leave the model reading the raw text."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(REWRITE_MASKED_USER))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))

        await iorails.generate_async(messages=REWRITE_MESSAGES)

        sent_messages = iorails.engine_registry.model_call.call_args.args[1]
        assert sent_messages[-1]["content"] == REWRITE_MASKED_USER

    async def test_an_input_rewrite_leaves_the_callers_messages_untouched(self, iorails):
        """The caller's conversation is theirs; masking it as a side effect would be a surprise."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(REWRITE_MASKED_USER))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))
        messages = [{"role": "user", "content": REWRITE_USER_TEXT}]

        await iorails.generate_async(messages=messages)

        assert messages == [{"role": "user", "content": REWRITE_USER_TEXT}]

    async def test_an_output_rewrite_is_what_the_caller_receives(self, iorails):
        """The whole point of an output rewrite: the caller reads the sanitized response."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=bot_message_rewrite("call me on <PHONE>"))
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="call me on 555-0100"))

        result = await iorails.generate_async(messages=REWRITE_MESSAGES)

        assert result == {"role": "assistant", "content": "call me on <PHONE>"}

    async def test_an_output_rewrite_reaches_the_structured_response(self, iorails):
        """The options path returns the same text as the bare one, not the pre-rewrite response."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=bot_message_rewrite("call me on <PHONE>"))
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="call me on 555-0100"))

        result = await iorails.generate_async(messages=REWRITE_MESSAGES, options={"log": {"activated_rails": True}})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": "call me on <PHONE>"}]


class TestRewriteReaders:
    """The two readers that turn a rail verdict into the text a direction should use."""

    def test_a_user_rewrite_is_read_by_the_input_reader(self):
        """The text an input rail produced, which the model and the later rails must read."""
        assert _rewritten_user_message(user_message_rewrite(REWRITE_MASKED_USER)) == REWRITE_MASKED_USER

    def test_a_bot_rewrite_is_read_by_the_output_reader(self):
        """The text an output rail produced, which is what the caller receives."""
        assert _rewritten_bot_message(bot_message_rewrite("call me on <PHONE>")) == "call me on <PHONE>"

    @pytest.mark.parametrize(
        "result",
        [RailResult.allow(), RailResult.block(reason="unsafe")],
        ids=["allow", "block"],
    )
    def test_a_verdict_without_a_rewrite_reads_as_nothing_to_apply(self, result):
        """Most requests to a rewriting rail come back as a plain verdict, and must change nothing."""
        assert _rewritten_user_message(result) is None
        assert _rewritten_bot_message(result) is None

    def test_each_reader_ignores_the_other_directions_rewrite(self):
        """The readers are target-specific, so a rewrite cannot be applied to the wrong variable."""
        assert _rewritten_user_message(bot_message_rewrite("call me on <PHONE>")) is None
        assert _rewritten_bot_message(user_message_rewrite(REWRITE_MASKED_USER)) is None

    def test_a_rewrite_to_empty_text_is_still_a_rewrite(self):
        """Redacting to nothing must not read as "no rewrite" the way an empty string would."""
        assert _rewritten_bot_message(bot_message_rewrite("")) == ""


@pytest.mark.asyncio
class TestGeneratedTurn:
    """``_do_generate_sequential`` reports the response together with the messages behind it."""

    async def test_a_rewritten_turn_carries_the_messages_the_model_read(self, iorails):
        """The caller of this helper runs the output rails, and must not re-read the stale list."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(REWRITE_MASKED_USER))
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))

        turn = await iorails._do_generate_sequential(REWRITE_MESSAGES, "req-1", {})

        assert turn.messages[-1]["content"] == REWRITE_MASKED_USER
        assert turn.response is not None

    async def test_an_unrewritten_turn_carries_the_messages_as_they_arrived(self, iorails):
        """Nothing is copied or rebuilt for the ordinary case where no rail rewrites."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))

        turn = await iorails._do_generate_sequential(REWRITE_MESSAGES, "req-1", {})

        assert turn.messages is REWRITE_MESSAGES

    async def test_a_blocked_turn_reports_no_response(self, iorails):
        """The block is signalled by the absent response, which is what the caller branches on."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.block(reason="unsafe"))
        iorails.engine_registry.model_call = AsyncMock()

        turn = await iorails._do_generate_sequential(REWRITE_MESSAGES, "req-1", {})

        assert turn.response is None
        assert turn.messages is REWRITE_MESSAGES
        iorails.engine_registry.model_call.assert_not_called()

    async def test_a_blocked_turn_carries_the_verdict_that_blocked_it(self, iorails):
        """The caller renders the message from this verdict, so a failed rail must survive the hop."""
        verdict = rail_failure("f5 guardrails scan input")
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=verdict)
        iorails.engine_registry.model_call = AsyncMock()

        turn = await iorails._do_generate_sequential(REWRITE_MESSAGES, "req-1", {})

        assert turn.blocked_by is verdict

    async def test_an_unblocked_turn_names_no_blocking_verdict(self, iorails):
        """Nothing blocked, so there is no verdict for the caller to render a message from."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))

        turn = await iorails._do_generate_sequential(REWRITE_MESSAGES, "req-1", {})

        assert turn.blocked_by is None


@pytest.mark.asyncio
class TestContentCaptureRecordsWhatTheModelRead:
    """Capture records the masked text, so a span cannot carry what a mask removed."""

    async def test_the_request_span_carries_the_masked_input(self, iorails):
        """A trace is a downstream system, and masking exists to keep the raw text out of them."""
        iorails._content_capture_enabled = True
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(REWRITE_MASKED_USER))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))

        with patch("nemoguardrails.guardrails.iorails.set_request_content") as capture:
            await iorails.generate_async(messages=REWRITE_MESSAGES)

        assert capture.call_args.args[1][-1]["content"] == REWRITE_MASKED_USER

    async def test_an_unmasked_turn_is_captured_as_it_arrived(self, iorails):
        """The ordinary request is unchanged: what the caller sent is what the span records."""
        iorails._content_capture_enabled = True
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult.allow())
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult.allow())
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="sure"))

        with patch("nemoguardrails.guardrails.iorails.set_request_content") as capture:
            await iorails.generate_async(messages=REWRITE_MESSAGES)

        assert capture.call_args.args[1] == REWRITE_MESSAGES
