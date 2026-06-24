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

"""Integration tests for the tool rails wired into IORails.

These drive the full top-level request path (``generate_async`` / ``stream_async``)
with only the aiohttp transport mocked, so the real ModelEngine parsing
(``parse_tools`` / ``extract_tool_results`` / ``extract_tool_calls``) and the
RailsManager tool rails run end to end. They are the IORails-level companions to
``test_tool_rails_e2e.py`` (which stops at the RailsManager surface).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nemoguardrails import Guardrails
from nemoguardrails.guardrails.guardrails_types import RailDirection
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.tool_helpers import (
    WEATHER_SCHEMA,
    make_tool_conversation,
    malformed_prior_tool_call_messages,
    multi_turn_reused_call_id_messages,
)

BASE_CONFIG = {"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}

TOOL_CONFIG = {
    **BASE_CONFIG,
    "rails": {
        "tool_output": {"flows": ["tool call validation"]},
        "tool_input": {"flows": ["tool result validation"]},
    },
}

SPECULATIVE_TOOL_CONFIG = {
    **BASE_CONFIG,
    "rails": {
        "input": {"speculative_generation": True},
        "tool_output": {"flows": ["tool call validation"]},
        "tool_input": {"flows": ["tool result validation"]},
    },
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": WEATHER_SCHEMA,
    },
}
LLM_PARAMS = {"tools": [WEATHER_TOOL], "tool_choice": "auto"}
MESSAGES = [{"role": "user", "content": "What's the weather in Paris?"}]

# Tools declared statically on the model (models[].parameters.tools) rather than per-request
# via options.llm_params. The engine merges these config-level defaults into the toolset the
# tool-call rail validates against, so a config-declared tool must be honored even when the
# request carries no llm_params.
CONFIG_TOOLS_CONFIG = {
    "models": [
        {
            "type": "main",
            "engine": "nim",
            "model": "meta/llama-3.3-70b-instruct",
            "parameters": {"tools": [WEATHER_TOOL]},
        }
    ],
    "rails": {
        "tool_output": {"flows": ["tool call validation"]},
        "tool_input": {"flows": ["tool result validation"]},
    },
}


def _config(rails: dict) -> RailsConfig:
    """Build a RailsConfig with the given ``rails`` block, under a stubbed API key."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        return RailsConfig.from_content(config={**BASE_CONFIG, "rails": rails})


def _inject_json_response(iorails: IORails, payload: dict) -> None:
    """Wire the 'main' engine's aiohttp client to return *payload* as JSON."""
    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=payload)
    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.closed = False
    engine = iorails.engine_registry._get_engine("main", ModelEngine)
    engine._client = mock_client
    engine._running = True


def _inject_sse_stream(iorails: IORails, raw_lines: list) -> None:
    """Wire the 'main' engine's aiohttp client to stream *raw_lines* via readline()."""
    all_lines = []
    for raw in raw_lines:
        for part in raw.split(b"\n"):
            if part:
                all_lines.append(part + b"\n")
    line_iter = iter(all_lines)

    async def _readline():
        return next(line_iter, b"")

    mock_content = MagicMock()
    mock_content.readline = _readline

    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.content = mock_content

    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.closed = False
    engine = iorails.engine_registry._get_engine("main", ModelEngine)
    engine._client = mock_client
    engine._running = True


def _inject_forbidden_transport(iorails: IORails) -> MagicMock:
    """Wire the 'main' engine with a transport whose ``post`` must never be called.

    Returns the mock ``post`` so the test can assert it stayed untouched. Any actual
    call raises immediately, so a regression that lets a blocked request reach the
    provider fails loudly here instead of hitting the live network.
    """
    mock_client = AsyncMock()
    mock_client.post = MagicMock(side_effect=AssertionError("provider must not be called when the request is blocked"))
    mock_client.closed = False
    engine = iorails.engine_registry._get_engine("main", ModelEngine)
    engine._client = mock_client
    engine._running = True
    return mock_client.post


def _tool_call_payload(name: str, arguments_json: str, call_id: str = "call_1") -> dict:
    """A non-streaming /v1/chat/completions response carrying a single tool call."""
    return {
        "id": "chatcmpl-1",
        "model": "meta/llama-3.3-70b-instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments_json}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _text_payload(text: str) -> dict:
    """A non-streaming /v1/chat/completions response carrying plain assistant text."""
    return {
        "id": "chatcmpl-1",
        "model": "meta/llama-3.3-70b-instruct",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def _sse(chunk: dict) -> bytes:
    return ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")


def _tool_call_sse_lines(name: str, arg_fragments: list, call_id: str = "call_1") -> list:
    """SSE lines streaming a single tool call: id/name first, then argument fragments, then finish."""
    chunks = [
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {"index": 0, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    ]
    for fragment in arg_fragments:
        chunks.append(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    chunks.append({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    lines = [_sse(chunk) for chunk in chunks]
    lines.append(b"data: [DONE]\n\n")
    return lines


async def _collect(stream) -> list:
    return [chunk async for chunk in stream]


def _stream_violation_chunks(chunks: list) -> list:
    """The guardrails_violation error payloads among streamed string chunks."""
    violations = []
    for chunk in chunks:
        if isinstance(chunk, str) and '"error"' in chunk:
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("error", {}).get("type") == "guardrails_violation":
                violations.append(parsed)
    return violations


@pytest_asyncio.fixture
async def iorails():
    """A started IORails with both tool rails enabled (transport mocked per test)."""
    async with started_iorails(TOOL_CONFIG) as engine:
        yield engine


class TestRouting:
    """A config with tool rails routes to IORails; misconfigured tool flows fall back."""

    def test_tool_rails_config_can_be_handled(self):
        assert IORails.can_handle(_config(TOOL_CONFIG["rails"])) is True

    def test_guardrails_dispatches_tool_rails_to_iorails(self):
        guardrails = Guardrails(_config(TOOL_CONFIG["rails"]))
        assert guardrails.use_iorails_engine is True
        assert isinstance(guardrails.rails_engine, IORails)

    def test_unknown_tool_output_flow_falls_back(self):
        reason = IORails.unsupported_reason(_config({"tool_output": {"flows": ["make a sandwich"]}}))
        assert reason is not None
        assert "tool output" in reason

    def test_misdirected_flow_falls_back(self):
        # The tool-result validator under tool_output is the wrong direction.
        reason = IORails.unsupported_reason(_config({"tool_output": {"flows": ["tool result validation"]}}))
        assert reason is not None
        assert "tool output" in reason

    def test_duplicate_tool_flow_falls_back(self):
        reason = IORails.unsupported_reason(
            _config({"tool_output": {"flows": ["tool call validation", "tool call validation"]}})
        )
        assert reason is not None
        assert "duplicate" in reason

    def test_normalized_duplicate_tool_flow_falls_back(self):
        # Two entries differing only by a `$model=` suffix normalize to the same tool
        # flow (the tool rails ignore the suffix); they must be caught as a duplicate
        # rather than slipping past and running twice.
        reason = IORails.unsupported_reason(
            _config({"tool_output": {"flows": ["tool call validation", "tool call validation $model=x"]}})
        )
        assert reason is not None
        assert "duplicate" in reason

    def test_iorails_construction_raises_on_duplicate_flow(self):
        # Why unsupported_reason pre-validates duplicates: IORails.__init__ itself would
        # raise on a dup flow, so without the pre-check Guardrails could not fall back.
        with pytest.raises(RuntimeError):
            IORails(_config({"tool_output": {"flows": ["tool call validation", "tool call validation"]}}))

    def test_tool_parallel_flag_warns_inert(self):
        # tool_*.parallel is inert (tool rails run sequentially); construction warns
        # rather than silently ignoring it.
        config = _config({"tool_output": {"flows": ["tool call validation"], "parallel": True}})
        with pytest.warns(UserWarning, match="not honored by IORails"):
            IORails(config)


class TestNonStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_allowed_tool_call_passes(self, iorails):
        _inject_json_response(iorails, _tool_call_payload("get_weather", '{"city": "Paris"}'))
        result = await iorails.generate_async(MESSAGES, options={"llm_params": LLM_PARAMS})
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_undeclared_tool_call_blocked(self, iorails):
        _inject_json_response(iorails, _tool_call_payload("rm_rf", "{}"))
        result = await iorails.generate_async(MESSAGES, options={"llm_params": LLM_PARAMS})
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_invalid_arguments_blocked(self, iorails):
        # Missing the required "city" argument violates the declared schema.
        _inject_json_response(iorails, _tool_call_payload("get_weather", "{}"))
        result = await iorails.generate_async(MESSAGES, options={"llm_params": LLM_PARAMS})
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}


class TestSpeculativeToolCalls:
    """The tool-call rail runs after generation on the speculative path too."""

    @pytest_asyncio.fixture
    async def speculative_iorails(self):
        async with started_iorails(SPECULATIVE_TOOL_CONFIG) as engine:
            yield engine

    @pytest.mark.asyncio
    async def test_undeclared_tool_call_blocked(self, speculative_iorails):
        _inject_json_response(speculative_iorails, _tool_call_payload("rm_rf", "{}"))
        result = await speculative_iorails.generate_async(MESSAGES, options={"llm_params": LLM_PARAMS})
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_allowed_tool_call_passes(self, speculative_iorails):
        _inject_json_response(speculative_iorails, _tool_call_payload("get_weather", '{"city": "Paris"}'))
        result = await speculative_iorails.generate_async(MESSAGES, options={"llm_params": LLM_PARAMS})
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_input_toggle_forwarded_to_input_rails(self, speculative_iorails):
        """On the speculative path, options.rails.input is forwarded to is_input_safe as the enabled argument."""
        spy = AsyncMock(wraps=speculative_iorails.rails_manager.is_input_safe)
        speculative_iorails.rails_manager.is_input_safe = spy
        _inject_json_response(speculative_iorails, _text_payload("ok"))
        await speculative_iorails.generate_async(MESSAGES, options={"rails": {"input": False}})
        assert spy.await_args.kwargs.get("enabled") is False


class TestConfigDeclaredTools:
    """Tools declared on the model (models[].parameters.tools) are validated by the tool-call
    rail end to end, even when the request carries no options.llm_params.tools."""

    @pytest_asyncio.fixture
    async def iorails_config_tools(self):
        async with started_iorails(CONFIG_TOOLS_CONFIG) as engine:
            yield engine

    @pytest.mark.asyncio
    async def test_config_declared_tool_call_passes(self, iorails_config_tools):
        """A call to a config-declared tool passes the rail when the request carries no llm_params."""
        _inject_json_response(iorails_config_tools, _tool_call_payload("get_weather", '{"city": "Paris"}'))
        result = await iorails_config_tools.generate_async(MESSAGES)
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_undeclared_tool_call_blocked_against_config_tools(self, iorails_config_tools):
        """A call to a tool absent from the config toolset is blocked when the request carries no llm_params."""
        _inject_json_response(iorails_config_tools, _tool_call_payload("rm_rf", "{}"))
        result = await iorails_config_tools.generate_async(MESSAGES)
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_streamed_undeclared_tool_call_blocked_against_config_tools(self, iorails_config_tools):
        """In streaming, a call to a tool absent from the config toolset is blocked with no llm_params."""
        _inject_sse_stream(iorails_config_tools, _tool_call_sse_lines("rm_rf", ["{}"]))
        chunks = await _collect(iorails_config_tools.stream_async(MESSAGES))
        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_output_rails"


class TestNonStreamingToolResults:
    @pytest.mark.asyncio
    async def test_linked_tool_result_passes(self, iorails):
        _inject_json_response(iorails, _text_payload("It is sunny."))
        result = await iorails.generate_async(make_tool_conversation(result_call_id="call_1"))
        assert result == {"role": "assistant", "content": "It is sunny."}

    @pytest.mark.asyncio
    async def test_unlinked_tool_result_blocked_before_generation(self, iorails):
        # A blocked tool result must short-circuit before the model is ever called:
        # wire a transport that fails if used, and assert it stayed untouched.
        forbidden_post = _inject_forbidden_transport(iorails)
        result = await iorails.generate_async(make_tool_conversation(result_call_id="call_999"))
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        forbidden_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_recycled_call_ids_across_turns_not_blocked(self, iorails):
        """Multi-turn conversation reusing the same call_id across turns is valid"""

        _inject_json_response(iorails, _text_payload("It's 12C in London."))
        result = await iorails.generate_async(multi_turn_reused_call_id_messages())
        assert result == {"role": "assistant", "content": "It's 12C in London."}

    @pytest.mark.asyncio
    async def test_malformed_prior_tool_call_does_not_block(self, iorails):
        """Malformed tool-call in prior turns must not block request."""
        _inject_json_response(iorails, _text_payload("It's 12C in London."))
        result = await iorails.generate_async(malformed_prior_tool_call_messages())
        assert result == {"role": "assistant", "content": "It's 12C in London."}


class TestStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_allowed_streamed_tool_call_passes(self, iorails):
        _inject_sse_stream(iorails, _tool_call_sse_lines("get_weather", ['{"city": ', '"Paris"}']))
        chunks = await _collect(iorails.stream_async(MESSAGES, options={"llm_params": LLM_PARAMS}))

        assert _stream_violation_chunks(chunks) == []
        terminal = json.loads(chunks[-1])
        assert terminal["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_undeclared_streamed_tool_call_blocked(self, iorails):
        _inject_sse_stream(iorails, _tool_call_sse_lines("rm_rf", ["{}"]))
        chunks = await _collect(iorails.stream_async(MESSAGES, options={"llm_params": LLM_PARAMS}))

        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_output_rails"
        # The tool-call chunk is suppressed: no chunk carries the tool call.
        assert not any(isinstance(c, str) and '"tool_calls"' in c for c in chunks)

    @pytest.mark.asyncio
    async def test_truncated_streamed_tool_call_fails_closed(self, iorails):
        # Truncated tool-call args must fail closed (parity with non-streaming): no tool-call
        # chunk is surfaced with silently-empty args; a generation error is emitted instead.
        _inject_sse_stream(iorails, _tool_call_sse_lines("get_weather", ['{"city": "Par']))
        chunks = await _collect(iorails.stream_async(MESSAGES, options={"llm_params": LLM_PARAMS}))

        assert not any(isinstance(c, str) and '"tool_calls"' in c for c in chunks)
        error_chunks = [c for c in chunks if isinstance(c, str) and '"error"' in c]
        assert error_chunks, f"expected a generation error chunk, got {chunks}"
        assert json.loads(error_chunks[0])["error"]["code"] == "generation_failed"


class TestStreamingToolResults:
    @pytest.mark.asyncio
    async def test_unlinked_tool_result_blocks_stream(self, iorails):
        """A guardrails_violation error chunk (param=tool_input_rails) is emitted, and the model is never called."""
        forbidden_post = _inject_forbidden_transport(iorails)
        chunks = await _collect(iorails.stream_async(make_tool_conversation(result_call_id="call_999")))
        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_input_rails"
        assert REFUSAL_MESSAGE not in chunks
        forbidden_post.assert_not_called()


class TestPerRequestToggles:
    @pytest.mark.asyncio
    async def test_tool_output_disabled_skips_tool_call_rail(self, iorails):
        # An undeclared tool call would normally block; disabling tool_output lets it through.
        _inject_json_response(iorails, _tool_call_payload("rm_rf", "{}"))
        options = {"llm_params": LLM_PARAMS, "rails": {"tool_output": False}}
        result = await iorails.generate_async(MESSAGES, options=options)
        assert result["tool_calls"][0]["function"]["name"] == "rm_rf"

    @pytest.mark.asyncio
    async def test_tool_input_disabled_skips_tool_result_rail(self, iorails):
        # An unlinked tool result would normally block; disabling tool_input lets it through.
        _inject_json_response(iorails, _text_payload("ok"))
        result = await iorails.generate_async(
            make_tool_conversation(result_call_id="call_999"),
            options={"rails": {"tool_input": False}},
        )
        assert result == {"role": "assistant", "content": "ok"}

    @pytest.mark.asyncio
    async def test_input_toggle_forwarded_to_input_rails(self, iorails):
        """options.rails.input is forwarded to is_input_safe as the enabled argument."""
        spy = AsyncMock(wraps=iorails.rails_manager.is_input_safe)
        iorails.rails_manager.is_input_safe = spy
        _inject_json_response(iorails, _text_payload("ok"))
        await iorails.generate_async(MESSAGES, options={"rails": {"input": False}})
        assert spy.await_args.kwargs.get("enabled") is False

    @pytest.mark.asyncio
    async def test_output_toggle_forwarded_to_output_rails(self, iorails):
        """options.rails.output is forwarded to is_output_safe as the enabled argument."""
        spy = AsyncMock(wraps=iorails.rails_manager.is_output_safe)
        iorails.rails_manager.is_output_safe = spy
        _inject_json_response(iorails, _text_payload("ok"))
        await iorails.generate_async(MESSAGES, options={"rails": {"output": False}})
        assert spy.await_args.kwargs.get("enabled") is False

    @pytest.mark.asyncio
    async def test_streamed_input_toggle_forwarded_to_input_rails(self, iorails):
        """In streaming, options.rails.input is forwarded to is_input_safe as the enabled argument."""
        spy = AsyncMock(wraps=iorails.rails_manager.is_input_safe)
        iorails.rails_manager.is_input_safe = spy
        text_lines = [
            _sse(
                {
                    "id": "c1",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
                }
            ),
            _sse({"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ]
        _inject_sse_stream(iorails, text_lines)
        await _collect(iorails.stream_async(MESSAGES, options={"rails": {"input": False}}))
        assert spy.await_args.kwargs.get("enabled") is False


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_duplicate_tool_definitions_block(self, iorails):
        # parse_tools raises on a duplicate tool name; the request must fail closed.
        _inject_json_response(iorails, _tool_call_payload("get_weather", '{"city": "Paris"}'))
        dup_params = {"tools": [WEATHER_TOOL, WEATHER_TOOL]}
        result = await iorails.generate_async(MESSAGES, options={"llm_params": dup_params})
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}


class TestToolRailBlockMetrics:
    """When metrics are enabled, a tool-rail block records the directional block metric."""

    @pytest.mark.asyncio
    async def test_nonstream_tool_result_block_records_input_metric(self, iorails):
        iorails._metrics_enabled = True
        with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_blocked:
            result = await iorails.generate_async(make_tool_conversation(result_call_id="call_999"))
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_blocked.assert_called_once_with(RailDirection.INPUT)

    @pytest.mark.asyncio
    async def test_nonstream_tool_call_block_records_output_metric(self, iorails):
        iorails._metrics_enabled = True
        _inject_json_response(iorails, _tool_call_payload("rm_rf", "{}"))
        with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_blocked:
            result = await iorails.generate_async(MESSAGES, options={"llm_params": LLM_PARAMS})
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        record_blocked.assert_called_once_with(RailDirection.OUTPUT)

    @pytest.mark.asyncio
    async def test_streamed_tool_result_block_records_input_metric(self, iorails):
        iorails._metrics_enabled = True
        with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_blocked:
            chunks = await _collect(iorails.stream_async(make_tool_conversation(result_call_id="call_999")))
        assert _stream_violation_chunks(chunks)
        record_blocked.assert_called_once_with(RailDirection.INPUT)

    @pytest.mark.asyncio
    async def test_streamed_tool_call_block_records_metric_and_captures(self, iorails):
        # Enable both metrics and content capture so the block records the OUTPUT metric
        # and appends the violation payload to the captured output.
        iorails._metrics_enabled = True
        iorails._content_capture_enabled = True
        _inject_sse_stream(iorails, _tool_call_sse_lines("rm_rf", ["{}"]))
        with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_blocked:
            chunks = await _collect(iorails.stream_async(MESSAGES, options={"llm_params": LLM_PARAMS}))
        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_output_rails"
        record_blocked.assert_called_once_with(RailDirection.OUTPUT)
