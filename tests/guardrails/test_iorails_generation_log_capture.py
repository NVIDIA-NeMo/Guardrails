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

"""Rail-call capture through the real CompiledRail pipeline, surfaced as GenerationLog entries."""

import json
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.types import LLMResponse, UsageInfo
from tests.guardrails.async_helpers import mock_rail_model, started_iorails
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG

_USER = [{"role": "user", "content": "hi"}]
_CS_INPUT = "content safety check input $model=content_safety"
_CS_MODEL = "nvidia/llama-3.1-nemoguard-8b-content-safety"
_MAIN_MODEL = "meta/llama-3.3-70b-instruct"

# {"User Safety": "safe", "Response Safety": "safe"} parses safe for both the input
# parser (reads User Safety) and the output parser (reads Response Safety).
_SAFE_BOTH = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
_UNSAFE_INPUT = json.dumps({"User Safety": "unsafe", "Safety Categories": "S1: Violence"})


def _cs_and_main_model_call(*, cs_request_id=None, main_request_id=None):
    """Build a ``model_call`` side_effect: content-safety returns _SAFE_BOTH, the main model returns "Hi"."""

    async def _model_call(model_type, messages, **kwargs):
        if model_type == "content_safety":
            return LLMResponse(
                content=_SAFE_BOTH,
                usage=UsageInfo(input_tokens=100, output_tokens=10, total_tokens=110),
                model=_CS_MODEL,
                request_id=cs_request_id,
            )
        return LLMResponse(
            content="Hi",
            usage=UsageInfo(input_tokens=20, output_tokens=5, total_tokens=25),
            model=_MAIN_MODEL,
            request_id=main_request_id,
        )

    return _model_call


def _rail_chat_completion(content=_SAFE_BOTH, request_id="req-cs"):
    """The content-safety engine's transport double, answering the rail's own model call."""
    return AsyncMock(
        return_value=LLMResponse(
            content=content,
            usage=UsageInfo(input_tokens=100, output_tokens=10, total_tokens=110),
            model=_CS_MODEL,
            request_id=request_id,
        )
    )


@pytest_asyncio.fixture
async def iorails():
    """Started IORails on a content-safety-only config (input + output rails)."""
    async with started_iorails(CONTENT_SAFETY_CONFIG) as engine:
        yield engine


class TestRailRecordCapture:
    """A real rail run captures its LLM call + verdict onto RailResult.records."""

    @pytest.mark.asyncio
    async def test_is_input_safe_captures_usage_and_verdict(self, iorails):
        """is_input_safe returns a record carrying the rail's tokens, model, and verdict."""
        mock_rail_model(
            iorails.engine_registry,
            AsyncMock(
                return_value=LLMResponse(
                    content=_SAFE_BOTH,
                    usage=UsageInfo(input_tokens=762, output_tokens=8, total_tokens=770),
                    model=_CS_MODEL,
                    request_id="req-cs-input",
                )
            ),
            model_type="content_safety",
        )

        result = await iorails.rails_manager.is_input_safe(_USER)

        assert result.is_safe is True
        assert len(result.records) == 1
        record = result.records[0]
        assert record.flow == _CS_INPUT
        assert record.rail_type == "input"
        assert record.made_call is True
        assert record.usage.total_tokens == 770
        assert record.llm_model_name == _CS_MODEL
        assert record.llm_provider_name == "nim"
        assert record.request_id == "req-cs-input"
        assert record.return_value == {"allowed": True, "policy_violations": []}

    @pytest.mark.asyncio
    async def test_failed_model_call_still_records_the_attempt(self, iorails):
        """A rail whose model call raises still yields a record naming the model it attempted."""
        mock_rail_model(
            iorails.engine_registry, AsyncMock(side_effect=RuntimeError("provider down")), model_type="content_safety"
        )

        result = await iorails.rails_manager.is_input_safe(_USER)

        assert result.is_safe is False
        assert len(result.records) == 1
        record = result.records[0]
        assert record.made_call is True
        assert record.usage.total_tokens == 0
        assert record.llm_model_name == _CS_MODEL
        assert record.llm_provider_name == "nim"
        assert record.duration is not None


class TestGenerationLogEndToEnd:
    """Full generate_async path builds a GenerationLog from real rail + main-call records."""

    @pytest.mark.asyncio
    async def test_stats_and_activated_rails(self, iorails):
        """log covers input CS + main + output CS calls, with the content-safety verdict."""
        iorails.engine_registry.model_call = AsyncMock(
            side_effect=_cs_and_main_model_call(cs_request_id="req-cs", main_request_id="req-main")
        )
        mock_rail_model(iorails.engine_registry, _rail_chat_completion(), model_type="content_safety")

        result = await iorails.generate_async(
            messages=_USER, options={"log": {"llm_calls": True, "activated_rails": True}}
        )

        assert isinstance(result, GenerationResponse)
        assert result.log is not None
        stats = result.log.stats
        assert stats.llm_calls_count == 3
        assert stats.llm_calls_total_prompt_tokens == 100 + 20 + 100
        assert stats.llm_calls_total_completion_tokens == 10 + 5 + 10
        assert stats.llm_calls_total_tokens == 110 + 25 + 110

        cs_rail = next(rail for rail in result.log.activated_rails if rail.name == _CS_INPUT)
        assert cs_rail.type == "input"
        assert cs_rail.executed_actions[0].return_value == {"allowed": True, "policy_violations": []}
        # The provider's response id is threaded through as request_id.
        cs_call = cs_rail.executed_actions[0].llm_calls[0]
        assert cs_call.request_id == "req-cs"
        assert cs_call.llm_provider_name == "nim"
        assert any(rail.type == "generation" for rail in result.log.activated_rails)

    @pytest.mark.asyncio
    async def test_llm_calls_capture_prompt_and_completion(self, iorails):
        """Each LLM-backed call in the log carries its serialized prompt and raw completion."""
        iorails.engine_registry.model_call = AsyncMock(side_effect=_cs_and_main_model_call())
        mock_rail_model(iorails.engine_registry, _rail_chat_completion(), model_type="content_safety")

        result = await iorails.generate_async(messages=_USER, options={"log": {"llm_calls": True}})

        assert isinstance(result, GenerationResponse)
        assert result.log is not None
        llm_calls = result.log.llm_calls or []
        cs_call = next(c for c in llm_calls if c.task and "content_safety_check_input" in c.task)
        main_call = next(c for c in llm_calls if c.task == "general")

        assert cs_call.completion == _SAFE_BOTH
        assert cs_call.prompt is not None
        assert "hi" in cs_call.prompt
        assert main_call.completion == "Hi"
        assert main_call.prompt is not None
        assert "hi" in main_call.prompt

    @pytest.mark.asyncio
    async def test_prompt_serializes_role_and_content(self, iorails):
        """The serialized prompt is role-labeled so a reader can tell system from user turns."""
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="Hi", usage=UsageInfo(input_tokens=20, output_tokens=5, total_tokens=25))
        )
        mock_rail_model(iorails.engine_registry, _rail_chat_completion(), model_type="content_safety")

        result = await iorails.generate_async(messages=_USER, options={"log": {"llm_calls": True}})

        assert result.log is not None
        llm_calls = result.log.llm_calls or []
        cs_call = next(c for c in llm_calls if c.task and "content_safety_check_input" in c.task)
        assert cs_call.prompt is not None
        assert "user:" in cs_call.prompt or "system:" in cs_call.prompt

    @pytest.mark.asyncio
    async def test_blocked_input_logs_verdict_and_stop(self, iorails):
        """A blocked input rail logs its unsafe verdict + stop, and the request refuses."""
        mock_rail_model(
            iorails.engine_registry, _rail_chat_completion(content=_UNSAFE_INPUT), model_type="content_safety"
        )

        result = await iorails.generate_async(messages=_USER, options={"log": {"activated_rails": True}})

        assert isinstance(result, GenerationResponse)
        assert result.response == [{"role": "assistant", "content": REFUSAL_MESSAGE}]
        assert result.log is not None
        cs_rail = next(rail for rail in result.log.activated_rails if rail.name == _CS_INPUT)
        assert cs_rail.stop is True
        verdict = cs_rail.executed_actions[0].return_value
        assert verdict["allowed"] is False
        assert len(verdict["policy_violations"]) >= 1
