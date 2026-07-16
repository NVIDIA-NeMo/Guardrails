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

"""GenerationLog + token usage for IORails (PR B), matching LLMRails' log contract.

When ``options.log`` requests ``activated_rails`` and/or ``llm_calls``, IORails
synthesizes a ``GenerationLog`` from the per-rail ``RailCallRecord``s carried on each
``RailResult`` plus the main generation call: ``activated_rails`` (one synthetic
``ExecutedAction`` per rail carrying the real verdict as ``return_value``), a flat
``llm_calls`` list (main + every rail), and aggregate ``stats``. ``internal_events``
and ``colang_history`` are Colang-runtime-only and raise ``NotImplementedError``.
Token usage lives only in ``log`` — it is no longer surfaced under ``llm_metadata``.
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailCallRecord, RailResult
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.types import LLMResponse, UsageInfo
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

_USER = [{"role": "user", "content": "hi"}]

_CS_INPUT_FLOW = "content safety check input $model=content_safety"
_CS_OUTPUT_FLOW = "content safety check output $model=content_safety"


@pytest_asyncio.fixture
async def iorails():
    """Started IORails instance with worker-queue teardown after each test."""
    async with started_iorails(NEMOGUARDS_CONFIG) as iorails:
        yield iorails


def _input_record(*, is_safe: bool = True, return_value=None) -> RailCallRecord:
    """A content-safety input-rail record with a NeMoGuard-style verdict + usage."""
    return RailCallRecord(
        flow=_CS_INPUT_FLOW,
        rail_type="input",
        is_safe=is_safe,
        action_name="content_safety_check_input",
        return_value=return_value if return_value is not None else {"allowed": is_safe, "policy_violations": []},
        task="content_safety_check_input $model=content_safety",
        usage=UsageInfo(input_tokens=762, output_tokens=8, total_tokens=770),
        llm_model_name="nvidia/llama-3.1-nemoguard-8b-content-safety",
        llm_provider_name="nim",
        started_at=1.0,
        finished_at=1.5,
        duration=0.5,
    )


def _output_record() -> RailCallRecord:
    """A content-safety output-rail record with usage."""
    return RailCallRecord(
        flow=_CS_OUTPUT_FLOW,
        rail_type="output",
        is_safe=True,
        action_name="content_safety_check_output",
        return_value={"allowed": True, "policy_violations": []},
        task="content_safety_check_output $model=content_safety",
        usage=UsageInfo(input_tokens=855, output_tokens=15, total_tokens=870),
        llm_model_name="nvidia/llama-3.1-nemoguard-8b-content-safety",
        llm_provider_name="nim",
        started_at=3.0,
        finished_at=3.4,
        duration=0.4,
    )


def _stub_pipeline(iorails: IORails, *, input_records=(), output_records=(), input_safe=True) -> None:
    """Stub input/output rails to return given records, and a main call with usage."""
    iorails.rails_manager.is_input_safe = AsyncMock(
        return_value=RailResult(
            is_safe=input_safe,
            reason=None if input_safe else "unsafe",
            triggered_rail=None if input_safe else _CS_INPUT_FLOW,
            records=tuple(input_records),
        )
    )
    iorails.rails_manager.is_output_safe = AsyncMock(
        return_value=RailResult(is_safe=True, records=tuple(output_records))
    )
    iorails.engine_registry.model_call = AsyncMock(
        return_value=LLMResponse(content="Hi", usage=UsageInfo(input_tokens=99, output_tokens=50, total_tokens=149))
    )


class TestLogGating:
    """`log` is only built when explicitly requested."""

    @pytest.mark.asyncio
    async def test_log_none_when_not_requested(self, iorails):
        """With options but no log flags, `res.log` stays None."""
        _stub_pipeline(iorails, input_records=[_input_record()])

        result = await iorails.generate_async(_USER, options={})

        assert isinstance(result, GenerationResponse)
        assert result.log is None


class TestLlmCallsAndStats:
    """Flat `llm_calls` list and aggregate `stats` cover the main call plus every rail."""

    @pytest.mark.asyncio
    async def test_llm_calls_and_stats_aggregate_main_and_rails(self, iorails):
        """`log.llm_calls` lists main + each rail call; `log.stats` sums their tokens."""
        _stub_pipeline(iorails, input_records=[_input_record()], output_records=[_output_record()])

        result = await iorails.generate_async(_USER, options={"log": {"llm_calls": True}})

        assert result.log is not None
        stats = result.log.stats
        assert stats.llm_calls_count == 3
        assert stats.llm_calls_total_prompt_tokens == 762 + 855 + 99
        assert stats.llm_calls_total_completion_tokens == 8 + 15 + 50
        assert stats.llm_calls_total_tokens == 770 + 870 + 149

        totals = sorted(call.total_tokens for call in result.log.llm_calls)
        assert totals == [149, 770, 870]


class TestActivatedRails:
    """`activated_rails` carries one synthetic action per rail with the real verdict."""

    @pytest.mark.asyncio
    async def test_activated_rail_carries_verdict_as_return_value(self, iorails):
        """The rail's structured verdict is preserved as executed_actions[0].return_value."""
        verdict = {"allowed": False, "policy_violations": ["Violence", "Criminal Planning/Confessions"]}
        _stub_pipeline(iorails, input_records=[_input_record(is_safe=False, return_value=verdict)], input_safe=False)

        result = await iorails.generate_async(_USER, options={"log": {"activated_rails": True}})

        assert result.log is not None
        rail = next(r for r in result.log.activated_rails if r.name == _CS_INPUT_FLOW)
        assert rail.type == "input"
        assert rail.executed_actions[0].action_name == "content_safety_check_input"
        assert rail.executed_actions[0].return_value == verdict

    @pytest.mark.asyncio
    async def test_stop_set_on_blocking_rail(self, iorails):
        """A blocking input rail is marked stop=True, and the blocked request still logs it."""
        _stub_pipeline(iorails, input_records=[_input_record(is_safe=False)], input_safe=False)

        result = await iorails.generate_async(_USER, options={"log": {"activated_rails": True}})

        assert isinstance(result, GenerationResponse)
        assert result.log is not None
        rail = next(r for r in result.log.activated_rails if r.name == _CS_INPUT_FLOW)
        assert rail.stop is True


class TestUnsupportedLogOptions:
    """Colang-runtime-only log fields raise NotImplementedError."""

    @pytest.mark.asyncio
    async def test_internal_events_raises(self, iorails):
        """Requesting internal_events raises — IORails runs no Colang event stream."""
        _stub_pipeline(iorails, input_records=[_input_record()])

        with pytest.raises(NotImplementedError):
            await iorails.generate_async(_USER, options={"log": {"internal_events": True}})

    @pytest.mark.asyncio
    async def test_colang_history_raises(self, iorails):
        """Requesting colang_history raises — IORails produces no Colang transcript."""
        _stub_pipeline(iorails, input_records=[_input_record()])

        with pytest.raises(NotImplementedError):
            await iorails.generate_async(_USER, options={"log": {"colang_history": True}})


class TestUsageRemovedFromLlmMetadata:
    """Token usage moved to `log`; `llm_metadata` is a pure provider_metadata passthrough."""

    @pytest.mark.asyncio
    async def test_llm_metadata_has_no_usage_key(self, iorails):
        """provider_metadata is surfaced verbatim; no `usage` sub-key is added."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(
                content="Hi",
                provider_metadata={"response_headers": {"nvcf-status": "fulfilled"}},
                usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )

        result = await iorails.generate_async(_USER, options={})

        assert result.llm_metadata == {"response_headers": {"nvcf-status": "fulfilled"}}

    @pytest.mark.asyncio
    async def test_llm_metadata_none_when_only_usage(self, iorails):
        """With usage but no provider_metadata, llm_metadata is None (usage no longer graft-in)."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.engine_registry.model_call = AsyncMock(
            return_value=LLMResponse(content="Hi", usage=UsageInfo(input_tokens=10, output_tokens=5, total_tokens=15))
        )

        result = await iorails.generate_async(_USER, options={})

        assert result.llm_metadata is None
