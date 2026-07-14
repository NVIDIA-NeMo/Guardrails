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

"""Unit tests for IORails check / check_async.

Mirrors the LLMRails check contract (tests/test_llmrails_check_async.py) but
drives the IORails direct-rails path, mocking RailsManager.is_input_safe /
is_output_safe to control verdicts.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import (
    REFUSAL_MESSAGE,
    IORails,
    _determine_rails_from_messages,
    _get_last_content_by_role,
)
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

SAFE = RailResult(is_safe=True)


def _unsafe(rail: str) -> RailResult:
    return RailResult(is_safe=False, reason="unsafe", triggered_rail=rail)


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
    engine = IORails(rails_config)
    try:
        yield engine
    finally:
        await engine.stop()


def _mock_rails(engine, *, input_result=SAFE, output_result=SAFE):
    engine.rails_manager.is_input_safe = AsyncMock(return_value=input_result)
    engine.rails_manager.is_output_safe = AsyncMock(return_value=output_result)


class TestCheckAsyncAutoDetect:
    """rail_types=None: which rails run is auto-detected from message roles."""

    @pytest.mark.asyncio
    async def test_input_passed(self, iorails):
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        assert result.rail is None
        iorails.rails_manager.is_input_safe.assert_awaited_once_with(messages)
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_input_blocked(self, iorails):
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))
        messages = [{"role": "user", "content": "bad"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.BLOCKED
        assert result.content == REFUSAL_MESSAGE
        assert result.rail == "content safety check input"

    @pytest.mark.asyncio
    async def test_output_passed(self, iorails):
        _mock_rails(iorails)
        messages = [{"role": "assistant", "content": "hi there"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "hi there"
        assert result.rail is None
        iorails.rails_manager.is_output_safe.assert_awaited_once_with(messages, "hi there")
        iorails.rails_manager.is_input_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_output_blocked(self, iorails):
        _mock_rails(iorails, output_result=_unsafe("content safety check output"))
        messages = [{"role": "assistant", "content": "bad answer"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.BLOCKED
        assert result.content == REFUSAL_MESSAGE
        assert result.rail == "content safety check output"

    @pytest.mark.asyncio
    async def test_both_passed(self, iorails):
        _mock_rails(iorails)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "hi there"
        iorails.rails_manager.is_input_safe.assert_awaited_once()
        iorails.rails_manager.is_output_safe.assert_awaited_once_with(messages, "hi there")

    @pytest.mark.asyncio
    async def test_both_input_blocked_skips_output(self, iorails):
        _mock_rails(iorails, input_result=_unsafe("jailbreak detection model"))
        messages = [
            {"role": "user", "content": "bad"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "jailbreak detection model"
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_both_output_blocked(self, iorails):
        _mock_rails(iorails, output_result=_unsafe("content safety check output"))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "bad answer"},
        ]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check output"

    @pytest.mark.asyncio
    async def test_no_user_or_assistant_returns_passed(self, iorails):
        _mock_rails(iorails)
        messages = [{"role": "system", "content": "Be helpful"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "Be helpful"
        iorails.rails_manager.is_input_safe.assert_not_awaited()
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_messages_returns_passed(self, iorails):
        _mock_rails(iorails)

        result = await iorails.check_async([])

        assert result.status == RailStatus.PASSED
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_system_and_user_runs_input(self, iorails):
        _mock_rails(iorails)
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
        ]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        iorails.rails_manager.is_input_safe.assert_awaited_once()
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_complex_conversation_returns_last_assistant(self, iorails):
        _mock_rails(iorails)
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "fine"},
        ]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "fine"


class TestCheckAsyncExplicitRailTypes:
    """rail_types provided: only the named rail types run, no auto-detection."""

    @pytest.mark.asyncio
    async def test_explicit_input_only(self, iorails):
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_output_only(self, iorails):
        _mock_rails(iorails)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.OUTPUT])

        assert result.status == RailStatus.PASSED
        assert result.content == "hi there"
        iorails.rails_manager.is_input_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_input_blocks(self, iorails):
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))
        messages = [{"role": "user", "content": "bad"}]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check input"

    @pytest.mark.asyncio
    async def test_explicit_output_blocks(self, iorails):
        _mock_rails(iorails, output_result=_unsafe("content safety check output"))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "bad answer"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.OUTPUT])

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check output"

    @pytest.mark.asyncio
    async def test_explicit_input_skips_blocking_output_rail(self, iorails):
        # Output rail would block, but only input is requested -> output not run.
        _mock_rails(iorails, output_result=_unsafe("content safety check output"))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "bad answer"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.PASSED
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_output_skips_blocking_input_rail(self, iorails):
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))
        messages = [
            {"role": "user", "content": "bad"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.OUTPUT])

        assert result.status == RailStatus.PASSED
        assert result.content == "hi there"
        iorails.rails_manager.is_input_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_both(self, iorails):
        _mock_rails(iorails)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT, RailType.OUTPUT])

        assert result.status == RailStatus.PASSED
        iorails.rails_manager.is_input_safe.assert_awaited_once()
        iorails.rails_manager.is_output_safe.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_explicit_both_input_blocked(self, iorails):
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))
        messages = [
            {"role": "user", "content": "bad"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT, RailType.OUTPUT])

        assert result.status == RailStatus.BLOCKED
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_both_output_blocked(self, iorails):
        _mock_rails(iorails, output_result=_unsafe("content safety check output"))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "bad answer"},
        ]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT, RailType.OUTPUT])

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check output"

    @pytest.mark.asyncio
    async def test_explicit_empty_rail_types_runs_nothing(self, iorails):
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages, rail_types=[])

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        iorails.rails_manager.is_input_safe.assert_not_awaited()
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_output_no_assistant_message_passes(self, iorails):
        # rail_types=[OUTPUT] with no assistant content to check must PASS, not false-BLOCK.
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages, rail_types=[RailType.OUTPUT])

        assert result.status == RailStatus.PASSED
        iorails.rails_manager.is_output_safe.assert_not_awaited()


class TestCheckAsyncBlockedResult:
    """Details of the BLOCKED RailsResult."""

    @pytest.mark.asyncio
    async def test_blocked_content_is_refusal_message(self, iorails):
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.content == REFUSAL_MESSAGE

    @pytest.mark.asyncio
    async def test_blocked_without_triggered_rail_has_none(self, iorails):
        # Defensive: a block with no triggered_rail surfaces rail=None, not a crash.
        _mock_rails(iorails, input_result=RailResult(is_safe=False, reason="unsafe"))

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.status == RailStatus.BLOCKED
        assert result.rail is None


class TestCheckSync:
    """Synchronous check() spins up an ephemeral engine via asyncio.run."""

    def test_check_passed(self, iorails_sync):
        _mock_rails(iorails_sync)
        messages = [{"role": "user", "content": "hello"}]

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"

    def test_check_blocked(self, iorails_sync):
        _mock_rails(iorails_sync, input_result=_unsafe("content safety check input"))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check([{"role": "user", "content": "bad"}])

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check input"

    def test_check_with_explicit_rails_skips_output(self, iorails_sync):
        _mock_rails(iorails_sync, output_result=_unsafe("content safety check output"))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "bad answer"},
        ]

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.PASSED
        iorails_sync.rails_manager.is_output_safe.assert_not_awaited()

    def test_check_marks_temp_engine_as_internal(self, iorails_sync):
        _mock_rails(iorails_sync)

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync) as mock_iorails:
            iorails_sync.check([{"role": "user", "content": "hello"}])

        mock_iorails.assert_called_once()
        assert mock_iorails.call_args.kwargs == {"_report_usage": False}
        # The ephemeral engine must be built with tracing and metrics disabled.
        passed_config = mock_iorails.call_args.args[0]
        assert passed_config.tracing is None or not passed_config.tracing.enabled
        assert passed_config.metrics is None or not passed_config.metrics.enabled

    def test_check_raises_when_called_from_async_loop(self, iorails_sync):
        async def call_check():
            iorails_sync.check([{"role": "user", "content": "hi"}])

        with pytest.raises(RuntimeError, match="inside async code"):
            asyncio.run(call_check())


class TestCheckAsyncAutoStart:
    """check_async drives the engine lifecycle like generate_async (full parity)."""

    @pytest.mark.asyncio
    async def test_check_async_calls_start(self, iorails):
        iorails.engine_registry.start = AsyncMock()
        _mock_rails(iorails)

        assert not iorails._running
        await iorails.check_async([{"role": "user", "content": "hi"}])

        iorails.engine_registry.start.assert_called_once()
        assert iorails._running

    @pytest.mark.asyncio
    async def test_check_async_start_is_idempotent(self, iorails):
        iorails.engine_registry.start = AsyncMock()
        _mock_rails(iorails)

        await iorails.check_async([{"role": "user", "content": "hi"}])
        await iorails.check_async([{"role": "user", "content": "hi"}])

        iorails.engine_registry.start.assert_called_once()


class TestCheckAsyncErrors:
    """check_async surfaces rail/engine exceptions instead of swallowing them."""

    @pytest.mark.asyncio
    async def test_check_async_propagates_exception(self, iorails):
        iorails.rails_manager.is_input_safe = AsyncMock(side_effect=RuntimeError("rail boom"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=SAFE)

        with pytest.raises(RuntimeError, match="rail boom"):
            await iorails.check_async([{"role": "user", "content": "hi"}])


class TestCheckHelpers:
    """Direct unit tests for the duplicated message helpers."""

    def test_determine_rails_user_only(self):
        assert _determine_rails_from_messages([{"role": "user", "content": "hi"}]) == {"rails": ["input"]}

    def test_determine_rails_assistant_only(self):
        assert _determine_rails_from_messages([{"role": "assistant", "content": "hi"}]) == {"rails": ["output"]}

    def test_determine_rails_both(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        assert _determine_rails_from_messages(msgs) == {"rails": ["input", "output"]}

    def test_determine_rails_none_when_no_user_or_assistant(self, caplog):
        with caplog.at_level(logging.WARNING, logger="nemoguardrails.guardrails.iorails"):
            assert _determine_rails_from_messages([{"role": "system", "content": "x"}]) is None
        assert "no user or assistant messages" in caplog.text

    def test_get_last_content_by_role_returns_last_match(self):
        msgs = [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
        assert _get_last_content_by_role(msgs, "user") == "second"

    def test_get_last_content_by_role_missing_returns_empty(self):
        assert _get_last_content_by_role([{"role": "system", "content": "x"}], "user") == ""
