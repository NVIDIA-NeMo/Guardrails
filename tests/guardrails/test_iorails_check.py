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
    """Build an unsafe RailResult carrying the given triggered-rail name."""
    return RailResult(is_safe=False, reason="unsafe", triggered_rail=rail)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def rails_config():
    """A RailsConfig with all four Nemoguard input/output rails."""
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails_sync(rails_config):
    """An unstarted IORails engine for synchronous check() tests."""
    return IORails(rails_config)


@pytest_asyncio.fixture
async def iorails(rails_config):
    """A started-on-first-use IORails engine, stopped on teardown."""
    engine = IORails(rails_config)
    try:
        yield engine
    finally:
        await engine.stop()


def _mock_rails(engine, *, input_result=SAFE, output_result=SAFE):
    """Stub the engine's is_input_safe / is_output_safe with fixed verdicts."""
    engine.rails_manager.is_input_safe = AsyncMock(return_value=input_result)
    engine.rails_manager.is_output_safe = AsyncMock(return_value=output_result)


class TestCheckAsyncAutoDetect:
    """rail_types=None: which rails run is auto-detected from message roles."""

    @pytest.mark.asyncio
    async def test_input_passed(self, iorails):
        """User-only messages run only input rails; a safe verdict returns PASSED with the user content."""
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
        """An unsafe input verdict returns BLOCKED with the refusal message and the blocking rail name."""
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))
        messages = [{"role": "user", "content": "bad"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.BLOCKED
        assert result.content == REFUSAL_MESSAGE
        assert result.rail == "content safety check input"

    @pytest.mark.asyncio
    async def test_output_passed(self, iorails):
        """Assistant-only messages run only output rails; a safe verdict returns PASSED with the assistant content."""
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
        """An unsafe output verdict returns BLOCKED with the refusal message and the blocking rail name."""
        _mock_rails(iorails, output_result=_unsafe("content safety check output"))
        messages = [{"role": "assistant", "content": "bad answer"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.BLOCKED
        assert result.content == REFUSAL_MESSAGE
        assert result.rail == "content safety check output"

    @pytest.mark.asyncio
    async def test_both_passed(self, iorails):
        """User+assistant messages run both rails; both-safe returns PASSED with the last assistant content."""
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
        """When input blocks, output rails are not run and the result is BLOCKED by the input rail."""
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
        """When input passes and output blocks, the result is BLOCKED by the output rail."""
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
        """Messages with no user/assistant role run no rails and return PASSED with the last content."""
        _mock_rails(iorails)
        messages = [{"role": "system", "content": "Be helpful"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "Be helpful"
        iorails.rails_manager.is_input_safe.assert_not_awaited()
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_messages_returns_passed(self, iorails):
        """An empty message list returns PASSED with empty content and runs no rails."""
        _mock_rails(iorails)

        result = await iorails.check_async([])

        assert result.status == RailStatus.PASSED
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_assistant_none_content_passes(self, iorails):
        """An assistant tool-call message with content=None returns PASSED with '' content, not a validation error."""
        _mock_rails(iorails)
        messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == ""
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_content_last_message_passes(self, iorails):
        """A trailing message with content=None returns PASSED with '' content instead of a validation error."""
        _mock_rails(iorails)
        messages = [{"role": "tool", "content": None, "tool_call_id": "t1"}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_user_empty_content_passes(self, iorails):
        """A user message with empty content returns PASSED without running input rails, instead of raising."""
        _mock_rails(iorails)
        messages = [{"role": "user", "content": ""}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == ""
        iorails.rails_manager.is_input_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_none_content_passes(self, iorails):
        """A user message with content=None returns PASSED without running input rails, instead of raising."""
        _mock_rails(iorails)
        messages = [{"role": "user", "content": None}]

        result = await iorails.check_async(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == ""
        iorails.rails_manager.is_input_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_system_and_user_runs_input(self, iorails):
        """A system+user conversation runs only input rails."""
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
        """A multi-turn conversation runs both rails and returns PASSED with the last assistant content."""
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
        """rail_types=[INPUT] runs only input rails, even when an assistant message is present."""
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_output_only(self, iorails):
        """rail_types=[OUTPUT] runs only output rails and skips input."""
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
        """rail_types=[INPUT] returns BLOCKED when the input rail is unsafe."""
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))
        messages = [{"role": "user", "content": "bad"}]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check input"

    @pytest.mark.asyncio
    async def test_explicit_output_blocks(self, iorails):
        """rail_types=[OUTPUT] returns BLOCKED when the output rail is unsafe."""
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
        """rail_types=[INPUT] does not run output rails even when they would block."""
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
        """rail_types=[OUTPUT] does not run input rails even when they would block."""
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
        """rail_types=[INPUT, OUTPUT] runs both rails."""
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
        """rail_types=[INPUT, OUTPUT] returns BLOCKED and skips output when input blocks."""
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
        """rail_types=[INPUT, OUTPUT] returns BLOCKED on the output rail when input passes."""
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
        """rail_types=[] runs no rails and returns PASSED with the checked content."""
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages, rail_types=[])

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        iorails.rails_manager.is_input_safe.assert_not_awaited()
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_output_no_assistant_message_passes(self, iorails):
        """rail_types=[OUTPUT] with no assistant content to check returns PASSED, not a false BLOCK."""
        _mock_rails(iorails)
        messages = [{"role": "user", "content": "hello"}]

        result = await iorails.check_async(messages, rail_types=[RailType.OUTPUT])

        assert result.status == RailStatus.PASSED
        iorails.rails_manager.is_output_safe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_input_no_user_message_passes(self, iorails):
        """rail_types=[INPUT] with no user content to check returns PASSED, not a false BLOCK."""
        _mock_rails(iorails)
        messages = [{"role": "assistant", "content": "earlier reply"}]

        result = await iorails.check_async(messages, rail_types=[RailType.INPUT])

        assert result.status == RailStatus.PASSED
        iorails.rails_manager.is_input_safe.assert_not_awaited()


class TestCheckAsyncBlockedResult:
    """Details of the BLOCKED RailsResult."""

    @pytest.mark.asyncio
    async def test_blocked_content_is_refusal_message(self, iorails):
        """A blocked check returns REFUSAL_MESSAGE as the content."""
        _mock_rails(iorails, input_result=_unsafe("content safety check input"))

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.content == REFUSAL_MESSAGE

    @pytest.mark.asyncio
    async def test_blocked_without_triggered_rail_has_none(self, iorails):
        """A block whose RailResult carries no triggered_rail surfaces rail=None rather than crashing."""
        _mock_rails(iorails, input_result=RailResult(is_safe=False, reason="unsafe"))

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.status == RailStatus.BLOCKED
        assert result.rail is None


class TestCheckSync:
    """Synchronous check() spins up an ephemeral engine via asyncio.run."""

    def test_check_passed(self, iorails_sync):
        """Sync check() returns PASSED for a safe input."""
        _mock_rails(iorails_sync)
        messages = [{"role": "user", "content": "hello"}]

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check(messages)

        assert result.status == RailStatus.PASSED
        assert result.content == "hello"

    def test_check_blocked(self, iorails_sync):
        """Sync check() returns BLOCKED with the blocking rail name for an unsafe input."""
        _mock_rails(iorails_sync, input_result=_unsafe("content safety check input"))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check([{"role": "user", "content": "bad"}])

        assert result.status == RailStatus.BLOCKED
        assert result.rail == "content safety check input"

    def test_check_with_explicit_rails_skips_output(self, iorails_sync):
        """Sync check() honors rail_types, skipping the output rail."""
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
        """Sync check() builds the ephemeral engine with _report_usage=False and tracing/metrics disabled."""
        _mock_rails(iorails_sync)

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync) as mock_iorails:
            iorails_sync.check([{"role": "user", "content": "hello"}])

        mock_iorails.assert_called_once()
        assert mock_iorails.call_args.kwargs == {"_report_usage": False}
        passed_config = mock_iorails.call_args.args[0]
        assert passed_config.tracing is None or not passed_config.tracing.enabled
        assert passed_config.metrics is None or not passed_config.metrics.enabled

    def test_check_raises_when_called_from_async_loop(self, iorails_sync):
        """Sync check() called inside a running loop raises a RuntimeError pointing to check_async."""

        async def call_check():
            """Invoke the sync check() from within a running event loop."""
            iorails_sync.check([{"role": "user", "content": "hi"}])

        with pytest.raises(RuntimeError, match="inside async code"):
            asyncio.run(call_check())


class TestCheckAsyncAutoStart:
    """check_async drives the engine lifecycle like generate_async (full parity)."""

    @pytest.mark.asyncio
    async def test_check_async_calls_start(self, iorails):
        """check_async starts the engine before running rails."""
        iorails.engine_registry.start = AsyncMock()
        _mock_rails(iorails)

        assert not iorails._running
        await iorails.check_async([{"role": "user", "content": "hi"}])

        iorails.engine_registry.start.assert_called_once()
        assert iorails._running

    @pytest.mark.asyncio
    async def test_check_async_start_is_idempotent(self, iorails):
        """Repeated check_async calls start the engine only once."""
        iorails.engine_registry.start = AsyncMock()
        _mock_rails(iorails)

        await iorails.check_async([{"role": "user", "content": "hi"}])
        await iorails.check_async([{"role": "user", "content": "hi"}])

        iorails.engine_registry.start.assert_called_once()


class TestCheckAsyncErrors:
    """check_async surfaces rail/engine exceptions instead of swallowing them."""

    @pytest.mark.asyncio
    async def test_check_async_propagates_exception(self, iorails):
        """An exception raised by a rail propagates out of check_async."""
        iorails.rails_manager.is_input_safe = AsyncMock(side_effect=RuntimeError("rail boom"))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=SAFE)

        with pytest.raises(RuntimeError, match="rail boom"):
            await iorails.check_async([{"role": "user", "content": "hi"}])


class TestCheckHelpers:
    """Direct unit tests for the duplicated message helpers."""

    def test_determine_rails_user_only(self):
        """User-only messages select input rails."""
        assert _determine_rails_from_messages([{"role": "user", "content": "hi"}]) == {"rails": ["input"]}

    def test_determine_rails_assistant_only(self):
        """Assistant-only messages select output rails."""
        assert _determine_rails_from_messages([{"role": "assistant", "content": "hi"}]) == {"rails": ["output"]}

    def test_determine_rails_both(self):
        """User+assistant messages select both input and output rails."""
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        assert _determine_rails_from_messages(msgs) == {"rails": ["input", "output"]}

    def test_determine_rails_none_when_no_user_or_assistant(self, caplog):
        """Messages without a user/assistant role return None and log a warning."""
        with caplog.at_level(logging.WARNING, logger="nemoguardrails.guardrails.iorails"):
            assert _determine_rails_from_messages([{"role": "system", "content": "x"}]) is None
        assert "no user or assistant messages" in caplog.text

    def test_get_last_content_by_role_returns_last_match(self):
        """Returns the content of the last message matching the role."""
        msgs = [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
        assert _get_last_content_by_role(msgs, "user") == "second"

    def test_get_last_content_by_role_missing_returns_empty(self):
        """Returns '' when no message matches the role."""
        assert _get_last_content_by_role([{"role": "system", "content": "x"}], "user") == ""

    def test_get_last_content_by_role_none_content_returns_empty(self):
        """content=None on the matched message is normalized to ''."""
        assert _get_last_content_by_role([{"role": "user", "content": None}], "user") == ""
