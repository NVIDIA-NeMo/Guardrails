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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from nemoguardrails.exceptions import InvalidCheckRequestError, RailTypeNotConfiguredError
from nemoguardrails.guardrails.guardrails_types import RailDirection, RailResult
from nemoguardrails.guardrails.iorails import (
    INTERNAL_ERROR_MESSAGE,
    REFUSAL_MESSAGE,
    IORails,
    _determine_rails_from_messages,
    _get_last_content_by_role,
)
from nemoguardrails.guardrails.rail_guard import rail_error_outcome
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType, ToolViolationType
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.rail_stubs import bot_message_rewrite, rail_failure, user_message_rewrite
from tests.guardrails.test_data import NEMOGUARDS_CONFIG, TOPIC_SAFETY_CONFIG
from tests.guardrails.test_tool_rails_iorails import (
    CONFIG_TOOLS_CONFIG,
    TOOL_CONFIG,
    WEATHER_TOOL,
    _inject_forbidden_transport,
)
from tests.guardrails.tool_helpers import (
    TOOL_CALL_QUESTION,
    UNREAD_TOOLS_MESSAGE,
    call_violation,
    make_tool_conversation,
    malformed_prior_tool_call_messages,
    multi_turn_reused_call_id_messages,
    result_violation,
    tool_call_turn,
    wire_tool_call,
)

SAFE = RailResult.allow()


def _unsafe(rail: str) -> RailResult:
    """Build an unsafe RailResult carrying the given triggered-rail name."""
    return RailResult.block(reason="unsafe", triggered_rail=rail)


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
        _mock_rails(iorails, input_result=RailResult.block(reason="unsafe"))

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.status == RailStatus.BLOCKED
        assert result.rail is None


class TestCheckAsyncBlockReason:
    """A BLOCKED check says why it blocked; a check that did not block carries no reason."""

    @pytest.mark.asyncio
    async def test_input_block_carries_the_rail_reason(self, iorails):
        """An input block reports the blocking rail's own reason."""
        blocked = RailResult.block(reason="unsafe request", triggered_rail="content safety check input")
        _mock_rails(iorails, input_result=blocked)

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.status == RailStatus.BLOCKED
        assert result.reason == "unsafe request"

    @pytest.mark.asyncio
    async def test_output_block_carries_the_rail_reason(self, iorails):
        """An output block reports the blocking rail's own reason."""
        blocked = RailResult.block(reason="unsafe answer", triggered_rail="content safety check output")
        _mock_rails(iorails, output_result=blocked)

        result = await iorails.check_async([{"role": "assistant", "content": "bad answer"}])

        assert result.status == RailStatus.BLOCKED
        assert result.reason == "unsafe answer"

    @pytest.mark.asyncio
    async def test_reason_falls_back_to_the_triggered_rail(self, iorails):
        """A block whose rail gave no reason reports the rail's name as the reason."""
        _mock_rails(iorails, input_result=RailResult.block(triggered_rail="content safety check input"))

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.reason == "content safety check input"

    @pytest.mark.asyncio
    async def test_reason_falls_back_to_unspecified(self, iorails):
        """A block naming neither a reason nor a rail reports the reason as unspecified."""
        _mock_rails(iorails, input_result=RailResult.block())

        result = await iorails.check_async([{"role": "user", "content": "bad"}])

        assert result.reason == "unspecified"

    @pytest.mark.asyncio
    async def test_failed_rail_reports_its_redacted_failure_reason(self, iorails):
        """A rail that broke reports the client-facing failure reason the fail-closed envelope wrote."""
        _mock_rails(iorails, input_result=rail_failure("f5 guardrails scan input"))

        result = await iorails.check_async([{"role": "user", "content": "hello"}])

        assert result.reason == "f5 guardrails scan input error: provider call failed"

    @pytest.mark.asyncio
    async def test_failed_rail_reason_carries_no_exception_text(self, iorails):
        """A rail that raised an unclassified error is reported by name only, whatever the exception said."""
        exc = ValueError("Details: see https://internal.example/debug token nvapi-abc123secret")
        failure = rail_error_outcome(None, "policyai moderation on input", exc)
        _mock_rails(iorails, input_result=RailResult(failure, triggered_rail="policyai moderation on input"))

        result = await iorails.check_async([{"role": "user", "content": "hello"}])

        assert result.reason == "policyai moderation on input error"

    @pytest.mark.asyncio
    async def test_passed_has_no_reason(self, iorails):
        """A passed check carries no reason."""
        _mock_rails(iorails)

        result = await iorails.check_async([{"role": "user", "content": "hello"}])

        assert result.status == RailStatus.PASSED
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_modified_has_no_reason(self, iorails):
        """A rewritten check carries no reason, because nothing blocked it."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite("masked"))

        result = await iorails.check_async([{"role": "user", "content": "raw"}])

        assert result.status == RailStatus.MODIFIED
        assert result.reason is None


class TestCheckAsyncFailedRail:
    """A rail that failed is reported as an internal error, not as a content refusal."""

    @pytest.mark.asyncio
    async def test_input_rail_failure_returns_the_internal_error_message(self, iorails):
        """A failed input rail blocks with the sentence LLMRails uses for a failed action."""
        _mock_rails(iorails, input_result=rail_failure("f5 guardrails scan input"))

        result = await iorails.check_async([{"role": "user", "content": "hello"}])

        assert result.status == RailStatus.BLOCKED
        assert result.content == INTERNAL_ERROR_MESSAGE
        assert result.rail == "f5 guardrails scan input"

    @pytest.mark.asyncio
    async def test_output_rail_failure_returns_the_internal_error_message(self, iorails):
        """A failed output rail renders the same internal error the input side does."""
        _mock_rails(iorails, output_result=rail_failure("f5 guardrails scan output"))

        result = await iorails.check_async([{"role": "assistant", "content": "hi there"}])

        assert result.status == RailStatus.BLOCKED
        assert result.content == INTERNAL_ERROR_MESSAGE
        assert result.rail == "f5 guardrails scan output"

    @pytest.mark.asyncio
    async def test_a_failed_rail_is_distinguishable_from_a_rail_that_fired(self, iorails):
        """The two blocks differ in content, which is the whole point of the marker."""
        _mock_rails(iorails, input_result=rail_failure("f5 guardrails scan input"))
        failed = await iorails.check_async([{"role": "user", "content": "hello"}])

        _mock_rails(iorails, input_result=_unsafe("f5 guardrails scan input"))
        decided = await iorails.check_async([{"role": "user", "content": "hello"}])

        assert failed.status == decided.status
        assert failed.rail == decided.rail
        assert failed.content != decided.content


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

    def test_check_blocked_carries_the_reason(self, iorails_sync):
        """Sync check() reports why the check blocked, as check_async does."""
        _mock_rails(iorails_sync, input_result=_unsafe("content safety check input"))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check([{"role": "user", "content": "bad"}])

        assert result.reason == "unsafe"

    def test_check_renders_the_internal_error_for_a_failed_rail(self, iorails_sync):
        """The sync wrapper carries the failed-rail rendering, not just the async path."""
        _mock_rails(iorails_sync, input_result=rail_failure("f5 guardrails scan input"))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=iorails_sync):
            result = iorails_sync.check([{"role": "user", "content": "hello"}])

        assert result.status == RailStatus.BLOCKED
        assert result.content == INTERNAL_ERROR_MESSAGE

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


USER_TEXT = "my ssn is 123-45-6789"
MASKED_USER_TEXT = "my ssn is <SSN>"
BOT_TEXT = "call me on 555-0100"
MASKED_BOT_TEXT = "call me on <PHONE>"
CONVERSATION = [{"role": "user", "content": USER_TEXT}, {"role": "assistant", "content": BOT_TEXT}]


@pytest.mark.asyncio
class TestCheckWithRewritingRails:
    """``check`` reports a rewrite as MODIFIED, carrying the text the rails produced."""

    async def test_an_input_rewrite_is_reported_with_the_new_text(self, iorails):
        """The caller gets back what the rails made of their message, not what they sent."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(MASKED_USER_TEXT))

        result = await iorails.check_async([{"role": "user", "content": USER_TEXT}])

        assert result.status == RailStatus.MODIFIED
        assert result.content == MASKED_USER_TEXT

    async def test_a_rewrite_names_no_rail(self, iorails):
        """A rewrite is not a rail triggering, and several rails may have contributed to the text."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(MASKED_USER_TEXT))

        result = await iorails.check_async([{"role": "user", "content": USER_TEXT}])

        assert result.rail is None

    async def test_an_output_rewrite_is_reported_with_the_new_text(self, iorails):
        """The output direction reports against the response its rails checked."""
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=bot_message_rewrite(MASKED_BOT_TEXT))

        result = await iorails.check_async(CONVERSATION, rail_types=[RailType.OUTPUT])

        assert result.status == RailStatus.MODIFIED
        assert result.content == MASKED_BOT_TEXT

    async def test_an_input_rewrite_reaches_the_output_rails(self, iorails):
        """Both directions run against one conversation, so the second sees what the first made of it."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(MASKED_USER_TEXT))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=SAFE)

        await iorails.check_async(CONVERSATION)

        checked_messages = iorails.rails_manager.is_output_safe.call_args.args[0]
        assert _get_last_content_by_role(checked_messages, "user") == MASKED_USER_TEXT

    async def test_an_input_rewrite_is_internal_when_the_output_is_reported(self, iorails):
        """With output rails in play the caller is told about the response, which nothing changed."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(MASKED_USER_TEXT))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=SAFE)

        result = await iorails.check_async(CONVERSATION)

        assert result.status == RailStatus.PASSED
        assert result.content == BOT_TEXT

    async def test_a_block_behind_a_rewrite_is_reported_as_blocked(self, iorails):
        """A later block decides the outcome; the rewrite has nothing left to be applied to."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=user_message_rewrite(MASKED_USER_TEXT))
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=_unsafe("content safety check output"))

        result = await iorails.check_async(CONVERSATION)

        assert result.status == RailStatus.BLOCKED
        assert result.content == REFUSAL_MESSAGE
        assert result.rail == "content safety check output"


class TestUnsatisfiableRailTypes:
    """Requesting a rail type with no configured flows raises RailTypeNotConfiguredError."""

    @pytest.fixture
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def input_only_config(self):
        return RailsConfig.from_content(config=TOPIC_SAFETY_CONFIG)

    @pytest_asyncio.fixture
    async def input_only_engine(self, input_only_config):
        engine = IORails(input_only_config)
        try:
            yield engine
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_unsatisfiable_output_raises(self, input_only_engine):
        with pytest.raises(RailTypeNotConfiguredError, match="output"):
            await input_only_engine.check_async([{"role": "user", "content": "hi"}], rail_types=[RailType.OUTPUT])

    @pytest.mark.asyncio
    async def test_satisfiable_input_succeeds(self, input_only_engine):
        _mock_rails(input_only_engine)
        result = await input_only_engine.check_async([{"role": "user", "content": "hi"}], rail_types=[RailType.INPUT])
        assert result.status == RailStatus.PASSED

    @pytest.mark.asyncio
    async def test_auto_detect_skips_validation(self, input_only_engine):
        _mock_rails(input_only_engine)
        result = await input_only_engine.check_async(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )
        assert result.status == RailStatus.PASSED


_SECRET_VALUE = "SECRET-VALUE"

# The Nemoguard input and output rails plus the global tool validators, for checks that mix rail families.
TOOL_AND_IO_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "tool_output": {"flows": ["tool call validation"]},
        "tool_input": {"flows": ["tool result validation"]},
    },
}

PER_TOOL_ONLY_CONFIG = {
    **TOOL_CONFIG,
    "rails": {
        "config": {"regex_detection": {"tool_output": {"run_sql": {"patterns": [r"DROP\s+TABLE"]}}}},
        "tool_output": {"per_tool": {"run_sql": ["regex check tool output"]}},
    },
}

_CLOSED_RUN_SQL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "additionalProperties": False},
    },
}


@pytest_asyncio.fixture
async def tool_iorails():
    """A started IORails with only the global tool validators, whose model must never be called."""
    async with started_iorails(TOOL_CONFIG) as engine:
        _inject_forbidden_transport(engine)
        yield engine


@pytest_asyncio.fixture
async def config_tools_iorails():
    """Like ``tool_iorails``, with ``get_weather`` declared on the main model's parameters."""
    async with started_iorails(CONFIG_TOOLS_CONFIG) as engine:
        _inject_forbidden_transport(engine)
        yield engine


@pytest_asyncio.fixture
async def configured_tool_iorails(request):
    """A started IORails for the config a test passes indirectly, whose model must never be called."""
    async with started_iorails(request.param) as engine:
        _inject_forbidden_transport(engine)
        yield engine


@pytest_asyncio.fixture
async def tool_and_io_iorails():
    """A started IORails with input, output and tool rails; the input and output rails are stubbed per test."""
    async with started_iorails(TOOL_AND_IO_CONFIG) as engine:
        _inject_forbidden_transport(engine)
        yield engine


_GET_TIME_TOOL = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}

_FINAL_TEXT_TURN = [*make_tool_conversation(), {"role": "assistant", "content": "It's 18C in Paris."}]

# Case id -> (engine config, messages, rail type, tools) for a tool check that finds nothing to block.
_PASSING_TOOL_CHECKS = {
    "declared_call_with_valid_arguments": (
        TOOL_CONFIG,
        tool_call_turn(wire_tool_call()),
        RailType.TOOL_CALL,
        [WEATHER_TOOL],
    ),
    "config_declared_tools_when_tools_is_none": (
        CONFIG_TOOLS_CONFIG,
        tool_call_turn(wire_tool_call()),
        RailType.TOOL_CALL,
        None,
    ),
    "earlier_malformed_turn_not_rechecked": (
        TOOL_CONFIG,
        malformed_prior_tool_call_messages(),
        RailType.TOOL_CALL,
        [WEATHER_TOOL],
    ),
    "final_text_turn_has_no_calls": (TOOL_CONFIG, _FINAL_TEXT_TURN, RailType.TOOL_CALL, []),
    "nameless_result_linked_by_call_id": (
        TOOL_CONFIG,
        make_tool_conversation(result_name=None),
        RailType.TOOL_RESULT,
        None,
    ),
    "call_ids_reused_across_turns": (TOOL_CONFIG, multi_turn_reused_call_id_messages(), RailType.TOOL_RESULT, None),
}

_PARIS = '{"city": "Paris"}'

# Case id -> (engine config, last call's arguments, tools, (violation type, argument path)).
_BLOCKING_TOOL_CALL_CHECKS = {
    "schema_invalid_arguments": (TOOL_CONFIG, "{}", [WEATHER_TOOL], ("arguments_invalid", "/city")),
    "empty_string_arguments_are_malformed": (TOOL_CONFIG, "", [WEATHER_TOOL], ("malformed_arguments", None)),
    "empty_tools_block_every_call": (CONFIG_TOOLS_CONFIG, _PARIS, [], ("tool_not_allowed", None)),
    "request_tools_replace_config_tools": (CONFIG_TOOLS_CONFIG, _PARIS, [_GET_TIME_TOOL], ("tool_not_allowed", None)),
    "duplicate_tool_definitions": (TOOL_CONFIG, _PARIS, [WEATHER_TOOL, WEATHER_TOOL], ("invalid_toolset", None)),
}


class TestCheckToolRailsPass:
    """Tool checks that find nothing to block."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("configured_tool_iorails", "messages", "rail_type", "tools"),
        list(_PASSING_TOOL_CHECKS.values()),
        ids=list(_PASSING_TOOL_CHECKS),
        indirect=["configured_tool_iorails"],
    )
    async def test_tool_check_passes(self, configured_tool_iorails, messages, rail_type, tools):
        """A declared, well-formed call or a linked result passes with no tool violations."""
        result = await configured_tool_iorails.check_async(messages, rail_types=[rail_type], tools=tools)

        assert (result.status, result.tool_violations) == (RailStatus.PASSED, None)


class TestCheckToolCalls:
    """``rail_types=[RailType.TOOL_CALL]`` validates the last assistant message's tool calls."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("configured_tool_iorails", "arguments", "tools", "expected"),
        list(_BLOCKING_TOOL_CALL_CHECKS.values()),
        ids=list(_BLOCKING_TOOL_CALL_CHECKS),
        indirect=["configured_tool_iorails"],
    )
    async def test_tool_call_check_blocks_with_one_violation(self, configured_tool_iorails, arguments, tools, expected):
        """A call the declared tools or its schema reject blocks with one violation of the expected type."""
        violation_type, argument_path = expected

        result = await configured_tool_iorails.check_async(
            tool_call_turn(wire_tool_call(arguments=arguments)), rail_types=[RailType.TOOL_CALL], tools=tools
        )

        assert result.status == RailStatus.BLOCKED
        assert [(v.violation_type, v.argument_path) for v in result.tool_violations] == [
            (ToolViolationType(violation_type), argument_path)
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("configured_tool_iorails", "rail"),
        [(TOOL_CONFIG, "tool call validation"), (PER_TOOL_ONLY_CONFIG, "regex check tool output")],
        ids=["global_validator", "per_tool_rail_only"],
        indirect=["configured_tool_iorails"],
    )
    @pytest.mark.parametrize(
        ("arguments", "tools", "expected"),
        [
            ('{"query": "SELECT 1"}', [], (ToolViolationType.TOOL_NOT_ALLOWED, None, None)),
            ('{"query": 5}', [_CLOSED_RUN_SQL], (ToolViolationType.ARGUMENTS_INVALID, "/query", "type")),
        ],
        ids=["undeclared_tool", "schema_invalid_arguments"],
    )
    async def test_a_bad_call_is_categorized_alike_whichever_rails_check_it(
        self, configured_tool_iorails, rail, arguments, tools, expected
    ):
        """The global validator and a per-tool rail's gate report the same violation; only the rail differs."""
        result = await configured_tool_iorails.check_async(
            tool_call_turn(wire_tool_call("run_sql", arguments)), rail_types=[RailType.TOOL_CALL], tools=tools
        )

        assert [(v.violation_type, v.argument_path, v.schema_keyword) for v in result.tool_violations] == [expected]
        assert [(v.index, v.tool_call_id) for v in result.tool_violations] == [(0, "call_1")]
        assert result.rail == rail

    @pytest.mark.asyncio
    async def test_undeclared_call_blocks(self, tool_iorails):
        """An undeclared call blocks, naming the rail, the reason, and the call in a ``tool_not_allowed`` violation."""
        result = await tool_iorails.check_async(
            tool_call_turn(wire_tool_call("delete_files", "{}")), rail_types=[RailType.TOOL_CALL], tools=[WEATHER_TOOL]
        )

        assert (result.status, result.content) == (RailStatus.BLOCKED, REFUSAL_MESSAGE)
        assert result.rail == "tool call validation"
        assert result.reason == "tool call 'delete_files' is not an allowed tool"
        assert result.tool_violations == [
            call_violation(
                "tool_not_allowed",
                "tool call 'delete_files' is not an allowed tool",
                tool_call_id="call_1",
                tool_name="delete_files",
                index=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_malformed_arguments_block_without_hiding_later_calls(self, tool_iorails):
        """Malformed arguments block with ``malformed_arguments``, never quoted, and the later calls are still checked."""
        messages = tool_call_turn(
            wire_tool_call(arguments='{"city": "SECRET-VALUE"'), wire_tool_call("delete_files", "{}", "call_2")
        )

        result = await tool_iorails.check_async(messages, rail_types=[RailType.TOOL_CALL], tools=[WEATHER_TOOL])

        assert (result.status, result.rail) == (RailStatus.BLOCKED, None)
        assert result.reason == "tool call extraction failed: tool call 'call_1' has malformed arguments"
        assert [(v.violation_type, v.index, v.tool_call_id) for v in result.tool_violations] == [
            (ToolViolationType.MALFORMED_ARGUMENTS, 0, "call_1"),
            (ToolViolationType.TOOL_NOT_ALLOWED, 1, "call_2"),
        ]
        assert _SECRET_VALUE not in result.model_dump_json()


class TestCheckToolResults:
    """``rail_types=[RailType.TOOL_RESULT]`` validates every tool result against the calls it links to."""

    @pytest.mark.asyncio
    async def test_unlinked_result_blocks(self, tool_iorails):
        """A result for an unknown call id blocks, and its violation points at the tool message's position."""
        messages = make_tool_conversation(result_call_id="call_999", result_name=None)

        result = await tool_iorails.check_async(messages, rail_types=[RailType.TOOL_RESULT])

        assert (result.status, result.rail) == (RailStatus.BLOCKED, "tool result validation")
        assert result.reason == "tool result for call_id 'call_999' does not correspond to a prior tool call"
        assert result.tool_violations == [
            result_violation(
                "unknown_call_id",
                "tool result for call_id 'call_999' does not correspond to a prior tool call",
                tool_call_id="call_999",
                index=2,
            )
        ]


class TestCheckToolRoundTrip:
    """A harness checks its model's tool calls, runs the tools, then checks the results, all without generation."""

    @pytest.mark.asyncio
    async def test_harness_round_trip_passes_both_checks(self, tool_iorails):
        """A declared call passes the tool-call check, and the nameless result the harness appends passes next."""
        messages = tool_call_turn(wire_tool_call())

        call_check = await tool_iorails.check_async(messages, rail_types=[RailType.TOOL_CALL], tools=[WEATHER_TOOL])
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "18C"})
        result_check = await tool_iorails.check_async(messages, rail_types=[RailType.TOOL_RESULT])

        assert (call_check.status, result_check.status) == (RailStatus.PASSED, RailStatus.PASSED)

    @pytest.mark.asyncio
    async def test_result_for_an_unknown_call_blocks_the_second_check(self, tool_iorails):
        """A passing tool-call check does not vouch for a result that answers a different call id."""
        messages = tool_call_turn(wire_tool_call())

        call_check = await tool_iorails.check_async(messages, rail_types=[RailType.TOOL_CALL], tools=[WEATHER_TOOL])
        messages.append({"role": "tool", "tool_call_id": "call_2", "content": "18C"})
        result_check = await tool_iorails.check_async(messages, rail_types=[RailType.TOOL_RESULT])

        assert (call_check.status, result_check.status) == (RailStatus.PASSED, RailStatus.BLOCKED)


def _record_calls(engine: IORails, *, input_result: RailResult = SAFE) -> MagicMock:
    """Spy on each rail family's manager entry point; the returned mock's calls give the order they ran in."""
    manager = engine.rails_manager
    order = MagicMock()
    spies = {
        "are_tool_results_safe": ("tool_result", AsyncMock(wraps=manager.are_tool_results_safe)),
        "is_input_safe": ("input", AsyncMock(return_value=input_result)),
        "are_latest_tool_calls_safe": ("tool_call", AsyncMock(wraps=manager.are_latest_tool_calls_safe)),
        "is_output_safe": ("output", AsyncMock(return_value=SAFE)),
    }
    for method, (family, spy) in spies.items():
        order.attach_mock(spy, family)
        setattr(manager, method, spy)
    return order


def _families_run(order: MagicMock) -> list[str]:
    """The rail families a check ran, in order, from the spy ``_record_calls`` returned."""
    return [name for name, _args, _kwargs in order.mock_calls]


# A user turn, an assistant turn with text and a declared call, and the call's result: every family has work.
FULL_TURN = [
    *tool_call_turn(wire_tool_call(), content="Let me check."),
    {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
]
ALL_RAIL_TYPES = [RailType.INPUT, RailType.OUTPUT, RailType.TOOL_CALL, RailType.TOOL_RESULT]


class TestCheckToolRailOrchestration:
    """How a check sequences tool rails with input and output rails, and what it reports."""

    @pytest.mark.asyncio
    async def test_rail_families_run_as_generation_orders_them(self, tool_and_io_iorails):
        """Tool results, input, tool calls, then output: the order generation runs them in."""
        order = _record_calls(tool_and_io_iorails)

        result = await tool_and_io_iorails.check_async(FULL_TURN, rail_types=ALL_RAIL_TYPES, tools=[WEATHER_TOOL])

        assert result.status == RailStatus.PASSED
        assert _families_run(order) == ["tool_result", "input", "tool_call", "output"]

    @pytest.mark.asyncio
    async def test_a_tool_call_block_stops_before_output_rails(self, tool_and_io_iorails):
        """A tool-call block ends the check, so the output rails never run."""
        order = _record_calls(tool_and_io_iorails)

        result = await tool_and_io_iorails.check_async(FULL_TURN, rail_types=ALL_RAIL_TYPES, tools=[])

        assert result.rail == "tool call validation"
        assert _families_run(order) == ["tool_result", "input", "tool_call"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rail_type", "tools", "expected_content"),
        [(RailType.TOOL_CALL, [WEATHER_TOOL], "Let me check."), (RailType.TOOL_RESULT, None, TOOL_CALL_QUESTION)],
        ids=["tool_call_reports_assistant_text", "tool_result_reports_user_text"],
    )
    async def test_tool_check_reports_the_text_of_its_direction(
        self, tool_and_io_iorails, rail_type, tools, expected_content
    ):
        """A tool-call check reports the assistant text, as an output check does; a tool-result check the user text."""
        result = await tool_and_io_iorails.check_async(FULL_TURN, rail_types=[rail_type], tools=tools)

        assert (result.status, result.content) == (RailStatus.PASSED, expected_content)

    @pytest.mark.asyncio
    async def test_input_rewrite_stays_visible_with_a_tool_call_check(self, tool_and_io_iorails):
        """With ``[INPUT, TOOL_CALL]``, an input rewrite is what the check reports, as MODIFIED."""
        _record_calls(tool_and_io_iorails, input_result=user_message_rewrite("What's the weather in <CITY>?"))

        result = await tool_and_io_iorails.check_async(
            FULL_TURN, rail_types=[RailType.INPUT, RailType.TOOL_CALL], tools=[WEATHER_TOOL]
        )

        assert (result.status, result.content) == (RailStatus.MODIFIED, "What's the weather in <CITY>?")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rail_types", "tools", "direction"),
        [([RailType.TOOL_RESULT], None, RailDirection.INPUT), ([RailType.TOOL_CALL], [], RailDirection.OUTPUT)],
        ids=["tool_result", "tool_call"],
    )
    async def test_a_tool_block_records_the_directional_block_metric(self, tool_iorails, rail_types, tools, direction):
        """A tool-result block counts as an input block and a tool-call block as an output block."""
        tool_iorails._metrics_enabled = True
        messages = make_tool_conversation(result_call_id="call_999")

        with patch("nemoguardrails.guardrails.iorails.record_request_blocked") as record_blocked:
            result = await tool_iorails.check_async(messages, rail_types=rail_types, tools=tools)

        assert result.status == RailStatus.BLOCKED
        record_blocked.assert_called_once_with(direction)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rail_types", "blocked_rail"),
        [([RailType.INPUT], "input"), ([RailType.OUTPUT], "output")],
    )
    async def test_input_or_output_block_has_no_tool_violations(self, iorails, rail_types, blocked_rail):
        """Only a tool rail block reports tool violations."""
        blocking = _unsafe(f"content safety check {blocked_rail}")
        _mock_rails(iorails, input_result=blocking, output_result=blocking)

        result = await iorails.check_async(CONVERSATION, rail_types=rail_types)

        assert result.status == RailStatus.BLOCKED
        assert result.tool_violations is None


class TestCheckToolRailConfiguration:
    """Which configs can run a tool check, and what happens when rail types are left to auto-detection."""

    @pytest.mark.asyncio
    async def test_unconfigured_tool_rail_type_raises(self, iorails):
        """A tool rail type whose config section has no rails raises ``RailTypeNotConfiguredError``."""
        with pytest.raises(RailTypeNotConfiguredError, match="rail type 'tool_call' has no configured rails"):
            await iorails.check_async(tool_call_turn(wire_tool_call()), rail_types=[RailType.TOOL_CALL])

    @pytest.mark.asyncio
    async def test_per_tool_rails_alone_count_as_configured(self):
        """A section with only ``per_tool`` rails runs them, and a per-tool block names the call it blocked."""
        messages = tool_call_turn(wire_tool_call("run_sql", '{"query": "DROP TABLE users"}'))
        run_sql = {
            "type": "function",
            "function": {"name": "run_sql", "parameters": {"type": "object", "additionalProperties": True}},
        }

        async with started_iorails(PER_TOOL_ONLY_CONFIG) as engine:
            result = await engine.check_async(messages, rail_types=[RailType.TOOL_CALL], tools=[run_sql])

        assert result.rail == "regex check tool output"
        assert [(v.violation_type, v.rail, v.index) for v in result.tool_violations] == [
            (ToolViolationType.PER_TOOL_RAIL, "regex check tool output", 0)
        ]

    @pytest.mark.asyncio
    async def test_per_tool_rails_alone_leave_unlisted_tools_unchecked(self):
        """Without the global validator only tools with per-tool rails are checked, so an undeclared tool passes."""
        messages = tool_call_turn(wire_tool_call("delete_files", "{}"))

        async with started_iorails(PER_TOOL_ONLY_CONFIG) as engine:
            result = await engine.check_async(messages, rail_types=[RailType.TOOL_CALL], tools=[])

        assert result.status == RailStatus.PASSED

    @pytest.mark.asyncio
    async def test_per_tool_rails_with_no_flows_are_not_configured(self):
        """A ``per_tool`` map whose lists are all empty runs nothing, so the tool rail type counts as unconfigured."""
        config = {**TOOL_CONFIG, "rails": {"tool_output": {"per_tool": {"run_sql": []}}}}

        async with started_iorails(config) as engine:
            with pytest.raises(RailTypeNotConfiguredError, match="rail type 'tool_call' has no configured rails"):
                await engine.check_async(tool_call_turn(wire_tool_call()), rail_types=[RailType.TOOL_CALL])

    @pytest.mark.asyncio
    async def test_tool_rail_type_without_a_main_model_raises(self):
        """Without a ``main`` model there is no wire format to read, so a tool check refuses to run."""
        config = {"models": [], "rails": TOOL_CONFIG["rails"]}

        async with started_iorails(config) as engine:
            with pytest.raises(RailTypeNotConfiguredError, match="needs a `main` model"):
                await engine.check_async(tool_call_turn(wire_tool_call()), rail_types=[RailType.TOOL_CALL])

    @pytest.mark.asyncio
    async def test_auto_detection_runs_input_and_output_only_and_logs_a_hint(self, tool_and_io_iorails, caplog):
        """With ``rail_types`` omitted, tool traffic runs only input and output rails, with an INFO hint."""
        order = _record_calls(tool_and_io_iorails)

        with caplog.at_level(logging.INFO, logger="nemoguardrails.guardrails.iorails"):
            result = await tool_and_io_iorails.check_async(FULL_TURN)

        assert result.status == RailStatus.PASSED
        assert _families_run(order) == ["input", "output"]
        assert "tool rails are configured but were not requested" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("messages", "hint_count"),
        [
            ([{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "call_1", "content": "18C"}], 1),
            ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], 0),
        ],
        ids=["tool_result_only", "no_tool_traffic"],
    )
    async def test_auto_detection_hints_only_at_tool_traffic(self, tool_and_io_iorails, caplog, messages, hint_count):
        """The INFO hint appears when the messages carry a tool result, and not when they carry no tool traffic."""
        _record_calls(tool_and_io_iorails)

        with caplog.at_level(logging.INFO, logger="nemoguardrails.guardrails.iorails"):
            await tool_and_io_iorails.check_async(messages)

        assert caplog.text.count("tool rails are configured but were not requested") == hint_count

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rail_types",
        [None, [], [RailType.INPUT], [RailType.OUTPUT, RailType.TOOL_RESULT]],
        ids=["auto_detection", "empty", "input", "output_and_tool_result"],
    )
    async def test_tools_without_a_tool_call_check_raise(self, tool_and_io_iorails, rail_types):
        """Only a tool_call check reads ``tools``, so any other check given them raises before a rail runs."""
        order = _record_calls(tool_and_io_iorails)

        with pytest.raises(InvalidCheckRequestError, match=f"^{UNREAD_TOOLS_MESSAGE}$"):
            await tool_and_io_iorails.check_async(FULL_TURN, rail_types=rail_types, tools=[WEATHER_TOOL])

        assert _families_run(order) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rail_types", "tools", "error"),
        [
            ([RailType.INPUT], [WEATHER_TOOL], InvalidCheckRequestError),
            ([RailType.TOOL_CALL], None, RailTypeNotConfiguredError),
        ],
        ids=["unread_tools", "unconfigured_rail_type"],
    )
    async def test_request_error_is_raised_before_queueing(self, iorails, caplog, rail_types, tools, error):
        """A request the check cannot serve raises before it is queued, so it takes no queue slot and logs no ERROR."""
        with patch.object(iorails._generate_async_queue, "submit") as submit, caplog.at_level(logging.WARNING):
            with pytest.raises(error):
                await iorails.check_async(CONVERSATION, rail_types=rail_types, tools=tools)

        submit.assert_not_called()
        assert [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR] == []

    def test_sync_check_forwards_tools(self):
        """Sync ``check`` hands ``tools`` to the engine it spins up."""
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            engine = IORails(RailsConfig.from_content(config=TOOL_CONFIG))

        with patch("nemoguardrails.guardrails.iorails.IORails", return_value=engine):
            result = engine.check(
                tool_call_turn(wire_tool_call()), rail_types=[RailType.TOOL_CALL], tools=[WEATHER_TOOL]
            )

        assert result.status == RailStatus.PASSED
