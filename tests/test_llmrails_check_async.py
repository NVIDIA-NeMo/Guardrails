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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.llmrails import (
    _determine_rails_from_messages,
    _get_blocking_rail,
    _get_last_content_by_role,
    _get_last_response_content,
    _normalize_messages_for_rails,
)
from nemoguardrails.rails.llm.options import ActivatedRail, GenerationLog, GenerationResponse, RailStatus


class TestDetermineRailsFromMessages:
    def test_empty_messages_returns_none(self, caplog):
        result = _determine_rails_from_messages([])
        assert result is None
        assert "check() called with no user or assistant messages" in caplog.text

    def test_user_only_returns_input_rails(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["input"]}

    def test_assistant_only_returns_output_rails(self):
        messages = [{"role": "assistant", "content": "hello"}]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["output"]}

    def test_user_and_assistant_returns_both_rails(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["input", "output"]}

    def test_only_system_messages_returns_none(self, caplog):
        messages = [{"role": "system", "content": "You are helpful."}]
        result = _determine_rails_from_messages(messages)
        assert result is None
        assert "check() called with no user or assistant messages" in caplog.text

    def test_only_context_messages_returns_none(self, caplog):
        messages = [{"role": "context", "content": {"key": "value"}}]
        result = _determine_rails_from_messages(messages)
        assert result is None

    def test_only_tool_messages_returns_none(self, caplog):
        messages = [{"role": "tool", "content": "tool output", "tool_call_id": "123"}]
        result = _determine_rails_from_messages(messages)
        assert result is None

    def test_system_and_user_returns_input_rails(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["input"]}

    def test_system_and_assistant_returns_output_rails(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "hello"},
        ]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["output"]}

    def test_multiple_users_returns_input_rails(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "hello again"},
        ]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["input"]}

    def test_complex_conversation_returns_both_rails(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
            {"role": "assistant", "content": "I'm fine"},
        ]
        result = _determine_rails_from_messages(messages)
        assert result == {"rails": ["input", "output"]}


class TestNormalizeMessagesForRails:
    def test_input_rails_returns_unchanged(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _normalize_messages_for_rails(messages, ["input"])
        assert result == messages

    def test_output_rails_with_user_returns_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_messages_for_rails(messages, ["output"])
        assert result == messages

    def test_output_rails_without_user_adds_empty_user(self):
        messages = [{"role": "assistant", "content": "hello"}]
        result = _normalize_messages_for_rails(messages, ["output"])
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": ""}
        assert result[1] == messages[0]

    def test_both_rails_returns_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_messages_for_rails(messages, ["input", "output"])
        assert result == messages

    def test_output_rails_with_system_only_adds_empty_user(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_messages_for_rails(messages, ["output"])
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": ""}


class TestGetContentByRole:
    def test_get_user_content(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _get_last_content_by_role(messages, "user") == "hello"

    def test_get_assistant_content(self):
        messages = [{"role": "assistant", "content": "hi there"}]
        assert _get_last_content_by_role(messages, "assistant") == "hi there"

    def test_returns_last_matching_role(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        assert _get_last_content_by_role(messages, "user") == "second"

    def test_role_not_found_returns_empty(self):
        messages = [{"role": "system", "content": "Be helpful"}]
        assert _get_last_content_by_role(messages, "user") == ""

    def test_empty_messages_returns_empty(self):
        assert _get_last_content_by_role([], "user") == ""

    def test_missing_content_returns_empty(self):
        messages = [{"role": "user"}]
        assert _get_last_content_by_role(messages, "user") == ""


class TestGetBlockingRail:
    def test_no_log_returns_none(self):
        response = GenerationResponse(response="hello")
        response.log = None
        assert _get_blocking_rail(response) is None

    def test_empty_activated_rails_returns_none(self):
        response = GenerationResponse(response="hello")
        response.log = GenerationLog(activated_rails=[])
        assert _get_blocking_rail(response) is None

    def test_no_blocking_rail_returns_none(self):
        response = GenerationResponse(response="hello")
        rail = ActivatedRail(type="input", name="test rail", stop=False)
        response.log = GenerationLog(activated_rails=[rail])
        assert _get_blocking_rail(response) is None

    def test_blocking_rail_returns_name(self):
        response = GenerationResponse(response="hello")
        rail = ActivatedRail(type="input", name="content safety", stop=True)
        response.log = GenerationLog(activated_rails=[rail])
        assert _get_blocking_rail(response) == "content safety"

    def test_multiple_rails_returns_first_blocking(self):
        response = GenerationResponse(response="hello")
        rail1 = ActivatedRail(type="input", name="rail1", stop=False)
        rail2 = ActivatedRail(type="input", name="rail2", stop=True)
        rail3 = ActivatedRail(type="output", name="rail3", stop=True)
        response.log = GenerationLog(activated_rails=[rail1, rail2, rail3])
        assert _get_blocking_rail(response) == "rail2"


class TestGetResponseContent:
    def test_string_response(self):
        response = GenerationResponse(response="hello world")
        assert _get_last_response_content(response) == "hello world"

    def test_list_response_with_content(self):
        response = GenerationResponse(response=[{"role": "assistant", "content": "hello"}])
        assert _get_last_response_content(response) == "hello"

    def test_list_response_multiple_messages(self):
        response = GenerationResponse(
            response=[
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )
        assert _get_last_response_content(response) == "second"

    def test_empty_list_response(self):
        response = GenerationResponse(response=[])
        assert _get_last_response_content(response) == ""

    def test_list_response_missing_content(self):
        response = GenerationResponse(response=[{"role": "assistant"}])
        assert _get_last_response_content(response) == ""


@pytest.fixture
def mock_rails():
    config = RailsConfig.from_content(
        """
        define flow input rail
          if $user_message == "block"
            bot refuse to respond
            stop
          else if $user_message == "modify"
            $user_message = "modified input"

        define flow output rail
          if $bot_message == "block output"
            bot refuse to respond
            stop
          else if $bot_message == "modify output"
            $bot_message = "modified output"
        """,
        """
        rails:
            input:
                flows:
                    - input rail
            output:
                flows:
                    - output rail
        """,
    )
    return LLMRails(config)


class TestCheckAsyncIntegration:
    @pytest.mark.asyncio
    async def test_input_passed(self, mock_rails):
        messages = [{"role": "user", "content": "hello"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        assert result.rail is None

    @pytest.mark.asyncio
    async def test_input_blocked(self, mock_rails):
        messages = [{"role": "user", "content": "block"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.BLOCKED
        assert result.rail is not None

    @pytest.mark.asyncio
    async def test_input_modified(self, mock_rails):
        messages = [{"role": "user", "content": "modify"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.MODIFIED
        assert result.content == "modified input"
        assert result.rail is None

    @pytest.mark.asyncio
    async def test_output_passed(self, mock_rails):
        messages = [{"role": "assistant", "content": "hello"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
        assert result.rail is None

    @pytest.mark.asyncio
    async def test_output_blocked(self, mock_rails):
        messages = [{"role": "assistant", "content": "block output"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.BLOCKED
        assert result.rail is not None

    @pytest.mark.asyncio
    async def test_output_modified(self, mock_rails):
        messages = [{"role": "assistant", "content": "modify output"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.MODIFIED
        assert result.content == "modified output"
        assert result.rail is None

    @pytest.mark.asyncio
    async def test_input_output_both_passed(self, mock_rails):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED

    @pytest.mark.asyncio
    async def test_input_output_input_blocked(self, mock_rails):
        messages = [
            {"role": "user", "content": "block"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_input_output_output_blocked(self, mock_rails):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "block output"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_input_output_input_modified_returns_passed(self, mock_rails):
        messages = [
            {"role": "user", "content": "modify"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED
        assert result.content == "hi there"

    @pytest.mark.asyncio
    async def test_no_user_or_assistant_returns_passed(self, mock_rails):
        messages = [{"role": "system", "content": "Be helpful"}]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED
        assert result.content == "Be helpful"

    @pytest.mark.asyncio
    async def test_empty_messages_returns_passed(self, mock_rails):
        result = await mock_rails.check_async([])
        assert result.status == RailStatus.PASSED
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_with_system_and_user(self, mock_rails):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED

    @pytest.mark.asyncio
    async def test_with_context_and_user(self, mock_rails):
        messages = [
            {"role": "context", "content": {"key": "value"}},
            {"role": "user", "content": "hello"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED

    @pytest.mark.asyncio
    async def test_complex_conversation(self, mock_rails):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
            {"role": "assistant", "content": "fine"},
        ]
        result = await mock_rails.check_async(messages)
        assert result.status == RailStatus.PASSED
        assert result.content == "fine"

    def test_check_sync_wrapper(self, mock_rails):
        messages = [{"role": "user", "content": "hello"}]
        result = mock_rails.check(messages)
        assert result.status == RailStatus.PASSED
        assert result.content == "hello"
