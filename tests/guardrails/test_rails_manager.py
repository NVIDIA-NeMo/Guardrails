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

"""Unit tests for rails_manager module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.model_manager import ModelManager
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import (
    CONTENT_SAFETY_INPUT_PROMPT,
    CONTENT_SAFETY_OUTPUT_PROMPT,
    NEMOGUARDS_V2_CONFIG,
    TOPIC_SAFETY_INPUT_PROMPT,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def rails_config():
    return RailsConfig.from_content(config=NEMOGUARDS_V2_CONFIG)


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def model_manager(rails_config):
    return ModelManager(rails_config)


@pytest.fixture
def rails_manager(rails_config, model_manager):
    return RailsManager(rails_config, model_manager)


# Alias used by some test classes
@pytest.fixture
def manager(rails_manager):
    return rails_manager


# ---------------------------------------------------------------------------
# RailsManager.__init__
# ---------------------------------------------------------------------------


class TestRailsManagerInit:
    """Test prompts and flows are correctly stored from config."""

    def test_stores_prompts(self, rails_manager):
        """Prompts are keyed by task name with underscored flow names."""
        assert "content_safety_check_input $model=content_safety" in rails_manager.prompts
        assert "content_safety_check_output $model=content_safety" in rails_manager.prompts
        assert "topic_safety_check_input $model=topic_control" in rails_manager.prompts
        assert (
            rails_manager.prompts["content_safety_check_input $model=content_safety"].content
            == CONTENT_SAFETY_INPUT_PROMPT
        )
        assert (
            rails_manager.prompts["content_safety_check_output $model=content_safety"].content
            == CONTENT_SAFETY_OUTPUT_PROMPT
        )
        assert (
            rails_manager.prompts["topic_safety_check_input $model=topic_control"].content == TOPIC_SAFETY_INPUT_PROMPT
        )

    def test_input_flows_populated(self, rails_manager):
        """Input flows list is populated from config.rails.input.flows."""
        assert "content safety check input $model=content_safety" in rails_manager.input_flows

    def test_output_flows_populated(self, rails_manager):
        """Output flows list is populated from config.rails.output.flows."""
        assert "content safety check output $model=content_safety" in rails_manager.output_flows

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def test_empty_rails_config(self):
        """Empty config results in no flows and no prompts."""
        config = RailsConfig.from_content(config={"models": []})
        mgr = RailsManager(config, MagicMock())
        assert mgr.input_flows == []
        assert mgr.output_flows == []
        assert mgr.prompts == {}


class TestStaticHelpers:
    """Test flow name parsing and prompt key conversion helpers."""

    def test_flow_name_with_model(self):
        """Strips the $model= parameter from a flow name."""
        assert (
            RailsManager._flow_name("content safety check input $model=content_safety") == "content safety check input"
        )

    def test_flow_name_without_model(self):
        """Returns the flow name unchanged when no $model= is present."""
        assert RailsManager._flow_name("self check input") == "self check input"

    def test_flow_model_type_extracts_model(self):
        """Extracts the model type after $model=."""
        assert RailsManager._flow_model_type("content safety check input $model=content_safety") == "content_safety"

    def test_flow_model_type_no_model_raises(self):
        """Raises RuntimeError when $model= is missing."""
        with pytest.raises(RuntimeError, match="doesn't contain a model type"):
            RailsManager._flow_model_type("self check input")

    def test_flow_to_prompt_key_with_model(self):
        """Converts spaces to underscores in the flow name portion only."""
        result = RailsManager._flow_to_prompt_key("content safety check input $model=content_safety")
        assert result == "content_safety_check_input $model=content_safety"

    def test_flow_to_prompt_key_without_model(self):
        """Converts all spaces to underscores when no $model= present."""
        result = RailsManager._flow_to_prompt_key("self check input")
        assert result == "self_check_input"

    def test_flow_to_prompt_key_preserves_model_param(self):
        """The $model= portion is preserved unchanged after conversion."""
        result = RailsManager._flow_to_prompt_key("content safety check output $model=content_safety")
        assert result == "content_safety_check_output $model=content_safety"


class TestLastContentByRole:
    """Test extracting the last message content for a given role."""

    def test_finds_last_user_message(self):
        """Returns the last user message when multiple exist."""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        result = RailsManager._last_content_by_role(messages, "user")
        assert result == "second"

    def test_finds_assistant_message(self):
        """Works for non-user roles like assistant."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = RailsManager._last_content_by_role(messages, "assistant")
        assert result == "hello"

    def test_no_matching_role_raises(self):
        """Raises RuntimeError when no message has the requested role."""
        messages = [{"role": "assistant", "content": "hello"}]
        with pytest.raises(RuntimeError, match="No user-role content"):
            RailsManager._last_content_by_role(messages, "user")

    def test_empty_messages_raises(self):
        """Raises RuntimeError on an empty message list."""
        with pytest.raises(RuntimeError, match="No user-role content"):
            RailsManager._last_content_by_role([], "user")

    def test_message_with_empty_content_skipped(self):
        """Empty-string content is falsy and gets skipped."""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": ""},
        ]
        result = RailsManager._last_content_by_role(messages, "user")
        assert result == "first"


class TestLastUserContent:
    """Test the _last_user_content convenience wrapper."""

    def test_delegates_to_last_content_by_role(self, manager):
        """Calls _last_content_by_role with role='user'."""
        messages = [{"role": "user", "content": "hello"}]
        result = manager._last_user_content(messages)
        assert result == "hello"


class TestRenderPrompt:
    """Test prompt template lookup and variable substitution."""

    def test_renders_user_input_template(self, manager):
        """Replaces {{ user_input }} in the content safety input prompt."""
        result = manager._render_prompt(
            "content_safety_check_input $model=content_safety",
            user_input="`test message`",
        )
        assert "`test message`" in result
        assert "{{ user_input }}" not in result

    def test_renders_both_user_input_and_bot_response(self, manager):
        """Replaces both {{ user_input }} and {{ bot_response }} in output prompt."""
        result = manager._render_prompt(
            "content_safety_check_output $model=content_safety",
            user_input="`user says`",
            bot_response="`bot says`",
        )
        assert "`user says`" in result
        assert "`bot says`" in result
        assert "{{ user_input }}" not in result
        assert "{{ bot_response }}" not in result

    def test_missing_prompt_key_raises(self, manager):
        """Raises RuntimeError for a prompt key not in the prompts dict."""
        with pytest.raises(RuntimeError, match="No prompt template found"):
            manager._render_prompt("nonexistent_task")

    def test_prompt_with_none_content_raises(self, manager):
        """Raises RuntimeError when the prompt template has content=None."""
        from nemoguardrails.rails.llm.config import TaskPrompt

        manager.prompts["null_content_task"] = TaskPrompt(
            task="null_content_task", content=None, messages=["placeholder"]
        )
        with pytest.raises(RuntimeError, match="No prompt template found"):
            manager._render_prompt("null_content_task")


class TestParseContentSafetyResult:
    """Test conversion of nemoguard parser output to RailResult."""

    def test_safe_result(self, manager):
        """[True] maps to RailResult(is_safe=True)."""
        result = manager._parse_content_safety_result([True])
        assert result == RailResult(is_safe=True)

    def test_unsafe_result_with_categories(self, manager):
        """[False, ...categories] maps to unsafe with comma-joined reason."""
        result = manager._parse_content_safety_result([False, "S1: Violence", "S17: Malware"])
        assert result.is_safe is False
        assert "S1: Violence" in result.reason
        assert "S17: Malware" in result.reason

    def test_unsafe_result_single_category(self, manager):
        """Single violated category appears in the reason string."""
        result = manager._parse_content_safety_result([False, "S2: Sexual"])
        assert result.is_safe is False
        assert "S2: Sexual" in result.reason

    def test_invalid_result_empty_raises(self, manager):
        """Empty list raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Content safety response invalid"):
            manager._parse_content_safety_result([])

    def test_invalid_result_true_with_extras_raises(self, manager):
        """[True, 'extra'] doesn't match either safe or unsafe pattern."""
        with pytest.raises(RuntimeError, match="Content safety response invalid"):
            manager._parse_content_safety_result([True, "extra"])


class TestParseContentSafetyResponses:
    """Test end-to-end JSON parsing of content safety model responses."""

    def test_input_safe_json(self, manager):
        """Safe input JSON returns RailResult(is_safe=True)."""
        response = json.dumps({"User Safety": "safe"})
        result = manager._parse_content_safety_input_response(response)
        assert result.is_safe is True

    def test_input_unsafe_json(self, manager):
        """Unsafe input JSON returns is_safe=False with violated categories."""
        response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "S1: Violence, S8: Hate/Identity Hate",
            }
        )
        result = manager._parse_content_safety_input_response(response)
        assert result.is_safe is False
        assert "S1: Violence" in result.reason

    def test_output_safe_json(self, manager):
        """Safe output JSON returns RailResult(is_safe=True)."""
        response = json.dumps({"User Safety": "safe"})
        result = manager._parse_content_safety_output_response(response)
        assert result.is_safe is True

    def test_output_unsafe_json(self, manager):
        """Unsafe output JSON returns is_safe=False."""
        response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "S12: Profanity",
            }
        )
        result = manager._parse_content_safety_output_response(response)
        assert result.is_safe is False

    def test_input_unparseable_json_returns_unsafe(self, manager):
        """Malformed JSON is treated as unsafe by the nemoguard parser."""
        result = manager._parse_content_safety_input_response("not json at all")
        assert result.is_safe is False


class TestIsInputSafe:
    """Test the is_input_safe orchestration of input rail checks."""

    @pytest.mark.asyncio
    async def test_all_input_rails_safe(self, manager):
        """Returns is_safe=True when all input rails pass."""
        safe_response = json.dumps({"User Safety": "safe"})
        manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        result = await manager.is_input_safe([{"role": "user", "content": "hello"}])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_content_safety_blocks_input(self, manager):
        """Returns is_safe=False with violated categories when content is unsafe."""
        unsafe_response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "S1: Violence",
            }
        )
        manager.model_manager.generate_async = AsyncMock(return_value=unsafe_response)

        result = await manager.is_input_safe([{"role": "user", "content": "violent content"}])
        assert result.is_safe is False
        assert "S1: Violence" in result.reason

    @pytest.mark.asyncio
    async def test_no_input_flows_returns_safe(self, manager):
        """Returns is_safe=True immediately when no input flows are configured."""
        manager.input_flows = []
        result = await manager.is_input_safe([{"role": "user", "content": "anything"}])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, manager):
        """Model exceptions are caught and returned as unsafe with error reason."""
        manager.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("timeout"))

        result = await manager.is_input_safe([{"role": "user", "content": "hello"}])
        assert result.is_safe is False
        assert "error" in result.reason.lower()


class TestIsOutputSafe:
    """Test the is_output_safe orchestration of output rail checks."""

    @pytest.mark.asyncio
    async def test_output_safe(self, manager):
        """Returns is_safe=True when output content is safe."""
        safe_response = json.dumps({"User Safety": "safe"})
        manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        result = await manager.is_output_safe([{"role": "user", "content": "hello"}], "Here's my response")
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_output_unsafe(self, manager):
        """Returns is_safe=False when output content is unsafe."""
        unsafe_response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "S2: Sexual",
            }
        )
        manager.model_manager.generate_async = AsyncMock(return_value=unsafe_response)

        result = await manager.is_output_safe([{"role": "user", "content": "hello"}], "bad response")
        assert result.is_safe is False

    @pytest.mark.asyncio
    async def test_no_output_flows_returns_safe(self, manager):
        """Returns is_safe=True immediately when no output flows are configured."""
        manager.output_flows = []
        result = await manager.is_output_safe([{"role": "user", "content": "hello"}], "any response")
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, manager):
        """Model exceptions are caught and returned as unsafe with error reason."""
        manager.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("fail"))

        result = await manager.is_output_safe([{"role": "user", "content": "hello"}], "response")
        assert result.is_safe is False
        assert "error" in result.reason.lower()


class TestRailDispatch:
    """Test flow dispatch for unknown/unrecognized rail types."""

    @pytest.mark.asyncio
    async def test_unknown_input_rail_returns_safe(self, manager):
        """Unrecognized input flow name is treated as safe (pass-through)."""
        result = await manager._run_input_rail("unknown rail $model=foo", [{"role": "user", "content": "hi"}])
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_unknown_output_rail_returns_safe(self, manager):
        """Unrecognized output flow name is treated as safe (pass-through)."""
        result = await manager._run_output_rail(
            "unknown rail $model=foo", [{"role": "user", "content": "hi"}], "response"
        )
        assert result.is_safe is True


class TestEndToEndContentSafetyCheck:
    """Test content safety input and output from prompt rendering, model call, and response"""

    @pytest.mark.asyncio
    async def test_content_safety_input_e2e(self, manager):
        """Renders the prompt template with user input and sends to content_safety model."""
        safe_response = json.dumps({"User Safety": "safe"})
        manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        flow = "content safety check input $model=content_safety"
        result = await manager._check_content_safety_input(flow, [{"role": "user", "content": "test input"}])
        assert result.is_safe is True

        # Verify the prompt was rendered with user input
        call_args = manager.model_manager.generate_async.call_args
        messages_sent = call_args[0][1]
        assert "test input" in messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_content_safety_output_e2e(self, manager):
        """Renders the prompt template with both user input and bot response."""
        safe_response = json.dumps({"User Safety": "safe"})
        manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        flow = "content safety check output $model=content_safety"
        result = await manager._check_content_safety_output(
            flow, [{"role": "user", "content": "user query"}], "bot answer"
        )
        assert result.is_safe is True

        call_args = manager.model_manager.generate_async.call_args
        messages_sent = call_args[0][1]
        prompt_content = messages_sent[0]["content"]
        assert "user query" in prompt_content
        assert "bot answer" in prompt_content
