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
    CONTENT_SAFETY_CONFIG,
    CONTENT_SAFETY_INPUT_PROMPT,
    CONTENT_SAFETY_OUTPUT_PROMPT,
    NEMOGUARDS_CONFIG,
)


# Fixtures using content-safety input and output config
@pytest.fixture
def content_safety_rails_config():
    return RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)


@pytest.fixture
def content_safety_model_manager(content_safety_rails_config):
    return ModelManager(content_safety_rails_config.models)


@pytest.fixture
def content_safety_rails_manager(content_safety_rails_config, content_safety_model_manager):
    return RailsManager(content_safety_rails_config, content_safety_model_manager)


# Fixtures using nemoguards config
@pytest.fixture
def nemoguards_rails_config():
    return RailsConfig.from_content(config=NEMOGUARDS_CONFIG)


@pytest.fixture
def nemoguards_model_manager(nemoguards_rails_config):
    return ModelManager(nemoguards_rails_config.models)


@pytest.fixture
def nemoguards_rails_manager(nemoguards_rails_config, nemoguards_model_manager):
    return RailsManager(nemoguards_rails_config, nemoguards_model_manager)


class TestRailsManagerInit:
    """Test prompts and flows are correctly stored from config."""

    def test_stores_prompts(self, content_safety_rails_manager):
        """Prompts are keyed by task name with underscored flow names."""
        assert "content_safety_check_input $model=content_safety" in content_safety_rails_manager.prompts
        assert "content_safety_check_output $model=content_safety" in content_safety_rails_manager.prompts

        assert (
            content_safety_rails_manager.prompts["content_safety_check_input $model=content_safety"].content
            == CONTENT_SAFETY_INPUT_PROMPT
        )
        assert (
            content_safety_rails_manager.prompts["content_safety_check_output $model=content_safety"].content
            == CONTENT_SAFETY_OUTPUT_PROMPT
        )

    def test_input_flows_populated(self, content_safety_rails_manager):
        """Input flows list is populated from config.rails.input.flows."""
        assert "content safety check input $model=content_safety" in content_safety_rails_manager.input_flows

    def test_output_flows_populated(self, content_safety_rails_manager):
        """Output flows list is populated from config.rails.output.flows."""
        assert "content safety check output $model=content_safety" in content_safety_rails_manager.output_flows

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
        with pytest.raises(RuntimeError, match="No user-role content in messages:"):
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

    def test_delegates_to_last_content_by_role(self, content_safety_rails_manager):
        """Calls _last_content_by_role with role='user'."""
        messages = [{"role": "user", "content": "hello"}]
        result = content_safety_rails_manager._last_user_content(messages)
        assert result == "hello"


class TestRenderPrompt:
    """Test prompt template lookup and variable substitution."""

    def test_renders_user_input_template(self, content_safety_rails_manager):
        """Replaces {{ user_input }} in the content safety input prompt."""
        result = content_safety_rails_manager._render_prompt(
            "content_safety_check_input $model=content_safety",
            user_input="`test message`",
        )
        assert "`test message`" in result
        assert "{{ user_input }}" not in result

    def test_renders_both_user_input_and_bot_response(self, content_safety_rails_manager):
        """Replaces both {{ user_input }} and {{ bot_response }} in output prompt."""
        result = content_safety_rails_manager._render_prompt(
            "content_safety_check_output $model=content_safety",
            user_input="`user says`",
            bot_response="`bot says`",
        )
        assert "`user says`" in result
        assert "`bot says`" in result
        assert "{{ user_input }}" not in result
        assert "{{ bot_response }}" not in result

    def test_missing_prompt_key_raises(self, content_safety_rails_manager):
        """Raises RuntimeError for a prompt key not in the prompts dict."""
        with pytest.raises(RuntimeError, match="No prompt template found"):
            content_safety_rails_manager._render_prompt("nonexistent_task")

    def test_prompt_with_none_content_raises(self, content_safety_rails_manager):
        """Raises RuntimeError when the prompt template has content=None."""
        from nemoguardrails.rails.llm.config import TaskPrompt

        content_safety_rails_manager.prompts["null_content_task"] = TaskPrompt(
            task="null_content_task", content=None, messages=["placeholder"]
        )
        with pytest.raises(RuntimeError, match="No prompt template found"):
            content_safety_rails_manager._render_prompt("null_content_task")


class TestParseContentSafetyResult:
    """Test conversion of nemoguard parser output to RailResult."""

    def test_safe_result(self, content_safety_rails_manager):
        """[True] maps to RailResult(is_safe=True)."""
        result = content_safety_rails_manager._parse_content_safety_result([True])
        assert result == RailResult(is_safe=True)

    def test_unsafe_result_with_categories(self, content_safety_rails_manager):
        """[False, ...categories] maps to unsafe with comma-joined reason."""
        result = content_safety_rails_manager._parse_content_safety_result(
            [False, "Guns and Illegal Weapons.", "Hate/Identity Hate."]
        )
        assert not result.is_safe
        assert "Guns and Illegal Weapons." in result.reason
        assert "Hate/Identity Hate." in result.reason

    def test_unsafe_result_single_category(self, content_safety_rails_manager):
        """Single violated category appears in the reason string."""
        result = content_safety_rails_manager._parse_content_safety_result([False, "Malware"])
        assert not result.is_safe
        assert "Malware" in result.reason

    def test_invalid_result_empty_raises(self, content_safety_rails_manager):
        """Empty list raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Content safety response invalid"):
            content_safety_rails_manager._parse_content_safety_result([])

    def test_invalid_result_true_with_extras_raises(self, content_safety_rails_manager):
        """[True, 'extra'] doesn't match either safe or unsafe pattern."""
        with pytest.raises(RuntimeError, match="Content safety response invalid"):
            content_safety_rails_manager._parse_content_safety_result([True, "extra"])


class TestParseContentSafetyResponses:
    """Test end-to-end JSON parsing of content safety model responses."""

    def test_input_safe_json(self, content_safety_rails_manager):
        """Safe input JSON returns RailResult(is_safe=True)."""
        response = json.dumps({"User Safety": "safe"})
        result = content_safety_rails_manager._parse_content_safety_input_response(response)
        assert result.is_safe

    def test_input_unsafe_json(self, content_safety_rails_manager):
        """Unsafe input JSON returns is_safe=False with violated categories."""
        response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "Guns and Illegal Weapons, Criminal Planning/Confessions",
            }
        )
        result = content_safety_rails_manager._parse_content_safety_input_response(response)
        assert not result.is_safe
        assert "Guns and Illegal Weapons" in result.reason
        assert "Criminal Planning/Confessions" in result.reason

    def test_input_safe_output_safe_json(self, content_safety_rails_manager):
        """Input-safe, Safe output JSON returns RailResult(is_safe=True)."""
        response = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
        result = content_safety_rails_manager._parse_content_safety_output_response(response)
        assert result.is_safe

    def test_input_unsafe_output_safe_json(self, content_safety_rails_manager):
        """Output-rails only looks at LLM Response safety, not user input safety
        so this returns safe. It also drops categories if the response is safe
        """
        response = json.dumps(
            {
                "User Safety": "unsafe",
                "Response Safety": "safe",
                "Safety Categories": "Violence, Criminal Planning/Confessions",
            }
        )
        result = content_safety_rails_manager._parse_content_safety_output_response(response)
        assert result.is_safe

    def test_input_safe_output_unsafe_json(self, content_safety_rails_manager):
        """Safe input and unsage output returns is_safe=False and categories"""
        response = json.dumps(
            {
                "User Safety": "safe",
                "Response Safety": "unsafe",
                "Safety Categories": "Fraud/Deception, Illegal Activity",
            }
        )
        result = content_safety_rails_manager._parse_content_safety_output_response(response)
        assert not result.is_safe
        assert "Fraud/Deception" in result.reason
        assert "Illegal Activity" in result.reason

    def test_input_unsafe_output_unsafe_json(self, content_safety_rails_manager):
        """Unsafe output JSON returns is_safe=False."""
        response = json.dumps(
            {
                "User Safety": "unsafe",
                "Response Safety": "unsafe",
                "Safety Categories": "Harassment, Threat",
            }
        )
        result = content_safety_rails_manager._parse_content_safety_output_response(response)
        assert not result.is_safe
        assert "Harassment" in result.reason
        assert "Threat" in result.reason

    def test_input_unparseable_json_returns_unsafe(self, content_safety_rails_manager):
        """Malformed JSON is treated as unsafe by the nemoguard parser."""
        result = content_safety_rails_manager._parse_content_safety_input_response("not json at all")
        assert not result.is_safe


class TestIsInputSafe:
    """Test end-to-end input-rails were called and parsed correctly from the public `is_input_safe` method"""

    @pytest.mark.asyncio
    async def test_content_safety_input_rails_safe(self, content_safety_rails_manager):
        """Returns is_safe=True when all input rails pass."""
        safe_response = json.dumps({"User Safety": "safe"})
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        result = await content_safety_rails_manager.is_input_safe([{"role": "user", "content": "hello"}])
        assert result.is_safe
        content_safety_rails_manager.model_manager.generate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_content_safety_blocks_input(self, content_safety_rails_manager):
        """Returns is_safe=False with violated categories when content is unsafe."""
        unsafe_response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "Violence",
            }
        )
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=unsafe_response)

        result = await content_safety_rails_manager.is_input_safe([{"role": "user", "content": "violent content"}])
        assert not result.is_safe
        assert "Violence" in result.reason

    @pytest.mark.asyncio
    async def test_no_input_flows_returns_safe(self, content_safety_rails_manager):
        """Returns is_safe=True immediately when no input flows are configured."""
        content_safety_rails_manager.input_flows = []
        result = await content_safety_rails_manager.is_input_safe([{"role": "user", "content": "anything"}])
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, content_safety_rails_manager):
        """Model exceptions are caught and returned as unsafe with error reason."""
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("timeout"))

        result = await content_safety_rails_manager.is_input_safe([{"role": "user", "content": "hello"}])
        assert not result.is_safe
        assert "error" in result.reason.lower()


class TestIsOutputSafe:
    """Test the is_output_safe orchestration of output rail checks."""

    @pytest.mark.asyncio
    async def test_output_safe(self, content_safety_rails_manager):
        """Returns is_safe=True when output content is safe."""
        safe_response = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        result = await content_safety_rails_manager.is_output_safe(
            [{"role": "user", "content": "hello"}], "Here's my response"
        )
        assert result.is_safe
        content_safety_rails_manager.model_manager.generate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_unsafe(self, content_safety_rails_manager):
        """Returns is_safe=False when output content is unsafe."""
        unsafe_response = json.dumps(
            {
                "User Safety": "safe",
                "Response Safety": "unsafe",
                "Safety Categories": "Controlled/Regulated Substances, Illegal Activity",
            }
        )
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=unsafe_response)

        result = await content_safety_rails_manager.is_output_safe(
            [{"role": "user", "content": "hello"}], "bad response"
        )
        assert not result.is_safe
        assert "Controlled/Regulated Substances" in result.reason
        content_safety_rails_manager.model_manager.generate_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_output_flows_returns_safe(self, content_safety_rails_manager):
        """Returns is_safe=True immediately when no output flows are configured."""
        content_safety_rails_manager.output_flows = []
        result = await content_safety_rails_manager.is_output_safe(
            [{"role": "user", "content": "hello"}], "any response"
        )
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_model_error_returns_unsafe(self, content_safety_rails_manager):
        """Model exceptions are caught and returned as unsafe with error reason."""
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(side_effect=RuntimeError("fail"))

        result = await content_safety_rails_manager.is_output_safe([{"role": "user", "content": "hello"}], "response")
        assert not result.is_safe
        assert "error" in result.reason.lower()


class TestRailDispatch:
    """Test flow dispatch for unknown/unrecognized rail types."""

    @pytest.mark.asyncio
    async def test_unknown_input_rail_raises(self, content_safety_rails_manager):
        """Unrecognized input flow name is treated as safe (pass-through)."""
        unknown_rail = "unknown rail $model=foo"
        with pytest.raises(RuntimeError, match="Input rail flow `unknown rail` not supported"):
            await content_safety_rails_manager._run_input_rail(unknown_rail, [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_unknown_output_rail_returns_safe(self, content_safety_rails_manager):
        """Unrecognized output flow name is treated as safe (pass-through)."""
        unknown_rail = "unknown rail $model=foo"
        with pytest.raises(RuntimeError, match="Output rail flow `unknown rail` not supported"):
            await content_safety_rails_manager._run_output_rail(
                unknown_rail, [{"role": "user", "content": "hi"}], "response"
            )


class TestEndToEndContentSafetyCheck:
    """Test content safety input and output from prompt rendering, model call, and response"""

    @pytest.mark.asyncio
    async def test_content_safety_input_safe_e2e(self, content_safety_rails_manager):
        """Renders the prompt template with user input and sends to content_safety model."""
        safe_response = json.dumps({"User Safety": "safe"})
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=safe_response)

        flow = "content safety check input $model=content_safety"
        result = await content_safety_rails_manager._check_content_safety_input(
            flow, [{"role": "user", "content": "test input"}]
        )
        assert result.is_safe

        # Verify the prompt was rendered with user input
        call_args = content_safety_rails_manager.model_manager.generate_async.call_args
        messages_sent = call_args[0][1]
        assert "test input" in messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_content_safety_input_unsafe_e2e(self, content_safety_rails_manager):
        """Renders the prompt template with user input and sends to content_safety model."""
        nemoguard_response = json.dumps(
            {
                "User Safety": "unsafe",
                "Safety Categories": "Violence, Criminal Planning/Confessions",
            }
        )
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=nemoguard_response)

        flow = "content safety check input $model=content_safety"
        result = await content_safety_rails_manager._check_content_safety_input(
            flow, [{"role": "user", "content": "test input"}]
        )
        assert not result.is_safe

        # Verify the prompt was rendered with user input
        call_args = content_safety_rails_manager.model_manager.generate_async.call_args
        messages_sent = call_args[0][1]
        assert "test input" in messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_content_safety_output_safe_e2e(self, content_safety_rails_manager):
        """Renders the prompt template with both user input and bot response."""
        nemoguard_response = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=nemoguard_response)

        flow = "content safety check output $model=content_safety"
        result = await content_safety_rails_manager._check_content_safety_output(
            flow, [{"role": "user", "content": "user query"}], "bot answer"
        )
        assert result.is_safe

        call_args = content_safety_rails_manager.model_manager.generate_async.call_args
        messages_sent = call_args[0][1]
        prompt_content = messages_sent[0]["content"]
        assert "user query" in prompt_content
        assert "bot answer" in prompt_content

    @pytest.mark.asyncio
    async def test_content_safety_output_unsafe_e2e(self, content_safety_rails_manager):
        """Renders the prompt template with both user input and bot response."""
        nemoguard_response = json.dumps(
            {
                "User Safety": "unsafe",
                "Response Safety": "unsafe",
                "Safety Categories": "Violence",
            }
        )
        content_safety_rails_manager.model_manager.generate_async = AsyncMock(return_value=nemoguard_response)

        flow = "content safety check output $model=content_safety"
        result = await content_safety_rails_manager._check_content_safety_output(
            flow, [{"role": "user", "content": "user query"}], "bot answer"
        )
        assert not result.is_safe
        assert "Violence" in result.reason

        call_args = content_safety_rails_manager.model_manager.generate_async.call_args
        messages_sent = call_args[0][1]
        prompt_content = messages_sent[0]["content"]
        assert "user query" in prompt_content
        assert "bot answer" in prompt_content
