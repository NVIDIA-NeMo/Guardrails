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

"""Unit tests for RailAction base class."""

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.rail_action import RailAction

# --- Concrete subclass for testing the base class ---


class DummyRailAction(RailAction):
    """Minimal concrete subclass that records calls for testing."""

    action_name = "some flow"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []
        self.fail_at: Optional[str] = None
        self.fake_response: Any = "dummy_response"

    def _extract_messages(self, messages, bot_response):
        self.calls.append("extract_messages")
        return {"user_input": "extracted"}

    def _create_prompt(self, model_type, extracted):
        self.calls.append("create_prompt")
        return [{"role": "user", "content": "test prompt"}]

    async def _get_response(self, model_type, prompt):
        self.calls.append("get_response")
        if self.fail_at == "get_response":
            raise RuntimeError("model call failed")
        return self.fake_response

    def _parse_response(self, response):
        self.calls.append("parse_response")
        return RailResult(is_safe=True)


@pytest.fixture
def dummy_action():
    model_manager = MagicMock()
    task_manager = MagicMock()
    return DummyRailAction(model_manager, task_manager)


class TestRunPipeline:
    """Test that run() calls pipeline steps in order."""

    @pytest.mark.asyncio
    async def test_calls_all_steps_in_order(self, dummy_action):
        result = await dummy_action.run("some flow $model=test", [{"role": "user", "content": "hi"}])
        assert dummy_action.calls == [
            "extract_messages",
            "create_prompt",
            "get_response",
            "parse_response",
        ]
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_flow_name_mismatch_propagates(self, dummy_action):
        with pytest.raises(RuntimeError, match="does not match expected action_name"):
            await dummy_action.run("wrong flow $model=test", [{"role": "user", "content": "hi"}])
        assert dummy_action.calls == []

    @pytest.mark.asyncio
    async def test_get_response_error_returns_unsafe(self, dummy_action):
        """Model/API errors are caught and returned as unsafe RailResult."""
        dummy_action.fail_at = "get_response"
        result = await dummy_action.run("some flow $model=test", [{"role": "user", "content": "hi"}])
        assert not result.is_safe
        assert "model call failed" in result.reason
        assert "parse_response" not in dummy_action.calls


class TestResponseHelpers:
    """Test the concrete _get_llm_response, _get_api_response helpers."""

    @pytest.mark.asyncio
    async def test_get_llm_response_delegates_to_model_manager(self, dummy_action):
        dummy_action.model_manager.generate_async = AsyncMock(return_value="llm output")
        result = await dummy_action._get_llm_response(
            "content_safety", [{"role": "user", "content": "test"}], temperature=0.01
        )
        assert result == "llm output"
        dummy_action.model_manager.generate_async.assert_awaited_once_with(
            "content_safety", [{"role": "user", "content": "test"}], temperature=0.01
        )

    @pytest.mark.asyncio
    async def test_get_api_response_delegates_to_model_manager(self, dummy_action):
        dummy_action.model_manager.api_call = AsyncMock(return_value={"jailbreak": False})
        result = await dummy_action._get_api_response("jailbreak_detection", {"input": "test"})
        assert result == {"jailbreak": False}
        dummy_action.model_manager.api_call.assert_awaited_once_with("jailbreak_detection", {"input": "test"})

    @pytest.mark.asyncio
    async def test_get_local_response_raises_not_implemented(self, dummy_action):
        with pytest.raises(NotImplementedError):
            await dummy_action._get_local_response()


class TestValidateFlowName:
    """Test _validate_flow_name on the base class."""

    def test_mismatched_flow_name_raises(self, dummy_action):
        dummy_action.action_name = "content safety check input"
        with pytest.raises(RuntimeError, match="does not match expected action_name"):
            dummy_action._validate_flow_name("topic safety check input $model=topic_control")


class TestStaticHelpers:
    """Test shared static utilities on the base class."""

    def test_last_user_content(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert RailAction._last_user_content(messages) == "second"

    def test_last_user_content_skips_empty(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": ""},
        ]
        assert RailAction._last_user_content(messages) == "first"

    def test_last_user_content_no_user_raises(self):
        with pytest.raises(RuntimeError, match="No user message"):
            RailAction._last_user_content([{"role": "assistant", "content": "hi"}])

    def test_last_user_content_empty_raises(self):
        with pytest.raises(RuntimeError, match="No user message"):
            RailAction._last_user_content([])

    def test_get_model_type_extracts_model(self, dummy_action):
        assert dummy_action._get_model_type("some flow $model=content_safety") == "content_safety"

    def test_get_model_type_returns_fallback(self, dummy_action):
        dummy_action.fallback_model = "default_model"
        assert dummy_action._get_model_type("some flow") == "default_model"

    def test_get_model_type_returns_none_without_fallback(self, dummy_action):
        assert dummy_action._get_model_type("some flow") is None

    def test_prompt_to_messages_string(self):
        result = RailAction._prompt_to_messages("hello world")
        assert result == [{"role": "user", "content": "hello world"}]

    def test_prompt_to_messages_list(self):
        result = RailAction._prompt_to_messages(
            [
                {"type": "system", "content": "be helpful"},
                {"type": "user", "content": "hi"},
            ]
        )
        assert result == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
