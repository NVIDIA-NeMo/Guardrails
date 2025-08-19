# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test for InternalEvent handling in topic_safety_check_input action."""

from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.colang.v2_x.runtime.flows import InternalEvent
from nemoguardrails.library.topic_safety.actions import topic_safety_check_input


@pytest.mark.asyncio
async def test_topic_safety_check_input_with_internal_events():
    """Test that topic_safety_check_input can handle InternalEvent objects without failing.

    This test would fail before the fix with:
    TypeError: 'InternalEvent' object is not subscriptable
    """
    internal_events = [
        InternalEvent(
            name="UtteranceUserActionFinished",
            arguments={"final_transcript": "Hello, how are you?"},
        ),
        InternalEvent(
            name="StartUtteranceBotAction",
            arguments={"script": "I'm doing well, thank you!"},
        ),
    ]

    class MockTaskManager:
        def render_task_prompt(self, task):
            return "Check if the conversation is on topic."

        def get_stop_tokens(self, task):
            return []

        def get_max_tokens(self, task):
            return 10

    llms = {"topic_control": "mock_llm"}
    llm_task_manager = MockTaskManager()

    with patch(
        "nemoguardrails.library.topic_safety.actions.llm_call", new_callable=AsyncMock
    ) as mock_llm_call:
        mock_llm_call.return_value = "on-topic"

        # should not raise TypeError: 'InternalEvent' object is not subscriptable
        result = await topic_safety_check_input(
            llms=llms,
            llm_task_manager=llm_task_manager,
            model_name="topic_control",
            context={"user_message": "Hello"},
            events=internal_events,
        )

    assert isinstance(result, dict)
    assert "on_topic" in result
    assert isinstance(result["on_topic"], bool)
    assert result["on_topic"] is True
