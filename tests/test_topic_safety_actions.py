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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.library.topic_safety.actions import (
    TOPIC_SAFETY_OUTPUT_RESTRICTION,
    topic_safety_check_input,
)
from nemoguardrails.types import LLMResponse


@pytest.fixture
def task_manager():
    manager = MagicMock()
    manager.render_task_prompt.return_value = "Stay on topic."
    manager.get_stop_tokens.return_value = []
    manager.get_max_tokens.return_value = 10
    return manager


async def _run_topic_safety(task_manager, **kwargs):
    with patch(
        "nemoguardrails.library.topic_safety.actions.llm_call",
        new_callable=AsyncMock,
    ) as mock_llm_call:
        mock_llm_call.return_value = LLMResponse(content="on-topic")
        result = await topic_safety_check_input(
            llms={"topic_control": "topic model"},
            llm_task_manager=task_manager,
            model_name="topic_control",
            context={"user_message": "current question"},
            **kwargs,
        )

    return result, mock_llm_call


@pytest.mark.asyncio
async def test_canonical_messages_are_authoritative_and_preserve_metadata(task_manager):
    canonical_messages = [
        {"role": "user", "content": "earlier question"},
        {
            "role": "assistant",
            "content": "earlier answer",
            "provider_metadata": {"request_id": "response-1"},
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "current question"}],
            "name": "customer",
        },
    ]

    result, mock_llm_call = await _run_topic_safety(
        task_manager,
        messages=canonical_messages,
        events=[{"type": "UserMessage", "text": "ignored event"}],
    )

    assert result.is_blocked is False
    assert mock_llm_call.await_args.args[1] == [
        {
            "type": "system",
            "content": f"Stay on topic.\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}",
        },
        *canonical_messages,
    ]


@pytest.mark.asyncio
async def test_legacy_events_preserve_conversation_order_and_append_current_input(task_manager):
    result, mock_llm_call = await _run_topic_safety(
        task_manager,
        events=[
            {"type": "UserMessage", "text": "earlier question"},
            {"type": "StartUtteranceBotAction", "script": "earlier answer"},
        ],
    )

    assert result.is_blocked is False
    assert mock_llm_call.await_args.args[1] == [
        {
            "type": "system",
            "content": f"Stay on topic.\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}",
        },
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"type": "user", "content": "current question"},
    ]


@pytest.mark.asyncio
async def test_absent_history_still_checks_current_input(task_manager):
    result, mock_llm_call = await _run_topic_safety(task_manager)

    assert result.is_blocked is False
    assert mock_llm_call.await_args.args[1] == [
        {
            "type": "system",
            "content": f"Stay on topic.\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}",
        },
        {"type": "user", "content": "current question"},
    ]


@pytest.mark.asyncio
async def test_empty_canonical_messages_take_precedence_over_legacy_history(task_manager):
    result, mock_llm_call = await _run_topic_safety(
        task_manager,
        messages=[],
        events=[{"type": "UserMessage", "text": "ignored event"}],
    )

    assert result.is_blocked is False
    assert mock_llm_call.await_args.args[1] == [
        {
            "type": "system",
            "content": f"Stay on topic.\n\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}",
        }
    ]
