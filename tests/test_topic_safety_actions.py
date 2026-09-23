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

"""Direct tests for what ``topic_safety_check_input`` sends to the model."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.library.topic_safety.actions import topic_safety_check_input
from nemoguardrails.types import LLMResponse


@pytest.fixture
def task_manager():
    manager = MagicMock()
    manager.render_task_prompt.return_value = "Is this on topic?"
    manager.get_stop_tokens.return_value = []
    manager.get_max_tokens.return_value = None
    return manager


@pytest.fixture
def llm_call_spy():
    target = "nemoguardrails.library.topic_safety.actions.llm_call"
    with patch(target, new_callable=AsyncMock) as spy:
        spy.return_value = LLMResponse(content="on-topic")
        yield spy


async def _check(task_manager, **overrides):
    kwargs = {
        "llms": {"test_model": MagicMock()},
        "llm_task_manager": task_manager,
        "model_name": "test_model",
        "context": {"user_message": "What is AI?"},
        "events": [],
    }
    kwargs.update(overrides)
    return await topic_safety_check_input(**kwargs)


def _llm_params(spy):
    return spy.await_args.kwargs["llm_params"]


def _sent_messages(spy):
    return spy.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.usefixtures("llm_call_spy")
@pytest.mark.parametrize("events", [None, []])
async def test_absent_conversation_history_yields_a_verdict(task_manager, events):
    """The action returns a verdict when the caller supplies no conversation history."""
    result = await _check(task_manager, events=events)

    assert result.is_blocked is False


@pytest.mark.asyncio
async def test_absent_events_send_only_the_system_and_user_turns(task_manager, llm_call_spy):
    """With no events the prompt carries the system instruction and the user turn, nothing else."""
    await _check(task_manager, events=None)

    messages = _sent_messages(llm_call_spy)
    assert [message["type"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "What is AI?"


@pytest.mark.asyncio
async def test_configured_max_tokens_reaches_the_model(task_manager, llm_call_spy):
    """A ``max_tokens`` configured for the task is forwarded in ``llm_params``."""
    task_manager.get_max_tokens.return_value = 42

    await _check(task_manager)

    assert _llm_params(llm_call_spy)["max_tokens"] == 42


@pytest.mark.asyncio
async def test_unconfigured_max_tokens_is_not_sent(task_manager, llm_call_spy):
    """With nothing configured the action sends no ``max_tokens``, leaving the provider default."""
    await _check(task_manager)

    assert "max_tokens" not in _llm_params(llm_call_spy)


@pytest.mark.asyncio
async def test_temperature_is_always_sent(task_manager, llm_call_spy):
    """The topic-safety temperature is sent regardless of whether ``max_tokens`` is configured."""
    await _check(task_manager)

    assert _llm_params(llm_call_spy)["temperature"] == pytest.approx(0.01)
