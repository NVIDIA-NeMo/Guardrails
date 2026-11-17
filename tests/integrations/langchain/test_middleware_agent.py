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

"""The round-trip ID through a real agent graph, which is what carries it between the hooks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from nemoguardrails.context import get_roundtrip_id, roundtrip_id
from nemoguardrails.integrations.langchain.middleware import (
    GuardrailsMiddleware,
    InputRailsMiddleware,
    OutputRailsMiddleware,
)
from nemoguardrails.rails.llm.options import RailsResult, RailStatus

THREAD = {"configurable": {"thread_id": "thread-7"}}


def _recording_middleware(middleware_class, seen: list, **kwargs):
    """A middleware whose rails record the round-trip ID of each leg they check."""
    rails = MagicMock()
    rails.config = MagicMock()
    rails.config.rails.input.flows = ["input flow"]
    rails.config.rails.output.flows = ["output flow"]

    async def check(messages, rail_types):
        seen.append((rail_types[0].value, get_roundtrip_id()))
        return RailsResult(status=RailStatus.PASSED, content=messages[-1]["content"])

    rails.check_async = AsyncMock(side_effect=check)
    with (
        patch("nemoguardrails.integrations.langchain.middleware.LLMRails", return_value=rails),
        patch("nemoguardrails.integrations.langchain.middleware.RailsConfig.from_path", return_value=MagicMock()),
    ):
        return middleware_class(config_path="./config", **kwargs)


def _agent(middleware: list, replies: int = 1, checkpointer=None):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="yo")] * replies))
    return create_agent(model=model, tools=[], middleware=middleware, checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_both_legs_of_one_model_call_share_the_id():
    seen = []
    agent = _agent([_recording_middleware(GuardrailsMiddleware, seen)])

    await agent.ainvoke({"messages": [("user", "hi")]})

    assert [leg for leg, _ in seen] == ["input", "output"]
    assert seen[0][1] == seen[1][1] is not None


@pytest.mark.asyncio
async def test_input_and_output_middlewares_of_one_call_share_the_id():
    """The documented pairing: two middlewares, each with its own before-model node."""
    seen = []
    agent = _agent(
        [
            _recording_middleware(InputRailsMiddleware, seen),
            _recording_middleware(OutputRailsMiddleware, seen),
        ]
    )

    await agent.ainvoke({"messages": [("user", "hi")]})

    assert [leg for leg, _ in seen] == ["input", "output"]
    assert seen[0][1] == seen[1][1] is not None


@pytest.mark.asyncio
async def test_each_model_call_of_a_checkpointed_thread_gets_its_own_id():
    seen = []
    agent = _agent([_recording_middleware(GuardrailsMiddleware, seen)], replies=2, checkpointer=InMemorySaver())

    await agent.ainvoke({"messages": [("user", "hi")]}, THREAD)
    await agent.ainvoke({"messages": [("user", "and again")]}, THREAD)

    first, second = seen[:2], seen[2:]
    assert first[0][1] == first[1][1]
    assert second[0][1] == second[1][1]
    assert first[0][1] != second[0][1]


@pytest.mark.asyncio
async def test_an_id_the_application_set_is_kept_on_a_checkpointed_thread():
    """The ID left in the checkpoint by the previous call never overrides the application's."""
    seen = []
    agent = _agent([_recording_middleware(GuardrailsMiddleware, seen)], replies=2, checkpointer=InMemorySaver())

    await agent.ainvoke({"messages": [("user", "hi")]}, THREAD)
    with roundtrip_id("app-round-trip"):
        await agent.ainvoke({"messages": [("user", "and again")]}, THREAD)

    assert [request for _, request in seen[2:]] == ["app-round-trip", "app-round-trip"]
