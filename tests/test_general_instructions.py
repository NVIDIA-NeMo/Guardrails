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

from unittest.mock import MagicMock

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import Instruction, Model, RailsConfig
from tests.utils import TestChat


def test_general_instructions_get_included_when_no_canonical_forms_are_defined():
    config: RailsConfig = RailsConfig.from_content(
        config={
            "models": [],
            "instructions": [
                {
                    "type": "general",
                    "content": "This is a conversation between a user and a bot.",
                }
            ],
        }
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  Hello there!",
        ],
    )

    chat >> "hello there!"
    chat << "Hello there!"

    info = chat.app.explain()
    assert (
        "This is a conversation between a user and a bot." in info.llm_calls[0].prompt
    )


def test_get_general_instructions_none():
    """Check we get None when RailsConfig.instructions is None."""

    config = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        colang_version="1.0",
        instructions=None,
    )

    actions = LLMGenerationActions(
        config,
        llm=None,
        llm_task_manager=MagicMock(spec=LLMTaskManager),
        get_embedding_search_provider_instance=MagicMock(),
    )

    instructions = actions._get_general_instructions()
    assert instructions is None


def test_get_general_instructions_empty_list():
    """Check an empty list of instructions returns an empty string"""

    config = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        colang_version="1.0",
    )
    config.instructions = []

    actions = LLMGenerationActions(
        config,
        llm=None,
        llm_task_manager=MagicMock(spec=LLMTaskManager),
        get_embedding_search_provider_instance=MagicMock(),
    )

    instructions = actions._get_general_instructions()
    assert instructions == ""


def test_get_general_instructions_list():
    """Check a list of instructions where the second one is general"""

    first_general_instruction = "Don't answer with any inappropriate content."
    instructions = [
        Instruction(type="specific", content="You're a helpful bot "),
        Instruction(type="general", content=first_general_instruction),
    ]

    config = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        colang_version="1.0",
        instructions=instructions,
    )

    actions = LLMGenerationActions(
        config,
        llm=None,
        llm_task_manager=MagicMock(spec=LLMTaskManager),
        get_embedding_search_provider_instance=MagicMock(),
    )

    instructions = actions._get_general_instructions()
    assert instructions == first_general_instruction


def test_get_sample_conversation_two_turns():
    """Check if the RailsConfig sample_conversation is None we get None back"""

    config = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        colang_version="1.0",
        sample_conversation=None,
    )

    actions = LLMGenerationActions(
        config,
        llm=None,
        llm_task_manager=MagicMock(spec=LLMTaskManager),
        get_embedding_search_provider_instance=MagicMock(),
    )

    conversation = actions._get_sample_conversation_two_turns()
    assert conversation is None


@pytest.mark.asyncio
async def test_search_flows_index_is_none():
    """Check if we try and search the flows index when None we get None back"""

    config = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        colang_version="1.0",
        sample_conversation=None,
    )

    actions = LLMGenerationActions(
        config,
        llm=None,
        llm_task_manager=MagicMock(spec=LLMTaskManager),
        get_embedding_search_provider_instance=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="No flows index found to search"):
        _ = await actions._search_flows_index(text="default action", max_results=1)


@pytest.mark.asyncio
async def test_generate_next_step_empty_event_list():
    """Check if we try and search the flows index when None we get None back"""

    config = RailsConfig(
        models=[Model(type="main", engine="openai", model="gpt-3.5-turbo")],
        colang_version="1.0",
        sample_conversation=None,
    )

    actions = LLMGenerationActions(
        config,
        llm=None,
        llm_task_manager=MagicMock(spec=LLMTaskManager),
        get_embedding_search_provider_instance=MagicMock(),
    )

    with pytest.raises(
        RuntimeError, match="No last user intent found from which to generate next step"
    ):
        _ = await actions.generate_next_step(events=[])


#
# @pytest.mark.asyncio
# async def test_generate_next_step_last_user_intent_is_none():
#
#     #
# events = [{"type": "UserIntent", "content": "You're a helpful bot "}
#           {"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
#
# actions._generate_next_step = MagicMock(return_value="default action")
