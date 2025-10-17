# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Optional

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def build_kb():
    with open(
        os.path.join(CONFIGS_FOLDER, "autoalign_groundness", "kb", "kb.md"), "r"
    ) as f:
        content = f.readlines()

    return content


@action(is_system_action=True)
async def retrieve_relevant_chunks():
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    context_updates = {}
    relevant_chunks = "\n".join(build_kb())
    context_updates["relevant_chunks"] = relevant_chunks
    context_updates["relevant_chunks_sep"] = relevant_chunks.split("\n")

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
async def test_groundness_correct(httpx_mock):
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "autoalign_groundness"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask about pluto",
            "bot provide pluto answer",
            "That's correct! Pluto's orbit is indeed eccentric, meaning it is not a perfect circle. This causes Pluto "
            "to come closer to the Sun than Neptune at times. However, despite this, the two planets do not collide "
            "due to a stable orbital resonance. Orbital resonance is when two objects orbiting a common point exert a "
            "regular influence on each other, keeping their orbits stable and preventing collisions. In the case of "
            "Pluto and Neptune, their orbits are in a specific ratio that keeps them from crashing into each other. "
            "It's a fascinating example of the intricate dance of celestial bodies in our solar system!",
        ],
    )

    async def mock_autoalign_groundedness_output_api(
        context: Optional[dict] = None, **kwargs
    ):
        query = context.get("bot_message")
        if (
            query
            == "That's correct! Pluto's orbit is indeed eccentric, meaning it is not a perfect circle. This "
            "causes Pluto to come closer to the Sun than Neptune at times. However, despite this, "
            "the two planets do not collide due to a stable orbital resonance. Orbital resonance is when two "
            "objects orbiting a common point exert a regular influence on each other, keeping their orbits "
            "stable and preventing collisions. In the case of Pluto and Neptune, their orbits are in a "
            "specific ratio that keeps them from crashing into each other. It's a fascinating example of the "
            "intricate dance of celestial bodies in our solar system!"
        ):
            return 0.52
        else:
            return 0.0

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    chat.app.register_action(
        mock_autoalign_groundedness_output_api, "autoalign_groundedness_output_api"
    )

    (
        chat
        >> "Pluto, with its eccentric orbit, comes closer to the Sun than Neptune at times, yet a stable orbital "
        "resonance ensures they do not collide."
    )

    await chat.bot_async(
        "That's correct! Pluto's orbit is indeed eccentric, meaning it is not a perfect circle. This causes Pluto to "
        "come closer to the Sun than Neptune at times. However, despite this, the two planets do not collide due to a "
        "stable orbital resonance. Orbital resonance is when two objects orbiting a common point exert a regular "
        "influence on each other, keeping their orbits stable and preventing collisions. In the case of Pluto and "
        "Neptune, their orbits are in a specific ratio that keeps them from crashing into each other. It's a "
        "fascinating example of the intricate dance of celestial bodies in our solar system!"
    )


@pytest.mark.asyncio
async def test_groundness_check_wrong(httpx_mock):
    # Test  - Very low score - Not factual
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "autoalign_groundness"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask about pluto",
            "bot provide pluto answer",
            "Actually, Pluto does have moons! In addition to Charon, which is the largest moon of Pluto and has a "
            "diameter greater than Pluto's, there are four other known moons: Styx, Nix, Kerberos, and Hydra. Styx "
            "and Nix were discovered in 2005, while Kerberos and Hydra were discovered in 2011 and 2012, "
            "respectively. These moons are much smaller than Charon and Pluto, but they are still significant in "
            "understanding the dynamics of the Pluto system. Isn't that fascinating?",
        ],
    )

    async def mock_autoalign_groundedness_output_api(
        context: Optional[dict] = None, **kwargs
    ):
        query = context.get("bot_message")
        if (
            query
            == "Actually, Pluto does have moons! In addition to Charon, which is the largest moon of Pluto and "
            "has a diameter greater than Pluto's, there are four other known moons: Styx, Nix, Kerberos, "
            "and Hydra. Styx and Nix were discovered in 2005, while Kerberos and Hydra were discovered in 2011 "
            "and 2012, respectively. These moons are much smaller than Charon and Pluto, but they are still "
            "significant in understanding the dynamics of the Pluto system. Isn't that fascinating?"
        ):
            return 0.0
        else:
            return 1.0

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    chat.app.register_action(
        mock_autoalign_groundedness_output_api, "autoalign_groundedness_output_api"
    )
    (
        chat
        >> "Pluto has no known moons; Charon, the smallest, has a diameter greater than Pluto's, along with the "
        "non-existent Styx, Nix, Kerberos, and Hydra."
    )
    await chat.bot_async(
        "Actually, Pluto does have moons! In addition to Charon, which is the largest moon of Pluto and "
        "has a diameter greater than Pluto's, there are four other known moons: Styx, Nix, Kerberos, "
        "and Hydra. Styx and Nix were discovered in 2005, while Kerberos and Hydra were discovered in 2011 "
        "and 2012, respectively. These moons are much smaller than Charon and Pluto, but they are still "
        "significant in understanding the dynamics of the Pluto system. Isn't that fascinating?"
    )


@pytest.mark.asyncio
async def test_factcheck():
    config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "autoalign_factchecker")
    )

    chat = TestChat(config, llm_completions=["factually correct response"])

    async def mock_autoalign_factcheck_output_api(
        context: Optional[dict] = None, **kwargs
    ):
        user_prompt = context.get("user_message")
        bot_response = context.get("bot_message")

        assert user_prompt == "mock user prompt"
        assert bot_response == "factually correct response"

        return 1.0

    chat.app.register_action(
        mock_autoalign_factcheck_output_api, "autoalign_factcheck_output_api"
    )

    chat >> "mock user prompt"
    await chat.bot_async("factually correct response")
