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

from typing import Any, Dict, Optional

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.library.clavata.request import (
    CreateJobResponse,
    Job,
    Report,
    Result,
    SectionReport,
)
from tests.utils import TestChat


@action(is_system_action=True)
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}
    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


BASE_COLANG = """
    define user express greeting
      "hi"

    define flow
      user express greeting
      bot express greeting

    define bot express greeting
      "Hello there!"

    define bot refuse to respond
      "I cannot respond to that request."
"""


@pytest.mark.unit
def test_clavata_no_active_policy_check(monkeypatch):
    """Test that without active policy checks, messages pass through."""
    monkeypatch.setenv("CLAVATA_API_KEY", "")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    Violence: 00000000-0000-0000-0000-000000000000
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat >> "Hi! I want to talk about dogs barking."
    chat << "Hello there!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_input_policy_check(monkeypatch):
    """Test that input policy checks block messages about animal sounds."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSounds: 00000000-0000-0000-0000-000000000000
                  input:
                    policy: AnimalSounds
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    # Mock response from Clavata API
    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        # Use the factory to create a response with a matching policy and label
        mock_response = create_clavata_response(labels={"DogBarking": True})

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Woof woof"
        # Block given the policy matched
        await chat.bot_async("I cannot respond to that request.")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_label_match_logic_any(monkeypatch):
    """Test that label_match_logic: ANY works correctly when at least one label matches."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSoundsPolicy: 00000000-0000-0000-0000-000000000000
                  input:
                    policy: AnimalSoundsPolicy
                    labels:
                      - DogBarking
                      - CatMeowing
                      - CowMooing
                    label_match_logic: ANY
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        # One matching label
        mock_response = create_clavata_response(
            labels={"DogBarking": False, "CatMeowing": False, "CowMooing": True},
        )

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Moo"
        # Block given ANY logic is used and one label matches
        await chat.bot_async("I cannot respond to that request.")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_label_match_logic_any_no_match(monkeypatch):
    """Test that label_match_logic: ANY allows messages when no specified labels match."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSoundsPolicy: 00000000-0000-0000-0000-000000000000
                  input:
                    policy: AnimalSoundsPolicy
                    labels:
                      - DogBarking
                      - CatMeowing
                    label_match_logic: ANY
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        mock_response = create_clavata_response(
            labels={"DogBarking": False, "CatMeowing": False, "HorseNeighing": False},
        )

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Hey"
        # Pass given no labels matched
        await chat.bot_async("Hello there!")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_label_match_logic_all(monkeypatch):
    """Test that label_match_logic: ALL requires all specified labels to match."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSoundsPolicy: 00000000-0000-0000-0000-000000000000
                  label_match_logic: ALL
                  input:
                    policy: AnimalSoundsPolicy
                    labels:
                      - DogBarking
                      - CatMeowing
                      - CowMooing
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        mock_response = create_clavata_response(
            labels={
                "DogBarking": True,
                "CatMeowing": True,
                "CowMooing": True,
                "HorseNeighing": False,
            },
        )

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Woof woof, meow, moo."
        # Block given all specified labels matched and we're using ALL logic
        await chat.bot_async("I cannot respond to that request.")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_label_match_logic_all_partial_match(monkeypatch):
    """Test that label_match_logic: ALL allows messages when only some labels match."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSoundsPolicy: 00000000-0000-0000-0000-000000000000
                  label_match_logic: ALL
                  input:
                    policy: AnimalSoundsPolicy
                    labels:
                      - DogBarking
                      - CatMeowing
                      - CowMooing
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        mock_response = create_clavata_response(
            labels={"DogBarking": True, "CatMeowing": True, "CowMooing": False},
        )

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Hi! I want to talk about dogs and cats but not cows."
        # Pass given not all specified labels matched
        await chat.bot_async("Hello there!")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_empty_labels(monkeypatch):
    """Test that any policy match blocks the message even if no labels are specified."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSoundsPolicy: 00000000-0000-0000-0000-000000000000
                  input:
                    policy: AnimalSoundsPolicy
                    # No labels specified, so any policy match should block
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        mock_response = create_clavata_response(labels={"SomeLabel": True})

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Hi! I want to talk about something that matches the policy."
        # Block given the policy matched
        await chat.bot_async("I cannot respond to that request.")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clavata_policy_no_match(monkeypatch):
    """Test that when the policy doesn't match at all, the message passes through."""
    monkeypatch.setenv("CLAVATA_API_KEY", "test_api_key")

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                clavata:
                  policies:
                    AnimalSoundsPolicy: 00000000-0000-0000-0000-000000000000
                  input:
                    policy: AnimalSoundsPolicy
              input:
                flows:
                  - clavata check input
        """,
        colang_content=BASE_COLANG,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello there!"',
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

        # No policy match
        mock_response = create_clavata_response()

        m.post(
            "https://gateway.app.clavata.ai:8443/v1/jobs",
            payload=mock_response,
            status=200,
        )

        chat >> "Hi! I want to talk about something innocent."
        # Pass given no policy match
        await chat.bot_async("Hello there!")


def create_clavata_response(
    failed=False,
    labels: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Factory function to create a properly formatted Clavata API response.

    Args:
        failed (bool): Whether the API call failed
        labels (dict): Dictionary of label names to match status, e.g. {"DogBarking": True, "CatMeowing": False}
                      If None, no labels will be included
        policy_name (str): The name of the policy
        policy_id (str): The ID of the policy

    Returns:
        dict: A mock Clavata API response
    """
    if labels is None:
        labels = {}

    response = CreateJobResponse(
        job=Job(
            status="JOB_STATUS_FAILED" if failed else "JOB_STATUS_COMPLETED",
            results=[
                Result(
                    report=Report(
                        result=(
                            "OUTCOME_FAILED"
                            if failed
                            else ("OUTCOME_TRUE" if labels else "OUTCOME_FALSE")
                        ),
                        sectionEvaluationReports=[
                            SectionReport(
                                name=lbl,
                                result=("OUTCOME_TRUE" if matched else "OUTCOME_FALSE"),
                                message="",
                            )
                            for lbl, matched in labels.items()
                        ],
                    )
                )
            ],
        )
    )

    return response.model_dump()
