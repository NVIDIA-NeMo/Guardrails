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
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


@action(is_system_action=True)
async def retrieve_relevant_chunks():
    """Retrieve relevant chunks from the knowledge base and add them to the context."""
    context_updates = {"relevant_chunks": "Shipping takes at least 3 days."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
async def test_fiddler_safety_rails(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/safety"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )
    with aioresponses() as m:
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-safety",
            payload={
                "fdl_harmful": 0.5,
                "fdl_violent": 0.5,
                "fdl_unethical": 0.5,
                "fdl_illegal": 0.5,
                "fdl_sexual": 0.5,
                "fdl_racist": 0.5,
                "fdl_jailbreaking": 0.5,
                "fdl_harassing": 0.5,
                "fdl_hateful": 0.5,
                "fdl_sexist": 0.5,
                "fdl_roleplaying": 0.5,
            },
        )

        chat >> "Do you ship within 2 days?"
        await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_fiddler_safety_rails_pass(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_ENVIRON", "https://testfiddler.ai")
    monkeypatch.setenv("FIDDLER_API_KEY", "")
    monkeypatch.setenv("FIDDLER_SAFETY_DETECTION_THRESHOLD", "0.8")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/safety"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-safety",
            payload={
                "fdl_harmful": 0.02,
                "fdl_violent": 0.02,
                "fdl_unethical": 0.02,
                "fdl_illegal": 0.02,
                "fdl_sexual": 0.02,
                "fdl_racist": 0.02,
                "fdl_jailbreaking": 0.02,
                "fdl_harassing": 0.02,
                "fdl_hateful": 0.02,
                "fdl_sexist": 0.02,
                "fdl_roleplaying": 0.02,
            },
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("Yes, shipping can be done in 2 days.")


@pytest.mark.asyncio
async def test_fiddler_thresholds(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/thresholds"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-safety",
            payload={
                "fdl_harmful": 0.5,
                "fdl_violent": 0.5,
                "fdl_unethical": 0.5,
                "fdl_illegal": 0.5,
                "fdl_sexual": 0.5,
                "fdl_racist": 0.5,
                "fdl_jailbreaking": 0.5,
                "fdl_harassing": 0.5,
                "fdl_hateful": 0.5,
                "fdl_sexist": 0.5,
                "fdl_roleplaying": 0.5,
            },
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_fiddler_faithfulness_rails(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/faithfulness"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-response-faithfulness",
            payload={"fdl_faithful_score": 0.001},
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("I'm sorry, I can't respond to that.")


@pytest.mark.asyncio
async def test_fiddler_faithfulness_rails_pass(monkeypatch):
    # Mock environment variables
    monkeypatch.setenv("FIDDLER_API_KEY", "")

    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "fiddler/faithfulness"))
    chat = TestChat(
        config,
        llm_completions=[
            "user ask general question",  # user intent
            "Yes, shipping can be done in 2 days.",  # bot response that will be intercepted
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            "https://testfiddler.ai/v3/guardrails/ftl-response-faithfulness",
            payload={"fdl_faithful_score": 0.5},
        )
        chat >> "Do you ship within 2 days?"
        await chat.bot_async("Yes, shipping can be done in 2 days.")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fiddler_safety_request_format_user_message(monkeypatch):
    """Unit test: Verify safety guardrail sends correct request format for user message."""
    from nemoguardrails.library.fiddler.actions import call_fiddler_safety_user

    monkeypatch.setenv("FIDDLER_API_KEY", "test-key")

    config = RailsConfig.from_content(
        yaml_content="""
        rails:
          config:
            fiddler:
              fiddler_endpoint: https://testfiddler.ai
              safety_threshold: 0.5
        """
    )

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "fdl_harmful": 0.1,
            "fdl_violent": 0.1,
            "fdl_unethical": 0.1,
            "fdl_illegal": 0.1,
            "fdl_sexual": 0.1,
            "fdl_racist": 0.1,
            "fdl_jailbreaking": 0.1,
            "fdl_harassing": 0.1,
            "fdl_hateful": 0.1,
            "fdl_sexist": 0.1,
            "fdl_roleplaying": 0.1,
        }
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    mock_post_context = AsyncMock()
    mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_context.__aexit__ = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_post_context)

    mock_client_session = MagicMock()
    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_session.return_value.__aexit__ = AsyncMock()

    with patch("nemoguardrails.library.fiddler.actions.aiohttp.ClientSession", mock_client_session):
        context = {"user_message": "Hello, how are you?"}
        result = await call_fiddler_safety_user(config, context)

    # Verify request format
    assert mock_session.post.called
    call_args = mock_session.post.call_args
    request_payload = call_args[1]["json"]
    assert "data" in request_payload
    assert request_payload["data"]["input"] == "Hello, how are you?"
    assert "prompt" not in request_payload["data"]  # Old format should not be present
    assert not isinstance(request_payload["data"]["input"], list)  # Should be string, not list


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fiddler_safety_request_format_bot_message(monkeypatch):
    """Unit test: Verify safety guardrail sends correct request format for bot message."""
    from nemoguardrails.library.fiddler.actions import call_fiddler_safety_bot

    monkeypatch.setenv("FIDDLER_API_KEY", "test-key")

    config = RailsConfig.from_content(
        yaml_content="""
        rails:
          config:
            fiddler:
              fiddler_endpoint: https://testfiddler.ai
              safety_threshold: 0.5
        """
    )

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(
        return_value={
            "fdl_harmful": 0.1,
            "fdl_violent": 0.1,
            "fdl_unethical": 0.1,
            "fdl_illegal": 0.1,
            "fdl_sexual": 0.1,
            "fdl_racist": 0.1,
            "fdl_jailbreaking": 0.1,
            "fdl_harassing": 0.1,
            "fdl_hateful": 0.1,
            "fdl_sexist": 0.1,
            "fdl_roleplaying": 0.1,
        }
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    mock_post_context = AsyncMock()
    mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_context.__aexit__ = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_post_context)

    mock_client_session = MagicMock()
    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_session.return_value.__aexit__ = AsyncMock()

    with patch("nemoguardrails.library.fiddler.actions.aiohttp.ClientSession", mock_client_session):
        context = {"bot_message": "I can help you with that."}
        result = await call_fiddler_safety_bot(config, context)

    # Verify request format
    assert mock_session.post.called
    call_args = mock_session.post.call_args
    request_payload = call_args[1]["json"]
    assert "data" in request_payload
    assert request_payload["data"]["input"] == "I can help you with that."
    assert "prompt" not in request_payload["data"]  # Old format should not be present
    assert not isinstance(request_payload["data"]["input"], list)  # Should be string, not list


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fiddler_faithfulness_request_format(monkeypatch):
    """Unit test: Verify faithfulness guardrail sends correct request format."""
    from nemoguardrails.library.fiddler.actions import call_fiddler_faithfulness

    monkeypatch.setenv("FIDDLER_API_KEY", "test-key")

    config = RailsConfig.from_content(
        yaml_content="""
        rails:
          config:
            fiddler:
              fiddler_endpoint: https://testfiddler.ai
              faithfulness_threshold: 0.3
        """
    )

    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"fdl_faithful_score": 0.2})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock()

    mock_post_context = AsyncMock()
    mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_context.__aexit__ = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_post_context)

    mock_client_session = MagicMock()
    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_session.return_value.__aexit__ = AsyncMock()

    with patch("nemoguardrails.library.fiddler.actions.aiohttp.ClientSession", mock_client_session):
        context = {
            "bot_message": "Shipping takes 2 days",
            "relevant_chunks": "Shipping takes at least 3 days. We ship worldwide.",
        }
        result = await call_fiddler_faithfulness(config, context)

    # Verify request format
    assert mock_session.post.called
    call_args = mock_session.post.call_args
    request_payload = call_args[1]["json"]
    assert "data" in request_payload
    assert request_payload["data"]["response"] == "Shipping takes 2 days"
    assert request_payload["data"]["context"] == "Shipping takes at least 3 days. We ship worldwide."
    # Verify old format (lists) is not present
    assert not isinstance(request_payload["data"].get("response"), list)
    assert not isinstance(request_payload["data"].get("context"), list)
