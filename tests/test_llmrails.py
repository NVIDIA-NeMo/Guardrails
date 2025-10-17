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
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.llmrails import get_action_details_from_flow_id
from tests.utils import FakeLLM, clean_events, event_sequence_conforms


@pytest.fixture
def rails_config():
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
            "user_messages": {
                "express greeting": ["Hello!"],
                "ask math question": ["What is 2 + 2?", "5 + 9"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
                {
                    "elements": [
                        {"user": "ask math question"},
                        {"execute": "compute"},
                        {"bot": "provide math response"},
                        {"bot": "ask if user happy"},
                    ]
                },
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
                "provide response": ["The answer is 234", "The answer is 1412"],
            },
        }
    )


@pytest.mark.asyncio
async def test_1(rails_config):
    llm = FakeLLM(
        responses=[
            "  express greeting",
            "  ask math question",
            '  "The answer is 5"',
            '  "Are you happy with the result?"',
        ]
    )

    async def compute(context: dict, what: Optional[str] = "2 + 3"):
        return eval(what)

    llm_rails = LLMRails(config=rails_config, llm=llm)
    llm_rails.runtime.register_action(compute)

    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]

    new_events = await llm_rails.runtime.generate_events(events)
    clean_events(new_events)

    expected_events = [
        {
            "data": {"user_message": "Hello!"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "Hello!",
                    "type": "UserMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "Hello!",
            "type": "UserMessage",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "intent": "express greeting",
                    "source_uid": "NeMoGuardrails",
                    "type": "UserIntent",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "intent": "express greeting",
            "source_uid": "NeMoGuardrails",
            "type": "UserIntent",
        },
        {
            "intent": "express greeting",
            "source_uid": "NeMoGuardrails",
            "type": "BotIntent",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"relevant_chunks": "\n"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "events": None,
            "is_success": True,
            "is_system_action": True,
            "return_value": "\n",
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"skip_output_rails": True},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "Hello! How are you?",
                    "type": "BotMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "Hello! How are you?",
            "type": "BotMessage",
        },
        {
            "data": {"bot_message": "Hello! How are you?", "skip_output_rails": False},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "action_info_modality": "bot_speech",
                    "action_info_modality_policy": "replace",
                    "script": "Hello! How are you?",
                    "source_uid": "NeMoGuardrails",
                    "type": "StartUtteranceBotAction",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_info_modality": "bot_speech",
            "action_info_modality_policy": "replace",
            "script": "Hello! How are you?",
            "source_uid": "NeMoGuardrails",
            "type": "StartUtteranceBotAction",
        },
        {
            "source_uid": "NeMoGuardrails",
            "type": "Listen",
        },
    ]

    # assert expected_events == new_events

    assert event_sequence_conforms(expected_events, new_events)

    events.extend(new_events)
    events.append({"type": "UtteranceUserActionFinished", "final_transcript": "2 + 3"})

    new_events = await llm_rails.runtime.generate_events(events)
    clean_events(new_events)

    expected_events = [
        {
            "data": {"user_message": "2 + 3"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "UserMessage", "text": "$user_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "2 + 3",
                    "type": "UserMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "2 + 3",
            "type": "UserMessage",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_user_intent",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "intent": "ask math question",
                    "source_uid": "NeMoGuardrails",
                    "type": "UserIntent",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "intent": "ask math question",
            "source_uid": "NeMoGuardrails",
            "type": "UserIntent",
        },
        {
            "action_name": "compute",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": False,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "compute",
            "action_params": {},
            "action_result_key": None,
            "events": [],
            "is_success": True,
            "is_system_action": False,
            "return_value": 5,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "intent": "provide math response",
            "source_uid": "NeMoGuardrails",
            "type": "BotIntent",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"relevant_chunks": "\n\n"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "events": None,
            "is_success": True,
            "is_system_action": True,
            "return_value": "\n\n",
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "The answer is 5",
                    "type": "BotMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "The answer is 5",
            "type": "BotMessage",
        },
        {
            "data": {"bot_message": "The answer is 5"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "action_info_modality": "bot_speech",
                    "action_info_modality_policy": "replace",
                    "script": "The answer is 5",
                    "source_uid": "NeMoGuardrails",
                    "type": "StartUtteranceBotAction",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_info_modality": "bot_speech",
            "action_info_modality_policy": "replace",
            "script": "The answer is 5",
            "source_uid": "NeMoGuardrails",
            "type": "StartUtteranceBotAction",
        },
        {
            "intent": "ask if user happy",
            "source_uid": "NeMoGuardrails",
            "type": "BotIntent",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "data": {"relevant_chunks": "\n\n\n"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "retrieve_relevant_chunks",
            "action_params": {},
            "action_result_key": None,
            "events": None,
            "is_success": True,
            "is_system_action": True,
            "return_value": "\n\n\n",
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "generate_bot_message",
            "action_params": {},
            "action_result_key": None,
            "events": [
                {
                    "source_uid": "NeMoGuardrails",
                    "text": "Are you happy with the result?",
                    "type": "BotMessage",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "source_uid": "NeMoGuardrails",
            "text": "Are you happy with the result?",
            "type": "BotMessage",
        },
        {
            "data": {"bot_message": "Are you happy with the result?"},
            "source_uid": "NeMoGuardrails",
            "type": "ContextUpdate",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {
                "event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}
            },
            "action_result_key": None,
            "events": [
                {
                    "action_info_modality": "bot_speech",
                    "action_info_modality_policy": "replace",
                    "script": "Are you happy with the result?",
                    "source_uid": "NeMoGuardrails",
                    "type": "StartUtteranceBotAction",
                }
            ],
            "is_success": True,
            "is_system_action": True,
            "return_value": None,
            "source_uid": "NeMoGuardrails",
            "status": "success",
            "type": "InternalSystemActionFinished",
        },
        {
            "action_info_modality": "bot_speech",
            "action_info_modality_policy": "replace",
            "script": "Are you happy with the result?",
            "source_uid": "NeMoGuardrails",
            "type": "StartUtteranceBotAction",
        },
        {
            "source_uid": "NeMoGuardrails",
            "type": "Listen",
        },
    ]

    # assert expected_events == new_events
    assert event_sequence_conforms(expected_events, new_events)


@pytest.mark.asyncio
async def test_2(rails_config):
    llm = FakeLLM(
        responses=[
            "  express greeting",
            "  ask math question",
            '  "The answer is 5"',
            '  "Are you happy with the result?"',
        ]
    )

    async def compute(what: Optional[str] = "2 + 3"):
        return eval(what)

    llm_rails = LLMRails(config=rails_config, llm=llm)
    llm_rails.runtime.register_action(compute)

    messages = [{"role": "user", "content": "Hello!"}]
    bot_message = await llm_rails.generate_async(messages=messages)

    assert bot_message == {"role": "assistant", "content": "Hello! How are you?"}
    messages.append(bot_message)

    messages.append({"role": "user", "content": "2 + 3"})
    bot_message = await llm_rails.generate_async(messages=messages)
    assert bot_message == {
        "role": "assistant",
        "content": "The answer is 5\nAre you happy with the result?",
    }


@pytest.fixture
def llm_config_with_main():
    """Fixture providing a basic config with a main LLM."""
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
            "user_messages": {
                "express greeting": ["Hello!"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
            },
        }
    )


@pytest.mark.asyncio
@patch(
    "nemoguardrails.rails.llm.llmrails.init_llm_model",
    return_value=FakeLLM(responses=["this should not be used"]),
)
async def test_llm_config_precedence(mock_init, llm_config_with_main):
    """Test that LLM provided via constructor takes precedence over config's main LLM."""
    injected_llm = FakeLLM(responses=["express greeting"])
    llm_rails = LLMRails(config=llm_config_with_main, llm=injected_llm)
    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    new_events = await llm_rails.runtime.generate_events(events)
    assert any(event.get("intent") == "express greeting" for event in new_events)
    assert not any(
        event.get("intent") == "this should not be used" for event in new_events
    )


@pytest.mark.asyncio
@patch(
    "nemoguardrails.rails.llm.llmrails.init_llm_model",
    return_value=FakeLLM(responses=["this should not be used"]),
)
async def test_llm_config_warning(mock_init, llm_config_with_main, caplog):
    """Test that a warning is logged when both constructor LLM and config main LLM are provided."""
    injected_llm = FakeLLM(responses=["express greeting"])
    caplog.clear()
    _ = LLMRails(config=llm_config_with_main, llm=injected_llm)
    warning_msg = "Both an LLM was provided via constructor and a main LLM is specified in the config"
    assert any(warning_msg in record.message for record in caplog.records)


@pytest.fixture
def llm_config_with_multiple_models():
    """Fixture providing a config with main LLM and content safety model."""
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                },
                {
                    "type": "content_safety",
                    "engine": "fake",
                    "model": "fake",
                },
            ],
            "user_messages": {
                "express greeting": ["Hello!"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
            },
        }
    )


@pytest.mark.asyncio
@patch(
    "nemoguardrails.rails.llm.llmrails.init_llm_model",
    return_value=FakeLLM(responses=["content safety response"]),
)
async def test_other_models_honored(mock_init, llm_config_with_multiple_models):
    """Test that other model configurations are still honored when main LLM is provided via constructor."""
    injected_llm = FakeLLM(responses=["express greeting"])
    llm_rails = LLMRails(config=llm_config_with_multiple_models, llm=injected_llm)
    assert hasattr(llm_rails, "content_safety_llm")
    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    new_events = await llm_rails.runtime.generate_events(events)
    assert any(event.get("intent") == "express greeting" for event in new_events)


@pytest.mark.asyncio
async def test_llm_constructor_with_empty_models_config():
    """Test that LLMRails can be initialized with constructor LLM when config has empty models list.

    This tests the fix for the IndexError that occurred when providing an LLM via constructor
    but having an empty models list in the config.
    """
    config = RailsConfig.parse_object(
        {
            "models": [],
            "user_messages": {
                "express greeting": ["Hello!"],
            },
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
            ],
            "bot_messages": {
                "express greeting": ["Hello! How are you?"],
            },
        }
    )

    injected_llm = FakeLLM(responses=["express greeting"])
    llm_rails = LLMRails(config=config, llm=injected_llm)
    assert llm_rails.llm == injected_llm

    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    new_events = await llm_rails.runtime.generate_events(events)
    assert any(event.get("intent") == "express greeting" for event in new_events)


@pytest.mark.asyncio
@patch(
    "nemoguardrails.rails.llm.llmrails.init_llm_model",
    return_value=FakeLLM(responses=["safe"]),
)
async def test_main_llm_from_config_registered_as_action_param(
    mock_init, llm_config_with_main
):
    """Test that main LLM initialized from config is properly registered as action parameter.

    This test ensures that when no LLM is provided via constructor and the main LLM
    is initialized from the config, it gets properly registered as an action parameter.
    This prevents the regression where actions expecting an 'llm' parameter would receive None.
    """
    from langchain_core.language_models.llms import BaseLLM

    from nemoguardrails.actions import action

    @action(name="test_llm_action")
    async def test_llm_action(llm: BaseLLM):
        assert llm is not None
        assert hasattr(llm, "agenerate_prompt")
        return "llm_action_success"

    llm_rails = LLMRails(config=llm_config_with_main)

    llm_rails.runtime.register_action(test_llm_action)

    assert llm_rails.llm is not None
    assert "llm" in llm_rails.runtime.registered_action_params
    assert llm_rails.runtime.registered_action_params["llm"] is llm_rails.llm

    # create events that trigger the test action through the public generate_events_async method
    events = [
        {"type": "UtteranceUserActionFinished", "final_transcript": "test"},
        {
            "type": "StartInternalSystemAction",
            "action_name": "test_llm_action",
            "action_params": {},
            "action_result_key": None,
            "action_uid": "test_action_uid",
            "is_system_action": False,
            "source_uid": "test",
        },
    ]

    result_events = await llm_rails.generate_events_async(events)

    action_finished_event = None
    for event in result_events:
        if (
            event["type"] == "InternalSystemActionFinished"
            and event["action_name"] == "test_llm_action"
        ):
            action_finished_event = event
            break

    assert action_finished_event is not None
    assert action_finished_event["status"] == "success"
    assert action_finished_event["return_value"] == "llm_action_success"


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
@patch.dict(os.environ, {"TEST_OPENAI_KEY": "secret-api-key-from-env"})
def test_api_key_environment_variable_passed_to_init_llm_model(mock_init_llm_model):
    """Test that API keys from environment variables are passed to init_llm_model."""
    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="openai",
                model="gpt-3.5-turbo",
                api_key_env_var="TEST_OPENAI_KEY",
                parameters={"temperature": 0.7},
            )
        ]
    )

    rails = LLMRails(config=config, verbose=False)

    mock_init_llm_model.assert_called_once()
    call_args = mock_init_llm_model.call_args

    # critical assertion: the kwargs should contain the API key from the environment
    # before the fix, this assertion would FAIL because api_key wouldnt be in kwargs
    assert call_args.kwargs["kwargs"]["api_key"] == "secret-api-key-from-env"
    assert call_args.kwargs["kwargs"]["temperature"] == 0.7

    assert call_args.kwargs["model_name"] == "gpt-3.5-turbo"
    assert call_args.kwargs["provider_name"] == "openai"
    assert call_args.kwargs["mode"] == "chat"


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
@patch.dict(os.environ, {"CONTENT_SAFETY_KEY": "safety-key-from-env"})
def test_api_key_environment_variable_for_non_main_models(mock_init_llm_model):
    """Test that API keys from environment variables work for non-main models too.

    This test ensures the fix works for all model types, not just the main model.
    """
    mock_main_llm = FakeLLM(responses=["main response"])
    mock_content_safety_llm = FakeLLM(responses=["safety response"])

    mock_init_llm_model.side_effect = [mock_main_llm, mock_content_safety_llm]

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="openai",
                model="gpt-3.5-turbo",
                parameters={"api_key": "hardcoded-key"},
            ),
            Model(
                type="content_safety",
                engine="openai",
                model="text-moderation-latest",
                api_key_env_var="CONTENT_SAFETY_KEY",
                parameters={"temperature": 0.0},
            ),
        ]
    )

    _ = LLMRails(config=config, verbose=False)

    assert mock_init_llm_model.call_count == 2

    main_call_args = mock_init_llm_model.call_args_list[0]
    assert main_call_args.kwargs["kwargs"]["api_key"] == "hardcoded-key"

    safety_call_args = mock_init_llm_model.call_args_list[1]
    assert safety_call_args.kwargs["kwargs"]["api_key"] == "safety-key-from-env"
    assert safety_call_args.kwargs["kwargs"]["temperature"] == 0.0


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_missing_api_key_environment_variable_graceful_handling(mock_init_llm_model):
    """Test that missing environment variables are handled gracefully during LLM initialization.

    This test ensures that when an api_key_env_var is specified but the environment
    variable doesn't exist during LLM initialization, the system doesn't crash and
    doesn't pass a None/empty API key.
    """
    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    with patch.dict(os.environ, {"TEMP_API_KEY": "temporary-key"}):
        config = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="openai",
                    model="gpt-3.5-turbo",
                    api_key_env_var="TEMP_API_KEY",
                    parameters={"temperature": 0.5},
                )
            ]
        )

    with patch.dict(os.environ, {}, clear=True):
        _ = LLMRails(config=config, verbose=False)

        mock_init_llm_model.assert_called_once()
        call_args = mock_init_llm_model.call_args

        assert "api_key" not in call_args.kwargs["kwargs"]
        assert call_args.kwargs["kwargs"]["temperature"] == 0.5


def test_api_key_environment_variable_logic_without_rails_init():
    """Test the _prepare_model_kwargs method directly to isolate the logic.

    This test shows that the extracted helper method works correctly
    """
    config = RailsConfig(models=[Model(type="main", engine="fake", model="fake")])
    rails = LLMRails(config=config, llm=FakeLLM(responses=[]))

    # case 1: env var exists
    class ModelWithEnvVar:
        def __init__(self):
            self.api_key_env_var = "MY_API_KEY"
            self.parameters = {"temperature": 0.8}

    with patch.dict(os.environ, {"MY_API_KEY": "my-secret-key"}):
        model = ModelWithEnvVar()
        kwargs = rails._prepare_model_kwargs(model)

        assert kwargs["api_key"] == "my-secret-key"
        assert kwargs["temperature"] == 0.8

    # case 2: env var doesn't exist
    with patch.dict(os.environ, {}, clear=True):
        model = ModelWithEnvVar()
        kwargs = rails._prepare_model_kwargs(model)

        assert "api_key" not in kwargs
        assert kwargs["temperature"] == 0.8

    # case 3: no api_key_env_var specified
    class ModelWithoutEnvVar:
        def __init__(self):
            self.api_key_env_var = None
            self.parameters = {"api_key": "direct-key", "temperature": 0.3}

    model = ModelWithoutEnvVar()
    kwargs = rails._prepare_model_kwargs(model)

    assert kwargs["api_key"] == "direct-key"
    assert kwargs["temperature"] == 0.3


@pytest.mark.asyncio
@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
async def test_stream_usage_enabled_for_streaming_supported_providers(
    mock_init_llm_model,
):
    """Test that stream_usage=True is set when streaming is enabled for supported providers."""
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4",
                }
            ],
            "streaming": True,
        }
    )

    LLMRails(config=config)

    mock_init_llm_model.assert_called_once()
    call_args = mock_init_llm_model.call_args
    kwargs = call_args.kwargs.get("kwargs", {})

    assert kwargs.get("stream_usage") is True


@pytest.mark.asyncio
@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
async def test_stream_usage_not_set_without_streaming(mock_init_llm_model):
    """Test that stream_usage is not set when streaming is disabled."""
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4",
                }
            ],
            "streaming": False,
        }
    )

    LLMRails(config=config)

    mock_init_llm_model.assert_called_once()
    call_args = mock_init_llm_model.call_args
    kwargs = call_args.kwargs.get("kwargs", {})

    assert "stream_usage" not in kwargs


@pytest.mark.asyncio
@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
async def test_stream_usage_enabled_for_all_providers_when_streaming(
    mock_init_llm_model,
):
    """Test that stream_usage is passed to ALL providers when streaming is enabled.

    With the new design, stream_usage=True is passed to ALL providers when
    streaming is enabled. Providers that don't support it will simply ignore it.
    """
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "unsupported",
                    "model": "whatever",
                }
            ],
            "streaming": True,
        }
    )

    LLMRails(config=config)

    mock_init_llm_model.assert_called_once()
    call_args = mock_init_llm_model.call_args
    kwargs = call_args.kwargs.get("kwargs", {})

    # stream_usage should be set for all providers when streaming is enabled
    assert kwargs.get("stream_usage") is True


# Add this test after the existing tests, around line 1100+


def test_register_methods_return_self():
    """Test that all register_* methods return self for method chaining."""
    config = RailsConfig.from_content(config={"models": []})
    rails = LLMRails(config=config, llm=FakeLLM(responses=[]))

    # Test register_action returns self
    def dummy_action():
        pass

    result = rails.register_action(dummy_action, "test_action")
    assert result is rails, "register_action should return self"

    # Test register_action_param returns self
    result = rails.register_action_param("test_param", "test_value")
    assert result is rails, "register_action_param should return self"

    # Test register_filter returns self
    def dummy_filter(text):
        return text

    result = rails.register_filter(dummy_filter, "test_filter")
    assert result is rails, "register_filter should return self"

    # Test register_output_parser returns self
    def dummy_parser(text):
        return text

    result = rails.register_output_parser(dummy_parser, "test_parser")
    assert result is rails, "register_output_parser should return self"

    # Test register_prompt_context returns self
    result = rails.register_prompt_context("test_context", "test_value")
    assert result is rails, "register_prompt_context should return self"

    # Test register_embedding_search_provider returns self
    from nemoguardrails.embeddings.index import EmbeddingsIndex

    class DummyEmbeddingProvider(EmbeddingsIndex):
        def __init__(self, **kwargs):
            pass

        def build(self):
            pass

        def search(self, text, max_results=5):
            return []

    result = rails.register_embedding_search_provider(
        "dummy_provider", DummyEmbeddingProvider
    )
    assert result is rails, "register_embedding_search_provider should return self"

    # Test register_embedding_provider returns self
    from nemoguardrails.embeddings.providers.base import EmbeddingModel

    class DummyEmbeddingModel(EmbeddingModel):
        def encode(self, texts):
            return []

    result = rails.register_embedding_provider(DummyEmbeddingModel, "dummy_embedding")
    assert result is rails, "register_embedding_provider should return self"


def test_method_chaining():
    """Test that method chaining works correctly with register_* methods."""
    config = RailsConfig.from_content(config={"models": []})
    rails = LLMRails(config=config, llm=FakeLLM(responses=[]))

    def dummy_action():
        return "action_result"

    def dummy_filter(text):
        return text.upper()

    def dummy_parser(text):
        return {"parsed": text}

    # Test chaining multiple register methods
    result = (
        rails.register_action(dummy_action, "chained_action")
        .register_action_param("chained_param", "param_value")
        .register_filter(dummy_filter, "chained_filter")
        .register_output_parser(dummy_parser, "chained_parser")
        .register_prompt_context("chained_context", "context_value")
    )

    assert result is rails, "Method chaining should return the same rails instance"

    # Verify that all registrations actually worked
    assert "chained_action" in rails.runtime.action_dispatcher.registered_actions
    assert "chained_param" in rails.runtime.registered_action_params
    assert rails.runtime.registered_action_params["chained_param"] == "param_value"


def test_explain_calls_ensure_explain_info():
    """Make sure if no `explain_info` attribute is present in LLMRails it's populated with
    an empty ExplainInfo object"""

    mock_llm = MagicMock(spec=BaseChatModel)
    config = RailsConfig.from_content(config={"models": []})
    rails = LLMRails(config=config, llm=mock_llm)
    rails.generate(messages=[{"role": "user", "content": "Hi!"}])

    rails.explain_info = None
    info = rails.explain()
    assert info == ExplainInfo()
    assert rails.explain_info == ExplainInfo()


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_initialization_disabled_by_default(mock_init_llm_model):
    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
            ),
        ]
    )

    rails = LLMRails(config=config, verbose=False)
    model_caches = rails.runtime.registered_action_params.get("model_caches")

    assert model_caches is None or len(model_caches) == 0


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_initialization_with_enabled_cache(mock_init_llm_model):
    from nemoguardrails.rails.llm.config import CacheStatsConfig, ModelCacheConfig

    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(
                    enabled=True,
                    maxsize=1000,
                    stats=CacheStatsConfig(enabled=False),
                ),
            ),
        ]
    )

    rails = LLMRails(config=config, verbose=False)
    model_caches = rails.runtime.registered_action_params.get("model_caches", {})

    assert "content_safety" in model_caches
    assert model_caches["content_safety"] is not None
    assert model_caches["content_safety"].maxsize == 1000


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_not_created_for_main_and_embeddings_models(mock_init_llm_model):
    from nemoguardrails.rails.llm.config import ModelCacheConfig

    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=1000),
            ),
            Model(
                type="embeddings",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=1000),
            ),
        ]
    )

    rails = LLMRails(config=config, verbose=False)
    model_caches = rails.runtime.registered_action_params.get("model_caches", {})

    assert "main" not in model_caches
    assert "embeddings" not in model_caches


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_initialization_with_zero_maxsize_raises_error(mock_init_llm_model):
    from nemoguardrails.rails.llm.config import ModelCacheConfig

    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=0),
            ),
        ]
    )

    with pytest.raises(ValueError, match="Invalid cache maxsize"):
        LLMRails(config=config, verbose=False)


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_initialization_with_stats_enabled(mock_init_llm_model):
    from nemoguardrails.rails.llm.config import CacheStatsConfig, ModelCacheConfig

    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(
                    enabled=True,
                    maxsize=5000,
                    stats=CacheStatsConfig(enabled=True, log_interval=60.0),
                ),
            ),
        ]
    )

    rails = LLMRails(config=config, verbose=False)
    model_caches = rails.runtime.registered_action_params.get("model_caches", {})

    cache = model_caches["content_safety"]
    assert cache is not None
    assert cache.track_stats is True
    assert cache.stats_logging_interval == 60.0
    assert cache.supports_stats_logging() is True


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_initialization_with_multiple_models(mock_init_llm_model):
    from nemoguardrails.rails.llm.config import ModelCacheConfig

    mock_llm = FakeLLM(responses=["response"])
    mock_init_llm_model.return_value = mock_llm

    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="fake",
                model="fake",
            ),
            Model(
                type="content_safety",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=1000),
            ),
            Model(
                type="jailbreak_detection",
                engine="fake",
                model="fake",
                cache=ModelCacheConfig(enabled=True, maxsize=2000),
            ),
        ]
    )

    rails = LLMRails(config=config, verbose=False)
    model_caches = rails.runtime.registered_action_params.get("model_caches", {})

    assert "main" not in model_caches
    assert "content_safety" in model_caches
    assert "jailbreak_detection" in model_caches
    assert model_caches["content_safety"].maxsize == 1000
    assert model_caches["jailbreak_detection"].maxsize == 2000
