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

import logging
import os
from typing import Optional
from unittest.mock import patch

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import GenerationOptions
from tests.conftest import REASONING_TRACE_MOCK_PATH
from tests.utils import FakeLLMModel, clean_events, event_sequence_conforms


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
    llm = FakeLLMModel(
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
            "action_params": {"event": {"_type": "UserMessage", "text": "$user_message"}},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {"event": {"_type": "UserMessage", "text": "$user_message"}},
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
            "action_params": {"event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {"event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}},
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
            "action_params": {"event": {"_type": "UserMessage", "text": "$user_message"}},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {"event": {"_type": "UserMessage", "text": "$user_message"}},
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
            "action_params": {"event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {"event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}},
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
            "action_params": {"event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}},
            "action_result_key": None,
            "is_system_action": True,
            "source_uid": "NeMoGuardrails",
            "type": "StartInternalSystemAction",
        },
        {
            "action_name": "create_event",
            "action_params": {"event": {"_type": "StartUtteranceBotAction", "script": "$bot_message"}},
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
    llm = FakeLLMModel(
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
    return_value=FakeLLMModel(responses=["this should not be used"]),
)
async def test_llm_config_precedence(mock_init, llm_config_with_main):
    """Test that LLM provided via constructor takes precedence over config's main LLM."""
    injected_llm = FakeLLMModel(responses=["express greeting"])
    llm_rails = LLMRails(config=llm_config_with_main, llm=injected_llm)
    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    new_events = await llm_rails.runtime.generate_events(events)
    assert any(event.get("intent") == "express greeting" for event in new_events)
    assert not any(event.get("intent") == "this should not be used" for event in new_events)


@pytest.mark.asyncio
@patch(
    "nemoguardrails.rails.llm.llmrails.init_llm_model",
    return_value=FakeLLMModel(responses=["this should not be used"]),
)
async def test_llm_config_warning(mock_init, llm_config_with_main, caplog):
    """Test that a warning is logged when both constructor LLM and config main LLM are provided."""
    injected_llm = FakeLLMModel(responses=["express greeting"])
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
    return_value=FakeLLMModel(responses=["content safety response"]),
)
async def test_other_models_honored(mock_init, llm_config_with_multiple_models):
    """Test that other model configurations are still honored when main LLM is provided via constructor."""
    injected_llm = FakeLLMModel(responses=["express greeting"])
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

    injected_llm = FakeLLMModel(responses=["express greeting"])
    llm_rails = LLMRails(config=config, llm=injected_llm)
    assert llm_rails.llm is injected_llm

    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    new_events = await llm_rails.runtime.generate_events(events)
    assert any(event.get("intent") == "express greeting" for event in new_events)


@pytest.fixture
def llm_config_with_main_streaming_param():
    """Fixture providing a config whose main model carries a LangChain-only `streaming` flag."""
    return RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4",
                    "parameters": {"streaming": True},
                },
            ],
            "user_messages": {"express greeting": ["Hello!"]},
            "flows": [
                {
                    "elements": [
                        {"user": "express greeting"},
                        {"bot": "express greeting"},
                    ]
                },
            ],
            "bot_messages": {"express greeting": ["Hello!"]},
        }
    )


@pytest.mark.asyncio
async def test_compat_check_skips_main_when_constructor_llm_provided(llm_config_with_main_streaming_param):
    """When self.llm is injected, the compat validator must skip the ignored `main` config entry."""
    injected_llm = FakeLLMModel(responses=["express greeting"])
    LLMRails(config=llm_config_with_main_streaming_param, llm=injected_llm)


@pytest.mark.asyncio
async def test_compat_check_runs_against_main_when_no_constructor_llm(llm_config_with_main_streaming_param):
    """When no constructor LLM is provided, the validator runs against `main` and raises."""
    with pytest.raises(ValueError, match=r"streaming"):
        LLMRails(config=llm_config_with_main_streaming_param)


@pytest.mark.asyncio
@patch(
    "nemoguardrails.rails.llm.llmrails.init_llm_model",
    return_value=FakeLLMModel(responses=["safe"]),
)
async def test_main_llm_from_config_registered_as_action_param(mock_init, llm_config_with_main):
    """Test that main LLM initialized from config is properly registered as action parameter.

    This test ensures that when no LLM is provided via constructor and the main LLM
    is initialized from the config, it gets properly registered as an action parameter.
    This prevents the regression where actions expecting an 'llm' parameter would receive None.
    """
    from nemoguardrails.actions import action
    from nemoguardrails.types import LLMModel

    @action(name="test_llm_action")
    async def test_llm_action(llm: LLMModel):
        assert llm is not None
        assert hasattr(llm, "generate_async")
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
        if event["type"] == "InternalSystemActionFinished" and event["action_name"] == "test_llm_action":
            action_finished_event = event
            break

    assert action_finished_event is not None
    assert action_finished_event["status"] == "success"
    assert action_finished_event["return_value"] == "llm_action_success"


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
@patch.dict(os.environ, {"TEST_OPENAI_KEY": "secret-api-key-from-env"})
def test_api_key_environment_variable_passed_to_init_llm_model(mock_init_llm_model):
    """Test that API keys from environment variables are passed to init_llm_model."""
    mock_llm = FakeLLMModel(responses=["response"])
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
    mock_main_llm = FakeLLMModel(responses=["main response"])
    mock_content_safety_llm = FakeLLMModel(responses=["safety response"])

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
    mock_llm = FakeLLMModel(responses=["response"])
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
    rails = LLMRails(config=config, llm=FakeLLMModel(responses=[]))

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


def test_register_methods_return_self():
    """Test that all register_* methods return self for method chaining."""
    config = RailsConfig.from_content(config={"models": []})
    rails = LLMRails(config=config, llm=FakeLLMModel(responses=[]))

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

    result = rails.register_embedding_search_provider("dummy_provider", DummyEmbeddingProvider)
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
    rails = LLMRails(config=config, llm=FakeLLMModel(responses=[]))

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

    fake_llm = FakeLLMModel(responses=["express greeting"])
    config = RailsConfig.from_content(config={"models": []})
    rails = LLMRails(config=config, llm=fake_llm)
    rails.generate(messages=[{"role": "user", "content": "Hi!"}])

    rails._explain_info = None
    info = rails.explain()
    assert info == ExplainInfo()
    assert rails._explain_info == ExplainInfo()


@patch("nemoguardrails.rails.llm.llmrails.init_llm_model")
def test_cache_initialization_disabled_by_default(mock_init_llm_model):
    mock_llm = FakeLLMModel(responses=["response"])
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

    mock_llm = FakeLLMModel(responses=["response"])
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

    mock_llm = FakeLLMModel(responses=["response"])
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

    mock_llm = FakeLLMModel(responses=["response"])
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

    mock_llm = FakeLLMModel(responses=["response"])
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

    mock_llm = FakeLLMModel(responses=["response"])
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


@pytest.mark.asyncio
async def test_generate_async_reasoning_content_field_passthrough():
    from nemoguardrails.rails.llm.options import GenerationOptions

    test_reasoning_trace = "Let me think about this step by step..."

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": []})
        llm = FakeLLMModel(responses=["The answer is 42"])
        llm_rails = LLMRails(config=config, llm=llm)

        result = await llm_rails.generate_async(
            messages=[{"role": "user", "content": "What is the answer?"}],
            options=GenerationOptions(),
        )

        assert result.reasoning_content == test_reasoning_trace
        assert isinstance(result.response, list)
        assert result.response[0]["content"] == "The answer is 42"


@pytest.mark.asyncio
async def test_generate_async_reasoning_content_none():
    from nemoguardrails.rails.llm.options import GenerationOptions

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = None

        config = RailsConfig.from_content(config={"models": []})
        llm = FakeLLMModel(responses=["Regular response"])
        llm_rails = LLMRails(config=config, llm=llm)

        result = await llm_rails.generate_async(
            messages=[{"role": "user", "content": "Hello"}],
            options=GenerationOptions(),
        )

        assert result.reasoning_content is None
        assert isinstance(result.response, list)
        assert result.response[0]["content"] == "Regular response"


@pytest.mark.asyncio
async def test_generate_async_reasoning_not_in_response_content():
    from nemoguardrails.rails.llm.options import GenerationOptions

    test_reasoning_trace = "Let me analyze this carefully..."

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": []})
        llm = FakeLLMModel(responses=["The answer is 42"])
        llm_rails = LLMRails(config=config, llm=llm)

        result = await llm_rails.generate_async(
            messages=[{"role": "user", "content": "What is the answer?"}],
            options=GenerationOptions(),
        )

        assert result.reasoning_content == test_reasoning_trace
        assert test_reasoning_trace not in result.response[0]["content"]
        assert result.response[0]["content"] == "The answer is 42"


@pytest.mark.asyncio
async def test_generate_async_reasoning_with_thinking_tags():
    test_reasoning_trace = "Step 1: Analyze\nStep 2: Respond"

    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = test_reasoning_trace

        config = RailsConfig.from_content(config={"models": [], "passthrough": True})
        llm = FakeLLMModel(responses=["The answer is 42"])
        llm_rails = LLMRails(config=config, llm=llm)

        result = await llm_rails.generate_async(messages=[{"role": "user", "content": "What is the answer?"}])

        expected_prefix = f"<think>{test_reasoning_trace}</think>\n"
        assert result["content"].startswith(expected_prefix)
        assert "The answer is 42" in result["content"]


@pytest.mark.asyncio
async def test_generate_async_no_thinking_tags_when_no_reasoning():
    with patch(REASONING_TRACE_MOCK_PATH) as mock_get_reasoning:
        mock_get_reasoning.return_value = None

        config = RailsConfig.from_content(config={"models": []})
        llm = FakeLLMModel(responses=["Regular response"])
        llm_rails = LLMRails(config=config, llm=llm)

        result = await llm_rails.generate_async(messages=[{"role": "user", "content": "Hello"}])

        assert not result["content"].startswith("<think>")
        assert result["content"] == "Regular response"


EMBEDDING_MODEL_CONFIG_BASE = {
    "models": [
        {"type": "main", "engine": "fake", "model": "fake"},
        {"type": "embeddings", "engine": "SentenceTransformers", "model": "intfloat/e5-large-v2"},
    ],
    "user_messages": {"express greeting": ["Hello!"]},
    "flows": [{"elements": [{"user": "express greeting"}, {"bot": "express greeting"}]}],
    "bot_messages": {"express greeting": ["Hello! How are you?"]},
}


def test_embedding_model_backfills_search_provider_parameters():
    config = RailsConfig.parse_object(EMBEDDING_MODEL_CONFIG_BASE)

    assert "embedding_model" not in config.core.embedding_search_provider.parameters
    assert "embedding_model" not in config.knowledge_base.embedding_search_provider.parameters

    rails = LLMRails(config=config, llm=FakeLLMModel(responses=["  express greeting"]))

    assert rails.config.core.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"
    assert rails.config.core.embedding_search_provider.parameters["embedding_engine"] == "SentenceTransformers"
    assert rails.config.knowledge_base.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"
    assert (
        rails.config.knowledge_base.embedding_search_provider.parameters["embedding_engine"] == "SentenceTransformers"
    )


def test_embedding_model_does_not_overwrite_explicit_parameters():
    config = RailsConfig.parse_object(
        {
            **EMBEDDING_MODEL_CONFIG_BASE,
            "core": {
                "embedding_search_provider": {
                    "name": "default",
                    "parameters": {"embedding_model": "my-core-model", "embedding_engine": "MyEngine"},
                }
            },
            "knowledge_base": {
                "embedding_search_provider": {
                    "name": "default",
                    "parameters": {"embedding_model": "my-kb-model", "embedding_engine": "MyKBEngine"},
                }
            },
        }
    )

    rails = LLMRails(config=config, llm=FakeLLMModel(responses=["  express greeting"]))

    assert rails.config.core.embedding_search_provider.parameters["embedding_model"] == "my-core-model"
    assert rails.config.core.embedding_search_provider.parameters["embedding_engine"] == "MyEngine"
    assert rails.config.knowledge_base.embedding_search_provider.parameters["embedding_model"] == "my-kb-model"
    assert rails.config.knowledge_base.embedding_search_provider.parameters["embedding_engine"] == "MyKBEngine"


def test_embedding_model_partial_backfill_only_fills_missing():
    config = RailsConfig.parse_object(
        {
            **EMBEDDING_MODEL_CONFIG_BASE,
            "core": {
                "embedding_search_provider": {
                    "name": "default",
                    "parameters": {"embedding_model": "my-core-model"},
                }
            },
            "knowledge_base": {
                "embedding_search_provider": {
                    "name": "default",
                    "parameters": {"embedding_engine": "MyKBEngine"},
                }
            },
        }
    )

    rails = LLMRails(config=config, llm=FakeLLMModel(responses=["  express greeting"]))

    assert rails.config.core.embedding_search_provider.parameters["embedding_model"] == "my-core-model"
    assert rails.config.core.embedding_search_provider.parameters["embedding_engine"] == "SentenceTransformers"
    assert rails.config.knowledge_base.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"
    assert rails.config.knowledge_base.embedding_search_provider.parameters["embedding_engine"] == "MyKBEngine"


def test_embedding_model_no_backfill_for_custom_provider():
    config = RailsConfig.parse_object(
        {
            **EMBEDDING_MODEL_CONFIG_BASE,
            "core": {
                "embedding_search_provider": {
                    "name": "custom",
                    "parameters": {"some_param": "value"},
                }
            },
        }
    )

    rails = LLMRails(config=config, llm=FakeLLMModel(responses=["  express greeting"]))

    assert "embedding_model" not in rails.config.core.embedding_search_provider.parameters
    assert "embedding_engine" not in rails.config.core.embedding_search_provider.parameters
    assert rails.config.core.embedding_search_provider.parameters["some_param"] == "value"

    assert rails.config.knowledge_base.embedding_search_provider.parameters["embedding_model"] == "intfloat/e5-large-v2"
    assert (
        rails.config.knowledge_base.embedding_search_provider.parameters["embedding_engine"] == "SentenceTransformers"
    )


def test_embedding_model_no_backfill_when_no_embeddings_model():
    config = RailsConfig.parse_object(
        {
            "models": [{"type": "main", "engine": "fake", "model": "fake"}],
            "user_messages": {"express greeting": ["Hello!"]},
            "flows": [{"elements": [{"user": "express greeting"}, {"bot": "express greeting"}]}],
            "bot_messages": {"express greeting": ["Hello! How are you?"]},
        }
    )

    rails = LLMRails(config=config, llm=FakeLLMModel(responses=["  express greeting"]))

    assert "embedding_model" not in rails.config.core.embedding_search_provider.parameters
    assert "embedding_engine" not in rails.config.core.embedding_search_provider.parameters
    assert "embedding_model" not in rails.config.knowledge_base.embedding_search_provider.parameters
    assert "embedding_engine" not in rails.config.knowledge_base.embedding_search_provider.parameters


@pytest.fixture
def no_main_llm_config():
    return RailsConfig.from_content(
        """
        define flow input rail
          if $user_message == "block"
            bot refuse to respond
            stop

        define flow output rail
          if $bot_message == "block output"
            bot refuse to respond
            stop
        """,
        """
        rails:
            input:
                flows:
                    - input rail
            output:
                flows:
                    - output rail
        """,
    )


def _count_no_llm_warnings(caplog):
    return sum(
        1 for record in caplog.records if record.levelno == logging.WARNING and NO_MAIN_LLM_WARNING in record.message
    )


USER_MSG = [{"role": "user", "content": "hello"}]
USER_ASSISTANT_MSG = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]

NO_MAIN_LLM_WARNING = "No main LLM specified in the config and no LLM provided via constructor."


class TestGenerateAsyncNoMainLLMWarning:
    @pytest.mark.parametrize(
        "options, has_llm, messages, expected_warnings",
        [
            pytest.param(None, False, USER_MSG, 1, id="no-options-warns"),
            pytest.param(GenerationOptions(), False, USER_MSG, 1, id="default-options-warns"),
            pytest.param({"rails": ["input", "dialog"]}, False, USER_MSG, 1, id="dialog-in-list-warns"),
            pytest.param(
                {"rails": ["input", "output", "retrieval", "dialog"]}, False, USER_MSG, 1, id="all-rails-warns"
            ),
            pytest.param({"rails": ["input"]}, False, USER_MSG, 0, id="input-only-no-warn"),
            pytest.param({"rails": ["output"]}, False, USER_ASSISTANT_MSG, 0, id="output-only-no-warn"),
            pytest.param({"rails": ["input", "output"]}, False, USER_ASSISTANT_MSG, 0, id="input-output-no-warn"),
            pytest.param(None, True, USER_MSG, 0, id="llm-no-options-no-warn"),
            pytest.param({"rails": ["input", "dialog"]}, True, USER_MSG, 0, id="llm-dialog-enabled-no-warn"),
        ],
    )
    @pytest.mark.asyncio
    async def test_warning_behavior(self, no_main_llm_config, caplog, options, has_llm, messages, expected_warnings):
        llm = FakeLLMModel(responses=["Hello!"]) if has_llm else None
        rails = LLMRails(no_main_llm_config, llm=llm)
        with caplog.at_level(logging.WARNING, logger="nemoguardrails.rails.llm.llmrails"):
            if expected_warnings > 0:
                with pytest.raises(Exception):
                    await rails.generate_async(messages=messages, options=options)
            else:
                await rails.generate_async(messages=messages, options=options)
        assert _count_no_llm_warnings(caplog) == expected_warnings
