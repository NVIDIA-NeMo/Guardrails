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

from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain.schema import Generation, LLMResult
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration

from nemoguardrails.context import explain_info_var, llm_call_info_var, llm_stats_var
from nemoguardrails.logging.callbacks import LoggingCallbackHandler
from nemoguardrails.logging.explain import ExplainInfo, LLMCallInfo
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.logging.utils import extract_model_name_and_base_url


@pytest.mark.asyncio
async def test_token_usage_tracking_with_usage_metadata():
    """Test that token usage is tracked when usage_metadata is available (stream_usage=True scenario)."""

    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    # simulate the LLM response with usage metadata (as would happen with stream_usage=True)
    ai_message = AIMessage(
        content="Hello! How can I help you?",
        usage_metadata={"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
    )

    chat_generation = ChatGeneration(message=ai_message)
    llm_result = LLMResult(generations=[[chat_generation]])

    # call the on_llm_end method
    await handler.on_llm_end(llm_result, run_id=uuid4())

    assert llm_call_info.total_tokens == 16
    assert llm_call_info.prompt_tokens == 10
    assert llm_call_info.completion_tokens == 6

    assert llm_stats.get_stat("total_tokens") == 16
    assert llm_stats.get_stat("total_prompt_tokens") == 10
    assert llm_stats.get_stat("total_completion_tokens") == 6


@pytest.mark.asyncio
async def test_token_usage_tracking_with_llm_output_fallback():
    """Test token usage tracking with legacy llm_output format."""

    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    # simulate LLM response with token usage in llm_output (fallback scenario)
    generation = Generation(text="Fallback response")
    llm_result = LLMResult(
        generations=[[generation]],
        llm_output={
            "token_usage": {
                "total_tokens": 20,
                "prompt_tokens": 12,
                "completion_tokens": 8,
            }
        },
    )

    await handler.on_llm_end(llm_result, run_id=uuid4())

    assert llm_call_info.total_tokens == 20
    assert llm_call_info.prompt_tokens == 12
    assert llm_call_info.completion_tokens == 8

    assert llm_stats.get_stat("total_tokens") == 20
    assert llm_stats.get_stat("total_prompt_tokens") == 12
    assert llm_stats.get_stat("total_completion_tokens") == 8


@pytest.mark.asyncio
async def test_no_token_usage_tracking_without_metadata():
    """Test that no token usage is tracked when metadata is not available."""

    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    # simulate LLM response without usage metadata (stream_usage=False scenario)
    ai_message = AIMessage(content="Hello! How can I help you?")
    chat_generation = ChatGeneration(message=ai_message)
    llm_result = LLMResult(generations=[[chat_generation]])

    await handler.on_llm_end(llm_result, run_id=uuid4())

    assert llm_call_info.total_tokens is None or llm_call_info.total_tokens == 0
    assert llm_call_info.prompt_tokens is None or llm_call_info.prompt_tokens == 0
    assert (
        llm_call_info.completion_tokens is None or llm_call_info.completion_tokens == 0
    )


@pytest.mark.asyncio
async def test_multiple_generations_token_accumulation():
    """Test that token usage accumulates across multiple generations."""

    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    ai_message1 = AIMessage(
        content="First response",
        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    )

    ai_message2 = AIMessage(
        content="Second response",
        usage_metadata={"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
    )

    chat_generation1 = ChatGeneration(message=ai_message1)
    chat_generation2 = ChatGeneration(message=ai_message2)
    llm_result = LLMResult(generations=[[chat_generation1, chat_generation2]])

    await handler.on_llm_end(llm_result, run_id=uuid4())

    assert llm_call_info.total_tokens == 19  # 8 + 11
    assert llm_call_info.prompt_tokens == 12  # 5 + 7
    assert llm_call_info.completion_tokens == 7  # 3 + 4

    assert llm_stats.get_stat("total_tokens") == 19
    assert llm_stats.get_stat("total_prompt_tokens") == 12
    assert llm_stats.get_stat("total_completion_tokens") == 7


@pytest.mark.asyncio
async def test_tool_message_labeling_in_logging():
    """Test that tool messages are labeled as 'Tool' in logging output."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        SystemMessage(content="System message"),
        ToolMessage(content="Tool result", tool_call_id="test_tool_call"),
    ]

    with patch("nemoguardrails.logging.callbacks.log") as mock_log:
        await handler.on_chat_model_start(
            serialized={},
            messages=[messages],
            run_id=uuid4(),
        )

        mock_log.info.assert_called()

        logged_prompt = None
        for call in mock_log.info.call_args_list:
            if "Prompt Messages" in str(call):
                logged_prompt = call[0][1]
                break

        assert logged_prompt is not None
        assert "[cyan]User[/]" in logged_prompt
        assert "[cyan]Bot[/]" in logged_prompt
        assert "[cyan]System[/]" in logged_prompt
        assert "[cyan]Tool[/]" in logged_prompt


@pytest.mark.asyncio
async def test_unknown_message_type_labeling():
    """Test that unknown message types display their actual type name."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    class CustomMessage(BaseMessage):
        def __init__(self, content, msg_type):
            super().__init__(content=content, type=msg_type)

    messages: list[BaseMessage] = [
        CustomMessage("Custom message", "custom"),
        CustomMessage("Function message", "function"),
    ]

    with patch("nemoguardrails.logging.callbacks.log") as mock_log:
        await handler.on_chat_model_start(
            serialized={},
            messages=[messages],
            run_id=uuid4(),
        )

        mock_log.info.assert_called()

        logged_prompt = None
        for call in mock_log.info.call_args_list:
            if "Prompt Messages" in str(call):
                logged_prompt = call[0][1]
                break

        assert logged_prompt is not None
        assert "[cyan]Custom[/]" in logged_prompt
        assert "[cyan]Function[/]" in logged_prompt


def test_extract_model_and_url_from_kwargs():
    """Test extracting model_name and openai_api_base from kwargs (ChatOpenAI case)."""
    serialized = {
        "kwargs": {
            "model_name": "gpt-4",
            "openai_api_base": "https://api.openai.com/v1",
            "temperature": 0.7,
        }
    }

    model_name, base_url = extract_model_name_and_base_url(serialized)

    assert model_name == "gpt-4"
    assert base_url == "https://api.openai.com/v1"


def test_extract_model_and_url_from_repr():
    """Test extracting from repr string (ChatNIM case)."""
    # Property values in single-quotes
    serialized = {
        "kwargs": {"temperature": 0.1},
        "repr": "ChatNIM(model='meta/llama-3.3-70b-instruct', client=<openai.OpenAI object at 0x10d8e4e90>, endpoint_url='https://nim.int.aire.nvidia.com/v1')",
    }

    model_name, base_url = extract_model_name_and_base_url(serialized)

    assert model_name == "meta/llama-3.3-70b-instruct"
    assert base_url == "https://nim.int.aire.nvidia.com/v1"

    # Property values in double-quotes
    serialized = {
        "repr": 'ChatOpenAI(model="gpt-3.5-turbo", base_url="https://custom.api.com/v1")'
    }

    model_name, base_url = extract_model_name_and_base_url(serialized)

    assert model_name == "gpt-3.5-turbo"
    assert base_url == "https://custom.api.com/v1"

    # Model is stored in the `model_name` property
    serialized = {
        "repr": "SomeProvider(model_name='custom-model-v2', api_base='https://example.com')"
    }

    model_name, base_url = extract_model_name_and_base_url(serialized)

    assert model_name == "custom-model-v2"
    assert base_url == "https://example.com"


def test_extract_model_and_url_from_various_url_properties():
    """Test extracting various URL property names."""
    test_cases = [
        ("api_base='https://api1.com'", "https://api1.com"),
        ("api_host='https://api2.com'", "https://api2.com"),
        ("azure_endpoint='https://azure.com'", "https://azure.com"),
        ("endpoint='https://endpoint.com'", "https://endpoint.com"),
        ("openai_api_base='https://openai.com'", "https://openai.com"),
    ]

    for url_pattern, expected_url in test_cases:
        serialized = {"repr": f"Provider(model='test-model', {url_pattern})"}
        model_name, base_url = extract_model_name_and_base_url(serialized)
        assert base_url == expected_url, f"Failed for pattern: {url_pattern}"


def test_extract_model_and_url_kwargs_priority_over_repr():
    """Test that kwargs values, if present, take priority over repr values."""
    serialized = {
        "kwargs": {
            "model_name": "gpt-4-from-kwargs",
            "openai_api_base": "https://kwargs.api.com",
        },
        "repr": "ChatOpenAI(model='gpt-3.5-from-repr', base_url='https://repr.api.com')",
    }

    model_name, base_url = extract_model_name_and_base_url(serialized)

    assert model_name == "gpt-4-from-kwargs"
    assert base_url == "https://kwargs.api.com"


def test_extract_model_and_url_with_missing_values():
    """Test extraction when values are missing."""
    # No model or URL
    serialized = {"kwargs": {"temperature": 0.7}}
    model_name, base_url = extract_model_name_and_base_url(serialized)
    assert model_name is None
    assert base_url is None

    # Only model, no URL
    serialized = {"kwargs": {"model_name": "gpt-4"}}
    model_name, base_url = extract_model_name_and_base_url(serialized)
    assert model_name == "gpt-4"
    assert base_url is None

    # Only URL, no model
    serialized = {"repr": "Provider(endpoint_url='https://example.com')"}
    model_name, base_url = extract_model_name_and_base_url(serialized)
    assert model_name is None
    assert base_url == "https://example.com"


def test_extract_model_and_url_with_empty_values():
    """Test extraction when values are empty strings."""
    serialized = {"kwargs": {"model_name": "", "openai_api_base": ""}}
    model_name, base_url = extract_model_name_and_base_url(serialized)
    assert model_name is None
    assert base_url is None


def test_extract_model_and_url_with_empty_serialized_data():
    """Test extraction with empty or minimal serialized dict."""
    serialized = {}
    model_name, base_url = extract_model_name_and_base_url(serialized)
    assert model_name is None
    assert base_url is None
