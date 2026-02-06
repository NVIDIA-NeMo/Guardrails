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

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from nemoguardrails.server import api


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    """Set the rails_config_path to test configs and required env vars."""
    original_path = api.app.rails_config_path
    original_engine = os.environ.get("MAIN_MODEL_ENGINE")
    test_configs_path = os.path.join(os.path.dirname(__file__), "..", "test_configs")
    api.app.rails_config_path = test_configs_path
    os.environ["MAIN_MODEL_ENGINE"] = "custom_llm"
    api.llm_rails_instances.clear()
    yield
    api.app.rails_config_path = original_path
    api.llm_rails_instances.clear()
    if original_engine is not None:
        os.environ["MAIN_MODEL_ENGINE"] = original_engine
    else:
        os.environ.pop("MAIN_MODEL_ENGINE", None)


@pytest.fixture(scope="function")
def openai_client():
    """Create an OpenAI client that uses the guardrails FastAPI app via TestClient."""
    # Create a TestClient for the FastAPI app
    test_client = TestClient(api.app)

    client = OpenAI(
        api_key="dummy-key",
        base_url="http://dummy-url/v1",
        http_client=test_client,
    )
    return client


def test_openai_client_chat_completion(openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={"guardrails": {"config_id": "with_custom_llm"}},
    )

    assert isinstance(response, ChatCompletion)
    assert response.id is not None

    assert response.choices[0] == Choice(
        finish_reason="stop",
        index=0,
        logprobs=None,
        message=ChatCompletionMessage(
            content="Custom LLM response",
            refusal=None,
            role="assistant",
            annotations=None,
            audio=None,
            function_call=None,
            tool_calls=None,
        ),
    )
    assert hasattr(response, "created")


def test_openai_client_chat_completion_parameterized(openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
        extra_body={"guardrails": {"config_id": "with_custom_llm"}},
    )

    # Verify response exists
    assert isinstance(response, ChatCompletion)
    assert response.id is not None
    assert response.choices[0] == Choice(
        finish_reason="stop",
        index=0,
        logprobs=None,
        message=ChatCompletionMessage(
            content="Custom LLM response",
            refusal=None,
            role="assistant",
            annotations=None,
        ),
    )
    assert hasattr(response, "created")


def test_openai_client_chat_completion_input_rails(openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=False,
        extra_body={"guardrails": {"config_id": "with_input_rails"}},
    )

    assert isinstance(response, ChatCompletion)
    assert response.id is not None
    assert isinstance(response.choices[0], Choice)
    assert hasattr(response, "created")


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_openai_client_chat_completion_streaming(openai_client):
    stream = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a short joke."}],
        stream=True,
        extra_body={"guardrails": {"config_id": "input_rails"}},
    )

    chunks = list(stream)
    assert len(chunks) > 0

    has_content = any(hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content for chunk in chunks)
    assert has_content, "At least one chunk should contain content"


def test_openai_client_error_handling_invalid_model(openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={"guardrails": {"config_id": "nonexistent_config"}},
    )

    assert (
        "Could not load" in response.choices[0].message.content
        or "error" in response.choices[0].message.content.lower()
    )


def test_openai_client_with_context(openai_client):
    """Test OpenAI client with context in guardrails."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={
            "guardrails": {
                "config_id": "with_custom_llm",
                "context": {"user_id": "test123", "session": "abc"},
            }
        },
    )

    assert isinstance(response, ChatCompletion)
    assert response.id.startswith("chatcmpl-")
    assert response.object == "chat.completion"
    assert response.model == "gpt-4o"
    assert response.choices[0].index == 0
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Custom LLM response"
    assert hasattr(response, "guardrails")
    assert response.guardrails["config_id"] == "with_custom_llm"


def test_openai_client_with_options(openai_client):
    """Test OpenAI client with custom options in guardrails."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={
            "guardrails": {
                "config_id": "with_custom_llm",
                "options": {
                    "rails": {"input": False, "output": False},
                },
            }
        },
    )

    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert response.model == "gpt-4o"
    assert response.choices[0].message.content == "Custom LLM response"
    assert response.guardrails["config_id"] == "with_custom_llm"


def test_openai_client_with_empty_state(openai_client):
    """Test OpenAI client with empty state in guardrails."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={
            "guardrails": {
                "config_id": "with_custom_llm",
                "state": {},
            }
        },
    )

    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert response.model == "gpt-4o"
    assert response.choices[0].message.content == "Custom LLM response"
    assert response.guardrails["config_id"] == "with_custom_llm"


def test_openai_client_with_all_guardrails_fields(openai_client):
    """Test OpenAI client with all guardrails fields populated."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={
            "guardrails": {
                "config_id": "with_custom_llm",
                "context": {"user_id": "test123"},
                "options": {
                    "rails": {"input": True, "output": True},
                    "log": {"activated_rails": True},
                },
                "state": {},
            }
        },
    )

    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert response.model == "gpt-4o"
    assert response.choices[0].message.content == "Custom LLM response"
    assert response.guardrails["config_id"] == "with_custom_llm"

    assert "log" in response.guardrails
    assert response.guardrails["log"] is not None
    assert "activated_rails" in response.guardrails["log"]
    assert isinstance(response.guardrails["log"]["activated_rails"], list)
    assert "stats" in response.guardrails["log"]
    assert isinstance(response.guardrails["log"]["stats"], dict)
    assert "total_duration" in response.guardrails["log"]["stats"]


def test_openai_client_with_multiple_configs(openai_client):
    """Test OpenAI client with multiple config_ids."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={
            "guardrails": {
                "config_ids": ["with_custom_llm"],
            }
        },
    )

    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert response.model == "gpt-4o"
    assert response.choices[0].message.content == "Custom LLM response"
    assert response.guardrails["config_id"] == "with_custom_llm"


def test_openai_client_with_rails_disabled(openai_client):
    """Test OpenAI client with all rails disabled.

    When dialog rails are disabled, the LLM is called directly without going
    through the dialog flow, resulting in the user message being echoed back.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_body={
            "guardrails": {
                "config_id": "with_custom_llm",
                "options": {
                    "rails": {
                        "input": False,
                        "output": False,
                        "dialog": False,
                    },
                },
            }
        },
    )

    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert response.model == "gpt-4o"
    assert response.choices[0].message.content == "hi"
    assert response.guardrails["config_id"] == "with_custom_llm"
