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
    """Set the rails_config_path to test configs."""
    original_path = api.app.rails_config_path
    # Use test_configs which have mock LLMs that don't need API keys
    test_configs_path = os.path.join(os.path.dirname(__file__), "..", "test_configs")
    api.app.rails_config_path = test_configs_path
    yield
    api.app.rails_config_path = original_path


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


def test_openai_client_list_models(openai_client):
    models = openai_client.models.list()

    # Verify the response structure matches the GuardrailsModel schema
    assert len(models.data) > 0
    model = models.data[0]
    # Verify it's a valid model response with required fields
    assert model.object == "model"
    assert model.owned_by == "nemo-guardrails"
    # Check the extra fields that GuardrailsModel adds
    assert hasattr(model, "config_id")
    assert hasattr(model, "engine")
    assert hasattr(model, "base_url")
    assert model.base_url == "https://localhost:8000/v1"


def test_openai_client_chat_completion(openai_client):
    response = openai_client.chat.completions.create(
        model="with_custom_llm",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
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
        model="with_custom_llm",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
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
        model="with_input_rails",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=False,
    )

    # Verify response exists
    assert isinstance(response, ChatCompletion)
    assert response.id is not None
    assert isinstance(response.choices[0], Choice)
    assert hasattr(response, "created")


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_openai_client_chat_completion_streaming(openai_client):
    stream = openai_client.chat.completions.create(
        model="input_rails",
        messages=[{"role": "user", "content": "Tell me a short joke."}],
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) > 0

    # Verify at least one chunk has content
    has_content = any(hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content for chunk in chunks)
    assert has_content, "At least one chunk should contain content"


def test_openai_client_error_handling_invalid_model(openai_client):
    response = openai_client.chat.completions.create(
        model="nonexistent_config",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    # The error should be in the content
    assert (
        "Could not load" in response.choices[0].message.content
        or "error" in response.choices[0].message.content.lower()
    )
