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
import time

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.model import Model

from nemoguardrails.server import api


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    """Set the rails config path to the test configs directory."""
    original_path = api.app.rails_config_path
    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs/simple_server"))
    yield

    # Restore the original path and clear cache after the test
    api.app.rails_config_path = original_path
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()


@pytest.fixture(scope="function")
def test_client():
    """Create a FastAPI TestClient for the guardrails server."""
    return TestClient(api.app)


@pytest.fixture(scope="function")
def openai_client(test_client):
    client = OpenAI(
        api_key="dummy-key",
        base_url="http://dummy-url/v1",
        http_client=test_client,
    )
    return client


def test_openai_client_list_models(openai_client):
    models = openai_client.models.list()

    # Verify the response structure matches the GuardrailsModel schema
    assert models.data[0] == Model(
        id="config_1",
        object="model",
        created=int(time.time()),
        owned_by="nemo-guardrails",
        guardrails_config_id="config_1",
    )


def test_openai_client_chat_completion(openai_client):
    response = openai_client.chat.completions.create(
        model="config_1",
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
            content="Hello!",
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
        model="config_1",
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
            content="Hello!",
            refusal=None,
            role="assistant",
            annotations=None,
        ),
    )
    assert hasattr(response, "created")


def test_openai_client_chat_completion_input_rails(openai_client):
    response = openai_client.chat.completions.create(
        model="input_rails",
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
