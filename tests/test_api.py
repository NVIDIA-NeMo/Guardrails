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

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from nemoguardrails.server.api import RequestBody

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def set_rails_config_path():
    api.app.rails_config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_configs"))
    yield
    api.app.rails_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples", "bots")
    )


def test_get():
    response = client.get("/v1/rails/configs")
    assert response.status_code == 200

    result = response.json()
    assert len(result) > 0


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "config_id": "general",
            "messages": [
                {
                    "content": "Hello",
                    "role": "user",
                }
            ],
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res["messages"]) == 1
    assert res["messages"][0]["content"]


@pytest.mark.skip(reason="Should only be run locally as it needs OpenAI key.")
def test_chat_completion_with_default_configs():
    api.set_default_config_id("general")

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "content": "Hello",
                    "role": "user",
                }
            ],
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res["messages"]) == 1
    assert res["messages"][0]["content"]


def test_request_body_validation():
    """Test RequestBody validation."""

    data = {
        "config_id": "test_config",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    request_body = RequestBody.model_validate(data)
    assert request_body.config_id == "test_config"
    assert request_body.config_ids == ["test_config"]

    data = {
        "config_ids": ["test_config1", "test_config2"],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    request_body = RequestBody.model_validate(data)
    assert request_body.config_ids == ["test_config1", "test_config2"]

    data = {
        "config_id": "test_config",
        "config_ids": ["test_config1", "test_config2"],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    with pytest.raises(ValueError, match="Only one of config_id or config_ids should be specified"):
        RequestBody.model_validate(data)

    data = {"messages": [{"role": "user", "content": "Hello"}]}
    request_body = RequestBody.model_validate(data)
    assert request_body.config_ids is None


def test_request_body_state():
    """Test RequestBody state handling."""
    data = {
        "config_id": "test_config",
        "messages": [{"role": "user", "content": "Hello"}],
        "state": {"key": "value"},
    }
    request_body = RequestBody.model_validate(data)
    assert request_body.state == {"key": "value"}


def test_request_body_messages():
    """Test RequestBody messages validation."""
    data = {
        "config_id": "test_config",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    }
    request_body = RequestBody.model_validate(data)
    assert len(request_body.messages) == 2

    data = {
        "config_id": "test_config",
        "messages": [{"content": "Hello"}],
    }
    request_body = RequestBody.model_validate(data)
    assert len(request_body.messages) == 1
