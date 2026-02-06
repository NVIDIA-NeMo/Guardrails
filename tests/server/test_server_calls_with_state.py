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

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.server import api

client = TestClient(api.app)


@pytest.fixture(scope="function", autouse=True)
def setup_test_env():
    original_engine = os.environ.get("MAIN_MODEL_ENGINE")
    os.environ["MAIN_MODEL_ENGINE"] = "custom_llm"
    api.llm_rails_instances.clear()
    yield
    api.llm_rails_instances.clear()
    if original_engine is not None:
        os.environ["MAIN_MODEL_ENGINE"] = original_engine
    else:
        os.environ.pop("MAIN_MODEL_ENGINE", None)


def _test_call(config_id):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "guardrails": {
                "config_id": config_id,
                "state": {},
            },
        },
    )
    assert response.status_code == 200
    res = response.json()
    print(res)
    assert len(res["choices"][0]["message"]) == 2
    assert res["choices"][0]["message"]["content"] == "Hello!"
    assert res["guardrails"]["state"]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "guardrails": {
                "config_id": config_id,
                "state": res["guardrails"]["state"],
            },
        },
    )
    res = response.json()
    assert res["choices"][0]["message"]["content"] == "Hello again!"


def test_1():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server")
    _test_call("config_1")


def test_2():
    api.app.rails_config_path = os.path.join(os.path.dirname(__file__), "..", "test_configs", "simple_server_2_x")
    _test_call("config_2")
