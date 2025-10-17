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

from fastapi.testclient import TestClient

from nemoguardrails.server import api

client = TestClient(api.app)


def _test_call(config_id):
    response = client.post(
        "/v1/chat/completions",
        json={
            "config_id": config_id,
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "state": {},
        },
    )
    assert response.status_code == 200
    res = response.json()
    assert len(res["messages"]) == 1
    assert res["messages"][0]["content"] == "Hello!"
    assert res.get("state")

    # When making a second call with the returned state, the conversations should continue
    # and we should get the "Hello again!" message.
    response = client.post(
        "/v1/chat/completions",
        json={
            "config_id": config_id,
            "messages": [
                {
                    "content": "hi",
                    "role": "user",
                }
            ],
            "state": res["state"],
        },
    )
    res = response.json()
    assert res["messages"][0]["content"] == "Hello again!"


def test_1():
    api.app.rails_config_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "simple_server"
    )
    _test_call("config_1")


def test_2():
    api.app.rails_config_path = os.path.join(
        os.path.dirname(__file__), "test_configs", "simple_server_2_x"
    )
    _test_call("config_2")
