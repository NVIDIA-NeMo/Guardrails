# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.server import api

client = TestClient(api.app)

ENDPOINT = "/v1/health"


def test_health_returns_200_pass_status():
    """GET /v1/health returns HTTP 200 with a JSON body of {"status": "pass"}."""
    response = client.get(ENDPOINT)

    assert response.status_code == 200
    assert response.json() == {"status": "pass"}


def test_health_uses_health_json_media_type():
    """GET /v1/health responds with exactly the application/health+json media type."""
    response = client.get(ENDPOINT)

    assert response.headers["content-type"] == "application/health+json"


def test_health_rejects_post_with_405():
    """POST /v1/health returns HTTP 405 because the endpoint is GET-only."""
    response = client.post(ENDPOINT)

    assert response.status_code == 405


def test_health_ok_without_config_or_cached_rails(monkeypatch):
    """GET /v1/health returns 200 with no default config and no cached rails instances."""
    monkeypatch.setattr(api.app, "default_config_id", None)
    monkeypatch.setattr(api, "llm_rails_instances", {})

    response = client.get(ENDPOINT)

    assert response.status_code == 200
    assert response.json() == {"status": "pass"}


def test_healthz_alias_matches_health_contract():
    """GET /healthz returns the same 200 application/health+json {"status": "pass"} response as /v1/health."""
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/health+json"
    assert response.json() == {"status": "pass"}
