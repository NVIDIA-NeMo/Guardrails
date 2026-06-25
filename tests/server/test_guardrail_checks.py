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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai is required for server tests")
from fastapi.testclient import TestClient

from nemoguardrails.rails.llm.options import RailsResult, RailStatus
from nemoguardrails.server import api

client = TestClient(api.app)

ENDPOINT = "/v1/checks"


@pytest.fixture(autouse=True)
def reset_server_state():
    original_default = api.app.default_config_id
    api.llm_rails_instances.clear()
    yield
    api.app.default_config_id = original_default
    api.llm_rails_instances.clear()


def _mock_rails(check_result: RailsResult, colang_version: str = "1.0") -> MagicMock:
    mock = MagicMock()
    mock.check_async = AsyncMock(return_value=check_result)
    mock.config.colang_version = colang_version
    return mock


def _post(body: dict, **kwargs):
    return client.post(ENDPOINT, json=body, **kwargs)


def _checked(result, config_id="test", colang_version="1.0"):
    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=_mock_rails(result, colang_version)):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": config_id},
            }
        )
    assert resp.status_code == 200
    return resp.json()


# --- Status mapping ---


@pytest.mark.parametrize(
    "rail_status, expected",
    [
        (RailStatus.PASSED, "passed"),
        (RailStatus.MODIFIED, "modified"),
        (RailStatus.BLOCKED, "blocked"),
    ],
)
def test_status_mapping(rail_status, expected):
    result = RailsResult(status=rail_status, content="x")
    data = _checked(result)
    assert data["status"] == expected


# --- Content and rail fields ---


def test_content_returned_on_passed():
    result = RailsResult(status=RailStatus.PASSED, content="hello there")
    data = _checked(result)
    assert data["content"] == "hello there"
    assert data.get("rail") is None


def test_content_returned_on_modified():
    result = RailsResult(status=RailStatus.MODIFIED, content="sanitized text")
    data = _checked(result)
    assert data["content"] == "sanitized text"


def test_content_returned_on_blocked():
    result = RailsResult(
        status=RailStatus.BLOCKED, content="I'm sorry, I can't help with that.", rail="self check input"
    )
    data = _checked(result)
    assert data["status"] == "blocked"
    assert data["content"] == "I'm sorry, I can't help with that."
    assert data["rail"] == "self check input"


def test_rail_null_on_passed():
    result = RailsResult(status=RailStatus.PASSED, content="ok")
    data = _checked(result)
    assert "rail" not in data


# --- Config resolution ---


def test_config_id_resolves():
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result)

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock) as mock_get:
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "my_config"},
            }
        )

    assert resp.status_code == 200
    mock_get.assert_called_once_with(["my_config"], model_name="test")


def test_config_string_resolves_via_get_rails():
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result)

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock) as mock_get:
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config": "my_config"},
            }
        )

    assert resp.status_code == 200
    mock_get.assert_called_once_with(["my_config"], model_name="test")


def test_inline_config():
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock_llm_rails = _mock_rails(result)

    with (
        patch("nemoguardrails.server.api.RailsConfig") as mock_config_cls,
        patch("nemoguardrails.server.api.LLMRails", return_value=mock_llm_rails),
    ):
        mock_config_cls.from_content.return_value = MagicMock()
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config": {"models": [], "rails": {}}},
            }
        )

    assert resp.status_code == 200
    mock_config_cls.from_content.assert_called_once()


def test_default_config_used_when_none_specified():
    api.app.default_config_id = "my_default"
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result)

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock) as mock_get:
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )

    assert resp.status_code == 200
    mock_get.assert_called_once_with(["my_default"], model_name="test")


# --- Validation ---


def test_empty_messages_returns_422():
    resp = _post(
        {
            "model": "test",
            "messages": [],
            "guardrails": {"config_id": "test"},
        }
    )
    assert resp.status_code == 422


def test_config_and_config_id_mutually_exclusive():
    resp = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {"config": "a", "config_id": "b"},
        }
    )
    assert resp.status_code == 422


def test_no_config_no_default_returns_422():
    api.app.default_config_id = None
    resp = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert resp.status_code == 422
    assert "config" in resp.json()["detail"].lower()


# --- Colang 2.0 rejection ---


def test_colang_v2_returns_422():
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result, colang_version="2.x")

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test"},
            }
        )

    assert resp.status_code == 422
    assert "colang 2.0" in resp.json()["detail"].lower()


# --- Error handling ---


def test_get_rails_failure_returns_422():
    with patch.object(api, "_get_rails", new_callable=AsyncMock, side_effect=ValueError("bad config")):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "bad"},
            }
        )

    assert resp.status_code == 422


def test_check_async_failure_returns_500():
    mock = MagicMock()
    mock.check_async = AsyncMock(side_effect=RuntimeError("boom"))
    mock.config.colang_version = "1.0"

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test"},
            }
        )

    assert resp.status_code == 500


# --- Context forwarding ---


def test_context_prepended_to_messages():
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result)

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test", "context": {"topic": "science"}},
            }
        )

    assert resp.status_code == 200
    call_args = mock.check_async.call_args
    messages = call_args.kwargs.get("messages") or call_args[0][0]
    assert messages[0]["role"] == "context"
    assert messages[0]["content"] == {"topic": "science"}
