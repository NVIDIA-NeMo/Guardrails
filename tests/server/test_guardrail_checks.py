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

from nemoguardrails.exceptions import RailTypeNotConfiguredError, RailTypeNotSupportedError
from nemoguardrails.rails import LLMRails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType, ToolViolation, ToolViolationType
from nemoguardrails.server import api
from nemoguardrails.testing.fake_model import FakeLLMModel

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


def test_reason_returned_on_blocked():
    """A blocked check's reason passes through to the response alongside the rail."""
    result = RailsResult(
        status=RailStatus.BLOCKED,
        content="I'm sorry, I can't respond to that.",
        rail="content safety check input",
        reason="unsafe request",
    )
    data = _checked(result)
    assert data["rail"] == "content safety check input"
    assert data["reason"] == "unsafe request"


@pytest.mark.parametrize(
    "result",
    [
        RailsResult(status=RailStatus.PASSED, content="ok"),
        RailsResult(status=RailStatus.BLOCKED, content="I'm sorry, I can't help with that.", rail="self check input"),
    ],
    ids=["passed", "blocked-without-reason"],
)
def test_reason_absent_when_none(result):
    """A result with no reason or tool violations, as on a pass or an LLMRails block, omits both from the response."""
    data = _checked(result)
    assert "reason" not in data
    assert "tool_violations" not in data


_UNLINKED_RESULT = ToolViolation(
    kind="tool_result",
    violation_type=ToolViolationType.UNKNOWN_CALL_ID,
    reason="tool result for call_id 'call_9' does not correspond to a prior tool call",
    tool_call_id="call_9",
    index=3,
)
_UNDECLARED_CALL = ToolViolation(
    kind="tool_call",
    violation_type=ToolViolationType.TOOL_NOT_ALLOWED,
    reason="tool call 'rm_rf' is not an allowed tool",
    tool_call_id="call_1",
    tool_name="rm_rf",
    index=0,
)


def _tool_block(*violations: ToolViolation) -> RailsResult:
    """A blocked tool check carrying *violations*."""
    return RailsResult(
        status=RailStatus.BLOCKED,
        content="I'm sorry, I can't respond to that.",
        rail="tool result validation",
        reason=violations[0].reason,
        tool_violations=list(violations),
    )


def test_tool_violations_returned_on_blocked():
    """A blocked tool check's violations reach the response, without their unset fields."""
    data = _checked(_tool_block(_UNDECLARED_CALL))
    assert data["tool_violations"] == [
        {
            "kind": "tool_call",
            "violation_type": "tool_not_allowed",
            "reason": "tool call 'rm_rf' is not an allowed tool",
            "tool_call_id": "call_1",
            "tool_name": "rm_rf",
            "index": 0,
        }
    ]


def test_result_violation_index_skips_the_prepended_context():
    """With context prepended, a result's index still counts the request's messages; a call's index is unchanged."""
    mock = _mock_rails(_tool_block(_UNLINKED_RESULT, _UNDECLARED_CALL))

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test", "context": {"topic": "weather"}},
            }
        )

    assert [violation["index"] for violation in resp.json()["tool_violations"]] == [2, 0]


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


def test_config_py_parser_has_library_and_service_parity_for_single_and_multiple_configs(tmp_path, monkeypatch):
    first_config_path = tmp_path / "first"
    first_config_path.mkdir()
    (first_config_path / "config.yml").write_text(
        """
models: []
rails:
  input:
    flows:
      - self check input
prompts:
  - task: self_check_input
    content: "{{ user_input }}"
    output_parser: policy_parser
""",
        encoding="utf-8",
    )
    (first_config_path / "config.py").write_text(
        """
def parse_policy_output(_response):
    return [False]

def init(app):
    app.register_output_parser(parse_policy_output, "policy_parser")
    app.register_action_param("first_config_initialized", True)
""",
        encoding="utf-8",
    )

    second_config_path = tmp_path / "second"
    second_config_path.mkdir()
    (second_config_path / "config.yml").write_text("models: []\n", encoding="utf-8")
    (second_config_path / "config.py").write_text(
        """
def init(app):
    app.register_action_param("second_config_initialized", True)
""",
        encoding="utf-8",
    )

    library_rails = LLMRails(
        RailsConfig.from_path(str(first_config_path)),
        llm=FakeLLMModel(responses=["unparsed output"]),
    )
    library_result = library_rails.check(
        messages=[{"role": "user", "content": "hello"}],
        rail_types=[RailType.INPUT],
    )

    created_rails = []

    def create_service_rails(config, verbose=False):
        rails = LLMRails(config, llm=FakeLLMModel(responses=["unparsed output"]), verbose=verbose)
        created_rails.append(rails)
        return rails

    monkeypatch.setattr(api.app, "rails_config_path", str(tmp_path))
    monkeypatch.setattr(api.app, "single_config_mode", False)
    monkeypatch.setattr(api, "LLMRails", create_service_rails)

    single_response = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "guardrails": {"config_id": "first", "rail_types": ["input"]},
        }
    )
    multiple_response = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "guardrails": {"config_ids": ["first", "second"], "rail_types": ["input"]},
        }
    )

    assert library_result.status is RailStatus.BLOCKED
    assert single_response.status_code == 200
    assert single_response.json()["status"] == library_result.status.value
    assert multiple_response.status_code == 200
    assert multiple_response.json()["status"] == RailStatus.BLOCKED.value
    assert created_rails[1].runtime.registered_action_params["first_config_initialized"] is True
    assert created_rails[1].runtime.registered_action_params["second_config_initialized"] is True


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


def test_no_config_no_default_returns_422():
    api.app.default_config_id = None
    resp = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert resp.status_code == 422
    assert "config" in resp.json()["error"]["message"].lower()


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
    assert "colang 2.0" in resp.json()["error"]["message"].lower()


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


@pytest.mark.parametrize(
    "rail_types_input, expected",
    [
        (["input"], [RailType.INPUT]),
        (["output"], [RailType.OUTPUT]),
        (["input", "output"], [RailType.INPUT, RailType.OUTPUT]),
        (["tool_call"], [RailType.TOOL_CALL]),
        (["tool_result"], [RailType.TOOL_RESULT]),
        (None, None),
    ],
)
def test_rail_types_passed_through(rail_types_input, expected):
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result)

    guardrails = {"config_id": "test"}
    if rail_types_input is not None:
        guardrails["rail_types"] = rail_types_input

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": guardrails,
            }
        )

    assert resp.status_code == 200
    mock.check_async.assert_called_once()
    assert mock.check_async.call_args.kwargs["rail_types"] == expected


def test_rail_types_invalid_value_returns_422():
    resp = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "guardrails": {"config_id": "test", "rail_types": ["invalid"]},
        }
    )
    assert resp.status_code == 422


# --- Unsatisfiable rail_types ---


def test_unsatisfiable_rail_types_returns_422():
    result = RailsResult(status=RailStatus.PASSED, content="hi")
    mock = _mock_rails(result)
    mock.check_async = AsyncMock(
        side_effect=RailTypeNotConfiguredError("Requested rail type 'output' has no configured rails.")
    )

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test", "rail_types": ["output"]},
            }
        )

    assert resp.status_code == 422
    assert "output" in resp.json()["error"]["message"]


def test_unsupported_rail_types_returns_422():
    """A rail type the serving engine cannot run is a 422 carrying the engine's message, not a 500."""
    mock = _mock_rails(RailsResult(status=RailStatus.PASSED, content="hi"))
    mock.check_async = AsyncMock(
        side_effect=RailTypeNotSupportedError("tool_call rail checks run on the IORails engine only")
    )

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "guardrails": {"config_id": "test", "rail_types": ["tool_call"]},
            }
        )

    assert resp.status_code == 422
    assert "IORails" in resp.json()["error"]["message"]


_WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


def test_engine_other_than_iorails_is_called_without_tools():
    """Only an IORails engine is handed the request's tools; another engine's check_async has no such parameter."""
    mock = _mock_rails(RailsResult(status=RailStatus.PASSED, content="hi"))

    with patch.object(api, "_get_rails", new_callable=AsyncMock, return_value=mock):
        resp = _post(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_WEATHER_TOOL],
                "guardrails": {"config_id": "test"},
            }
        )

    assert resp.status_code == 200
    assert "tools" not in mock.check_async.call_args.kwargs


def test_custom_tools_are_rejected():
    """The request's tools are still validated as chat-completion tools, so a custom tool is a 422."""
    resp = _post(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "custom", "custom": {"name": "grep"}}],
            "guardrails": {"config_id": "test", "rail_types": ["tool_call"]},
        }
    )
    assert resp.status_code == 422
