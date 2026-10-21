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

"""The hooks server through its ASGI app, on a real IORails engine whose model is never called."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import jsonschema
import pytest
from fastapi.testclient import TestClient

from nemoguardrails import telemetry
from nemoguardrails.exceptions import LLMCallException, NonStreamingWorkQueueFullError, RailTypeNotConfiguredError
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.model_engine import ModelEngine, ModelEngineError
from nemoguardrails.http.errors import HTTPConnectionError
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks import HookAgent
from nemoguardrails.server.hooks import app as hooks_app
from nemoguardrails.server.hooks.app import HooksConfigError, create_hooks_app
from tests.server.test_hooks_models import (
    CLAUDE_CODE_BASH,
    CLAUDE_CODE_DESTRUCTIVE_BASH,
    CLAUDE_CODE_MCP,
    CLAUDE_CODE_WRITE_WINDOWS,
    CODEX_APPLY_PATCH,
    CODEX_MCP,
    CODEX_OUTPUT_SCHEMA,
    NEMO_RELAY_EMPTY_QUERY,
    NEMO_RELAY_SEARCH,
)

AGENTS: list[HookAgent] = ["claude-code", "codex", "nemo-relay"]
PRE_TOOL_USE_AGENTS: list[HookAgent] = ["claude-code", "codex"]

MAIN_MODEL = {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}

# `{"type": "object"}` alone declares a tool that takes no arguments, so every call with one would be denied.
ANY_ARGUMENTS = {"type": "object", "additionalProperties": True}


def _main_model(*tools: dict) -> dict:
    """The `main` model, declaring *tools*."""
    return {**MAIN_MODEL, "parameters": {"tools": list(tools)}}


def _tool(name: str, parameters: dict) -> dict:
    """A tool declaration, in the Chat Completions shape."""
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


HOOKS_CONFIG = {
    "models": [_main_model(_tool("Bash", ANY_ARGUMENTS))],
    "rails": {
        "config": {"regex_detection": {"tool_call": {"Bash": {"patterns": [r"rm\s+-rf"]}}}},
        "tool_call": {"per_tool": {"Bash": ["regex check tool call $argument=command"]}},
    },
}

# The rail raises on a `$argument=` its tool's closed schema can't accept, so every `Read` call fails it.
RAIL_FAILURE_CONFIG = {
    "models": [
        _main_model(
            _tool(
                "Read",
                {"type": "object", "properties": {"file_path": {"type": "string"}}, "additionalProperties": False},
            )
        )
    ],
    "rails": {"tool_call": {"per_tool": {"Read": ["regex check tool call $argument=x"]}}},
}

ALLOWLIST_CONFIG = {
    "models": [_main_model(_tool("Bash", ANY_ARGUMENTS))],
    "rails": {"tool_call": {"flows": ["tool call validation"]}},
}

_HOOK_BODIES = {"claude-code": CLAUDE_CODE_DESTRUCTIVE_BASH, "codex": CODEX_MCP, "nemo-relay": NEMO_RELAY_SEARCH}

# The spec's `422` examples.
UNSUPPORTED_HOOK_EVENT = {
    "error": {
        "message": "hook_event_name 'PostToolUse' is not handled on /v1/hooks/claude-code; send PreToolUse",
        "type": "invalid_request_error",
        "param": "hook_event_name",
        "code": "unsupported_hook_event",
    }
}
MISSING_TOOL_NAME = {
    "error": {
        "message": "body.tool_name: Field required",
        "type": "invalid_request_error",
        "param": "tool_name",
        "code": None,
    }
}
RELAY_UNKNOWN_FIELD = {
    "error": {
        "message": "body.session_id: Extra inputs are not permitted",
        "type": "invalid_request_error",
        "param": "session_id",
        "code": None,
    }
}

# The spec's `500` and `503` examples that don't depend on the configured rails.
CHECK_FAILED = {
    "error": {
        "message": "the tool-call check failed: tool call extraction failed",
        "type": "server_error",
        "param": None,
        "code": None,
    }
}
INTERNAL_ERROR = {"error": {"message": "Internal server error", "type": "server_error", "param": None, "code": None}}
OVERLOADED = {
    "error": {
        "message": "IORails admission queue is full. Please retry later.",
        "type": "server_error",
        "param": None,
        "code": "queue_full",
    }
}

# The spec's `denyNotAllowed` example.
DENY_NOT_ALLOWED = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "NeMo Guardrails: tool call 'WebFetch' is not an allowed tool",
    }
}

MODEL_CALL_FAILURE = "model 'llama_guard' timed out"

EXAMPLE_CONFIGS_PATH = Path(__file__).parents[2] / "examples" / "configs" / "agent_hooks"
EXAMPLE_CONFIGS = {"claude-code": "claude_code", "codex": "codex", "nemo-relay": "nemo_relay"}


def _hook(agent: HookAgent, tool_name: str, tool_input: Any) -> dict:
    """A hook on *agent*'s path, calling *tool_name* with *tool_input*."""
    return {**_HOOK_BODIES[agent], "tool_name": tool_name, "tool_input": tool_input}


def _allowed(agent: HookAgent) -> dict:
    """*agent*'s allow."""
    return {"decision": "allow"} if agent == "nemo-relay" else {}


def _denied(agent: HookAgent, reason: str) -> dict:
    """*agent*'s deny, for *reason*."""
    if agent == "nemo-relay":
        return {"decision": "deny", "reason": reason}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"NeMo Guardrails: {reason}",
        }
    }


def _app(config: dict):
    """A hooks server for *config*."""
    return create_hooks_app(RailsConfig.from_content(config=config))


def _error_param(response) -> tuple[int, Any, Any]:
    """The status, `param` and `code` of an error response."""
    error = response.json()["error"]
    return response.status_code, error["param"], error["code"]


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    """Provider credentials, no engine switch, and the process's deployment type restored after the test."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)


@pytest.fixture(autouse=True)
def model_call(monkeypatch):
    """The model transport, which fails the test if a check ever reaches it."""
    call = AsyncMock(side_effect=AssertionError("the hooks server never calls a model"))
    monkeypatch.setattr(ModelEngine, "call", call)
    yield call
    call.assert_not_called()


@pytest.fixture
def client():
    """A running hooks server for `HOOKS_CONFIG`."""
    with TestClient(_app(HOOKS_CONFIG)) as client:
        yield client


@pytest.fixture
def failing_check(client, monkeypatch):
    """Make every check on *client* raise the exception the test passes in."""

    def fail_with(exc: BaseException) -> TestClient:
        monkeypatch.setattr(client.app.state.engine, "check_async", AsyncMock(side_effect=exc))
        return client

    return fail_with


class TestChecks:
    @pytest.mark.parametrize("agent", AGENTS)
    def test_call_no_rail_blocks_is_allowed(self, client, agent):
        """A call its tool's rail lets through gets the agent's allow."""
        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", {"command": "npm test"}))

        assert (response.status_code, response.json()) == (200, _allowed(agent))

    @pytest.mark.parametrize("agent", AGENTS)
    def test_call_a_rail_blocks_is_denied_with_the_rail_as_reason(self, client, agent):
        """A call its tool's rail blocks gets the agent's deny, naming the rail."""
        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", {"command": "rm -rf /tmp/build"}))

        assert (response.status_code, response.json()) == (200, _denied(agent, "regex check tool call"))

    @pytest.mark.parametrize(
        "agent, call",
        [
            ("claude-code", f"'{CLAUDE_CODE_DESTRUCTIVE_BASH['tool_use_id']}'"),
            ("codex", f"'{CODEX_MCP['tool_use_id']}'"),
            ("nemo-relay", "at index 0"),
        ],
    )
    @pytest.mark.parametrize("tool_input", ["rm -rf /tmp/build", ["rm", "-rf"], None])
    def test_input_that_is_not_an_object_is_denied_as_malformed(self, client, agent, call, tool_input):
        """A `tool_input` that isn't a JSON object is denied as malformed arguments, naming the call."""
        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", tool_input))

        reason = f"tool call extraction failed: tool call {call} has malformed arguments"
        assert (response.status_code, response.json()) == (200, _denied(agent, reason))

    @pytest.mark.parametrize("agent", AGENTS)
    def test_tool_without_rails_is_allowed(self, client, agent):
        """A call to a tool no rail is configured for is allowed without a check, though no model declares it."""
        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Read", {"file_path": "/etc/passwd"}))

        assert (response.status_code, response.json()) == (200, _allowed(agent))

    @pytest.mark.parametrize("agent", AGENTS)
    def test_body_is_read_as_json_whatever_its_content_type(self, client, agent):
        """The body is parsed as JSON without a `Content-Type` saying so."""
        body = json.dumps(_hook(agent, "Bash", {"command": "rm -rf /tmp/build"}))

        response = client.post(f"/v1/hooks/{agent}", content=body, headers={"Content-Type": "text/plain"})

        assert (response.status_code, response.json()) == (200, _denied(agent, "regex check tool call"))

    @pytest.mark.parametrize("command", ["npm test", "rm -rf /tmp/build"], ids=["allow", "deny"])
    @pytest.mark.parametrize("agent", PRE_TOOL_USE_AGENTS)
    def test_agent_responses_validate_against_codex_output_schema(self, client, agent, command):
        """Every Claude Code and Codex answer validates against Codex's output schema, which rejects unknown fields."""
        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", {"command": command}))

        jsonschema.validate(response.json(), CODEX_OUTPUT_SCHEMA)

    def test_each_check_logs_one_line_without_the_tool_input(self, client, caplog):
        """A check logs one line with the agent, event, tool, call id, outcome and rail, and never the tool input."""
        caplog.set_level(logging.INFO)

        client.post("/v1/hooks/claude-code", json=_hook("claude-code", "Bash", {"command": "rm -rf /tmp/secret-7f3a"}))

        assert [record.getMessage() for record in caplog.records if record.name == hooks_app.__name__] == [
            "Hook check on /v1/hooks/claude-code: event=PreToolUse tool_name='Bash' "
            "tool_use_id='toolu_01GHI789...' outcome=deny code=None rail='regex check tool call'"
        ]
        assert "secret-7f3a" not in caplog.text


class TestSpecExamples:
    @pytest.mark.parametrize(
        "agent, body, expected",
        [
            ("claude-code", CLAUDE_CODE_BASH, {}),
            ("claude-code", CLAUDE_CODE_WRITE_WINDOWS, {}),
            ("claude-code", CLAUDE_CODE_MCP, {}),
            ("codex", CODEX_APPLY_PATCH, {}),
            ("codex", CODEX_MCP, {}),
            ("nemo-relay", NEMO_RELAY_SEARCH, {"decision": "allow"}),
            ("nemo-relay", NEMO_RELAY_EMPTY_QUERY, {"decision": "allow"}),
            (
                "claude-code",
                CLAUDE_CODE_DESTRUCTIVE_BASH,
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "NeMo Guardrails: regex check tool call",
                    }
                },
            ),
        ],
        ids=[
            "claudeCodeBash",
            "claudeCodeWriteWindows",
            "claudeCodeMcp",
            "codexApplyPatch",
            "codexMcp",
            "nemoRelaySearch",
            "nemoRelayEmptyQuery",
            "claudeCodeDestructiveBash",
        ],
    )
    def test_spec_request_example_gets_the_spec_response(self, client, agent, body, expected):
        """Each spec request example gets the spec's allow, or its deny shape with this config's rail as reason."""
        response = client.post(f"/v1/hooks/{agent}", json=body)

        assert (response.status_code, response.json()) == (200, expected)

    def test_undeclared_tool_with_the_allowlist_gets_the_spec_deny(self):
        """A tool the `main` model doesn't declare, with `tool call validation` configured, gets `denyNotAllowed`."""
        with TestClient(_app(ALLOWLIST_CONFIG)) as client:
            response = client.post("/v1/hooks/claude-code", json=_hook("claude-code", "WebFetch", {"url": "x"}))

        assert (response.status_code, response.json()) == (200, DENY_NOT_ALLOWED)

    @pytest.mark.parametrize(
        "body, expected",
        [
            (CLAUDE_CODE_BASH, {}),
            (CLAUDE_CODE_DESTRUCTIVE_BASH, _denied("claude-code", "regex check tool call")),
            (CODEX_APPLY_PATCH, {}),
            (CODEX_MCP, {}),
        ],
        ids=["claudeCodeBash", "claudeCodeDestructiveBash", "codexApplyPatch", "codexMcp"],
    )
    def test_spec_request_example_gets_the_same_answer_on_either_agents_path(self, client, body, expected):
        """A Claude Code or Codex request example is checked and answered the same on both agents' paths."""
        responses = [client.post(f"/v1/hooks/{agent}", json=body) for agent in PRE_TOOL_USE_AGENTS]

        assert [(response.status_code, response.json()) for response in responses] == [(200, expected)] * 2

    @pytest.mark.parametrize(
        "agent, body, expected",
        [
            ("claude-code", {**CLAUDE_CODE_BASH, "hook_event_name": "PostToolUse"}, UNSUPPORTED_HOOK_EVENT),
            (
                "claude-code",
                {key: value for key, value in CLAUDE_CODE_BASH.items() if key != "tool_name"},
                MISSING_TOOL_NAME,
            ),
            ("nemo-relay", {**NEMO_RELAY_SEARCH, "session_id": "abc123"}, RELAY_UNKNOWN_FIELD),
        ],
        ids=["unsupportedHookEvent", "missingToolName", "relayUnknownField"],
    )
    def test_unprocessable_request_gets_the_spec_error(self, client, agent, body, expected):
        """Each of the spec's `422` examples is the body the server answers its request with."""
        response = client.post(f"/v1/hooks/{agent}", json=body)

        assert (response.status_code, response.json()) == (422, expected)


class TestValidation:
    @pytest.mark.parametrize("agent", AGENTS)
    def test_event_the_path_does_not_handle_is_unsupported_hook_event(self, client, agent):
        """A `hook_event_name` the path doesn't handle gets `422 unsupported_hook_event` at `hook_event_name`."""
        response = client.post(f"/v1/hooks/{agent}", json={**_HOOK_BODIES[agent], "hook_event_name": "Stop"})

        assert _error_param(response) == (422, "hook_event_name", "unsupported_hook_event")

    @pytest.mark.parametrize(
        "agent, field",
        [
            ("claude-code", "tool_use_id"),
            ("codex", "tool_use_id"),
            ("nemo-relay", "tool_input"),
            ("claude-code", "hook_event_name"),
        ],
    )
    def test_missing_field_is_a_schema_error_at_that_field(self, client, agent, field):
        """A body missing a required field gets a schema `422` naming that field."""
        body = {key: value for key, value in _HOOK_BODIES[agent].items() if key != field}

        response = client.post(f"/v1/hooks/{agent}", json=body)

        assert _error_param(response) == (422, field, None)

    @pytest.mark.parametrize(
        "agent, field, value",
        [
            ("claude-code", "turn_id", 7),
            ("codex", "mcp_server", None),
            ("claude-code", "session_id", ["not", "a", "str"]),
        ],
    )
    def test_field_the_server_does_not_read_is_accepted_with_any_value(self, client, agent, field, value):
        """A field the server doesn't read, even the other agent's, is accepted with any value and the call checked."""
        body = {**_hook(agent, "Bash", {"command": "rm -rf /tmp/build"}), field: value}

        response = client.post(f"/v1/hooks/{agent}", json=body)

        assert (response.status_code, response.json()) == (200, _denied(agent, "regex check tool call"))

    @pytest.mark.parametrize("body", [b"{not json", b"", b"\xff\xfe"], ids=["malformed", "empty", "not-utf-8"])
    def test_body_that_is_not_json_is_a_schema_error_without_a_param(self, client, body):
        """A body that doesn't parse as JSON gets a schema `422` with no `param`."""
        response = client.post("/v1/hooks/claude-code", content=body)

        assert (response.status_code, response.json()["error"]) == (
            422,
            {"message": "body: JSON decode error", "type": "invalid_request_error", "param": None, "code": None},
        )

    @pytest.mark.parametrize("body", [[CLAUDE_CODE_BASH], "PreToolUse", None])
    def test_body_that_is_not_an_object_is_a_schema_error_without_a_param(self, client, body):
        """A JSON body that isn't an object gets a schema `422` with no `param`."""
        response = client.post("/v1/hooks/claude-code", json=body)

        assert _error_param(response) == (422, None, None)

    def test_schema_error_never_quotes_the_tool_input(self, client, caplog):
        """A schema `422` reports fields and reasons only, so the tool input reaches neither the body nor the log."""
        caplog.set_level(logging.INFO)
        body = {**_hook("codex", "Bash", {"command": "cat /tmp/secret-7f3a"}), "tool_use_id": 7}

        response = client.post("/v1/hooks/codex", json=body)

        assert response.status_code == 422
        assert "secret-7f3a" not in response.text
        assert "secret-7f3a" not in caplog.text


class TestErrors:
    @pytest.mark.parametrize("agent", AGENTS)
    def test_rail_that_fails_is_rail_failed_error(self, agent):
        """A call whose only rail raises gets `500 rail_failed`, naming the rail, rather than a deny."""
        with TestClient(_app(RAIL_FAILURE_CONFIG)) as client:
            response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Read", {"file_path": "/etc/hosts"}))

        assert (response.status_code, response.json()) == (
            500,
            {
                "error": {
                    "message": "rail 'regex check tool call' failed: regex check tool call error",
                    "type": "server_error",
                    "param": None,
                    "code": "rail_failed",
                }
            },
        )

    @pytest.mark.parametrize(
        "exc",
        [
            LLMCallException(MODEL_CALL_FAILURE),
            ModelEngineError(MODEL_CALL_FAILURE, model_name="llama_guard", status=504),
            HTTPConnectionError(MODEL_CALL_FAILURE),
        ],
        ids=["LLMCallException", "ModelEngineError", "HTTPClientError"],
    )
    def test_failed_model_call_is_rail_failed_error(self, failing_check, exc):
        """A rail's model call that fails and propagates out of the check gets `500 rail_failed`."""
        response = failing_check(exc).post("/v1/hooks/codex", json=CODEX_MCP)

        assert (response.status_code, response.json()) == (
            500,
            {
                "error": {
                    "message": f"a tool-call rail failed: {MODEL_CALL_FAILURE}",
                    "type": "server_error",
                    "param": None,
                    "code": "rail_failed",
                }
            },
        )

    @pytest.mark.parametrize("agent", AGENTS)
    def test_check_that_cannot_extract_the_call_is_check_failed_error(self, client, monkeypatch, agent):
        """A check that fails before any rail decides gets the spec's `checkFailed` error, with no code."""
        monkeypatch.setattr(ModelEngine, "extract_latest_tool_calls", MagicMock(side_effect=RuntimeError("a bug")))

        response = client.post(f"/v1/hooks/{agent}", json=_HOOK_BODIES[agent])

        assert (response.status_code, response.json()) == (500, CHECK_FAILED)

    def test_full_admission_queue_is_retryable_queue_full(self, failing_check):
        """A full admission queue gets the spec's `503 queue_full`, with `Retry-After`."""
        response = failing_check(NonStreamingWorkQueueFullError()).post("/v1/hooks/claude-code", json=CLAUDE_CODE_BASH)

        assert (response.status_code, response.headers["retry-after"], response.json()) == (503, "30", OVERLOADED)

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("unexpected"), RailTypeNotConfiguredError("Requested rail type 'tool_call' has no rails.")],
        ids=["RuntimeError", "RailTypeNotConfiguredError"],
    )
    def test_unexpected_failure_is_internal_error(self, failing_check, exc):
        """Any other failure, a library `ValueError` included, gets the spec's `500` internal error."""
        response = failing_check(exc).post("/v1/hooks/nemo-relay", json=NEMO_RELAY_SEARCH)

        assert (response.status_code, response.json()) == (500, INTERNAL_ERROR)


class TestStartup:
    @pytest.mark.parametrize(
        "config, reason",
        [
            ({"models": [MAIN_MODEL]}, "Requested rail type 'tool_call' has no configured rails."),
            ({"models": [MAIN_MODEL], "rails": {"tool_call": {"per_tool": {"Bash": []}}}}, "has no configured rails"),
            (
                {"models": [], "rails": ALLOWLIST_CONFIG["rails"]},
                "Requested rail type 'tool_call' needs a `main` model",
            ),
        ],
        ids=["no-tool-call-rails", "per-tool-entry-without-flows", "no-main-model"],
    )
    def test_config_a_tool_call_check_refuses_is_refused_before_the_engine(self, monkeypatch, config, reason):
        """A config without tool-call rails or a `main` model is refused, before any engine is built."""
        engine = MagicMock()
        monkeypatch.setattr(hooks_app, "Guardrails", engine)

        with pytest.raises(HooksConfigError, match="needs tool-call rails and a `main` model") as exc_info:
            _app(config)

        assert reason in str(exc_info.value)
        engine.assert_not_called()

    def test_config_iorails_cannot_serve_is_refused(self):
        """A config IORails can't serve, here Colang 2.x, is refused with IORails's reason."""
        with pytest.raises(HooksConfigError, match="IORails, which can't serve this config: IORails supports Colang"):
            _app({**ALLOWLIST_CONFIG, "colang_version": "2.x"})

    def test_engine_that_fails_to_build_is_refused(self, monkeypatch):
        """An engine that refuses the config is reported as a config the server can't serve."""
        monkeypatch.setattr(hooks_app, "Guardrails", MagicMock(side_effect=ValueError("IORails cannot be used")))

        with pytest.raises(HooksConfigError, match="can't build its engine: IORails cannot be used"):
            _app(HOOKS_CONFIG)

    def test_engine_that_is_not_iorails_is_refused(self, monkeypatch):
        """An engine other than IORails is refused, so the routes never run on LLMRails."""
        monkeypatch.setattr(hooks_app, "Guardrails", MagicMock(return_value=SimpleNamespace(rails_engine=object())))

        with pytest.raises(HooksConfigError, match="isn't IORails"):
            _app(HOOKS_CONFIG)

    def test_tools_that_do_not_parse_are_refused(self):
        """A `main` model whose tools don't parse, which would deny every call, is refused."""
        config = {**HOOKS_CONFIG, "models": [_main_model(_tool("Bash", ANY_ARGUMENTS), _tool("Bash", ANY_ARGUMENTS))]}

        with pytest.raises(HooksConfigError, match="can't read the `main` model's tools: duplicate tool 'Bash'"):
            _app(config)

    def test_tool_schema_that_is_not_json_schema_is_refused(self):
        """A declared tool whose schema isn't valid JSON Schema, which would fail every call to it, is refused."""
        config = {**HOOKS_CONFIG, "models": [_main_model(_tool("Bash", {"type": "not-a-type"}))]}

        with pytest.raises(
            HooksConfigError,
            match="can't use the `main` model's tools: declared schema for tool 'Bash' is not valid JSON Schema",
        ):
            _app(config)

    def test_engine_is_iorails_without_the_engine_switch(self):
        """The server builds an IORails engine of its own, with `NEMO_GUARDRAILS_IORAILS_ENGINE` unset."""
        assert isinstance(_app(HOOKS_CONFIG).state.engine.rails_engine, IORails)

    def test_engine_reports_usage_as_an_api_deployment(self, monkeypatch):
        """The deployment type is `api` by the time the engine reports usage, which it does when built."""
        deployment_types = []
        monkeypatch.setattr(
            telemetry,
            "report_usage",
            lambda *args, **kwargs: deployment_types.append(telemetry._deployment_type_override),
        )

        _app(HOOKS_CONFIG)

        assert deployment_types == [telemetry.DeploymentTypeEnum.API]

    def test_rails_that_deny_every_call_are_warned_about(self, caplog):
        """The allowlist with no tools declared, and a `per_tool` rail on an undeclared tool, each log a warning."""
        config = {
            "models": [MAIN_MODEL],
            "rails": {
                "tool_call": {
                    "flows": ["tool call validation"],
                    "per_tool": {"Bash": ["regex check tool call"], "Write": []},
                }
            },
        }

        _app(config)

        assert [
            (record.levelno, record.getMessage()) for record in caplog.records if record.name == hooks_app.__name__
        ] == [
            (
                logging.WARNING,
                "`tool call validation` is in rails.tool_call.flows, but the `main` model declares no tools, "
                "so every tool call is denied",
            ),
            (
                logging.WARNING,
                "rails.tool_call.per_tool has rails for 'Bash', but the `main` model doesn't declare that tool, "
                "so every call to it is denied",
            ),
        ]

    @pytest.mark.parametrize("config", [HOOKS_CONFIG, ALLOWLIST_CONFIG], ids=["per-tool", "allowlist"])
    def test_rails_on_declared_tools_log_no_warning(self, caplog, config):
        """A config whose rails all check declared tools logs no warning."""
        _app(config)

        assert [record for record in caplog.records if record.name == hooks_app.__name__] == []

    def test_engine_starts_with_the_server_and_stops_with_it(self, monkeypatch):
        """The engine starts before the server answers a request, and stops when the server shuts down."""
        app = _app(HOOKS_CONFIG)
        engine = app.state.engine
        events = []
        monkeypatch.setattr(engine, "startup", AsyncMock(side_effect=lambda: events.append("startup")))
        monkeypatch.setattr(engine, "shutdown", AsyncMock(side_effect=lambda: events.append("shutdown")))

        with TestClient(app) as client:
            events.append("serving")
            client.get("/v1/health")

        assert events == ["startup", "serving", "shutdown"]

    def test_engine_that_fails_to_start_stops_the_server(self, monkeypatch):
        """An engine that fails to start fails the server's startup, so it never answers."""
        app = _app(HOOKS_CONFIG)
        monkeypatch.setattr(app.state.engine, "startup", AsyncMock(side_effect=RuntimeError("Failed to start engine")))

        with pytest.raises(RuntimeError, match="Failed to start engine"):
            with TestClient(app):
                pass


def _example_app(agent: HookAgent):
    """A hooks server for *agent*'s example config, `examples/configs/agent_hooks/<harness>`."""
    return create_hooks_app(RailsConfig.from_path(str(EXAMPLE_CONFIGS_PATH / EXAMPLE_CONFIGS[agent])))


class TestExampleConfig:
    def test_there_is_one_example_per_agent(self):
        """Each agent path has an example config of its own, and there's no other example config."""
        examples = sorted(path.name for path in EXAMPLE_CONFIGS_PATH.iterdir() if path.is_dir())

        assert examples == sorted(EXAMPLE_CONFIGS.values())

    @pytest.mark.parametrize("agent", AGENTS)
    def test_example_config_starts_without_warnings(self, caplog, agent):
        """Each example declares every tool it has rails for, so the server starts without a warning."""
        _example_app(agent)

        assert [record for record in caplog.records if record.name == hooks_app.__name__] == []

    @pytest.mark.parametrize(
        "agent, tool_name, tool_input, reason",
        [
            ("claude-code", "Bash", {"command": "npm test"}, None),
            ("claude-code", "Bash", {"command": "git push --force origin main"}, "regex check tool call"),
            ("claude-code", "Write", {"file_path": "C:\\project\\.env.local", "content": "x"}, "regex check tool call"),
            ("claude-code", "Read", {"file_path": "/home/user/project/.ENV"}, "regex check tool call"),
            (
                "claude-code",
                "Edit",
                {"file_path": "/home/user/src/index.ts", "old_string": "a", "new_string": "b"},
                None,
            ),
            ("claude-code", "TodoWrite", {"todos": []}, "tool call 'TodoWrite' is not an allowed tool"),
            (
                "claude-code",
                "apply_patch",
                CODEX_APPLY_PATCH["tool_input"],
                "tool call 'apply_patch' is not an allowed tool",
            ),
            ("codex", "Bash", {"command": "rm -rf /tmp/build"}, "regex check tool call"),
            ("codex", "apply_patch", CODEX_APPLY_PATCH["tool_input"], None),
            (
                "codex",
                "apply_patch",
                {"command": "*** Begin Patch\n*** Update File: config/.env\n@@\n-a\n+b\n*** End Patch\n"},
                "regex check tool call",
            ),
            (
                "codex",
                "apply_patch",
                {
                    "command": "*** Begin Patch\n*** Update File: notes.txt\n*** Move to: config/.env\n"
                    "@@\n-a\n+b\n*** End Patch\n"
                },
                "regex check tool call",
            ),
            (
                "codex",
                "Write",
                {"file_path": "/home/user/notes.txt", "content": "x"},
                "tool call 'Write' is not an allowed tool",
            ),
            ("nemo-relay", "search", NEMO_RELAY_SEARCH["tool_input"], None),
            (
                "nemo-relay",
                "search",
                NEMO_RELAY_EMPTY_QUERY["tool_input"],
                "arguments for tool 'search' do not match its schema: 'minLength' failed at '/query'",
            ),
            (
                "nemo-relay",
                "search",
                {},
                "arguments for tool 'search' do not match its schema: 'required' failed at '/query'",
            ),
            ("nemo-relay", "Bash", {"command": "ls"}, "tool call 'Bash' is not an allowed tool"),
        ],
        ids=[
            "claude-code-safe-command",
            "claude-code-force-push",
            "claude-code-secrets-file-on-windows",
            "claude-code-secrets-file-in-upper-case",
            "claude-code-source-file-edit",
            "claude-code-undeclared-tool",
            "claude-code-codex-tool",
            "codex-destructive-command",
            "codex-safe-patch",
            "codex-patch-to-secrets-file",
            "codex-patch-renaming-to-secrets-file",
            "codex-claude-code-tool",
            "nemo-relay-search",
            "nemo-relay-empty-query",
            "nemo-relay-no-query",
            "nemo-relay-undeclared-tool",
        ],
    )
    def test_example_config_rails(self, agent, tool_name, tool_input, reason):
        """Each agent's example allows safe calls to its own tools and denies the calls its rails target."""
        with TestClient(_example_app(agent)) as client:
            response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, tool_name, tool_input))

        expected = _allowed(agent) if reason is None else _denied(agent, reason)
        assert (response.status_code, response.json()) == (200, expected)


class TestRoutes:
    @pytest.mark.parametrize("path", ["/v1/health", "/healthz"])
    def test_health_check_answers_like_the_main_server(self, client, path):
        """Both health routes answer `200 {"status": "pass"}` as `application/health+json`, as the main server does."""
        response = client.get(path)

        assert (response.status_code, response.headers["content-type"], response.json()) == (
            200,
            "application/health+json",
            {"status": "pass"},
        )

    @pytest.mark.parametrize(
        "method, path",
        [
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/checks"),
            ("POST", "/v1/hooks/gemini"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
        ],
    )
    def test_no_other_route_exists(self, client, method, path):
        """The server answers only the hooks and the health checks: any other path is a `404`."""
        response = client.request(method, path)

        assert response.status_code == 404
