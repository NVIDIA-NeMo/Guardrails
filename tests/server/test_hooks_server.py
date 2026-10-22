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
from typing import Any, Optional
from unittest.mock import ANY, AsyncMock, MagicMock

import jsonschema
import pytest
from fastapi.testclient import TestClient

from nemoguardrails import telemetry
from nemoguardrails.exceptions import LLMCallException, NonStreamingWorkQueueFullError, RailTypeNotConfiguredError
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.model_engine import ModelEngine, ModelEngineError
from nemoguardrails.http.errors import HTTPConnectionError
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailType
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


# Per-tool rails alone, with no model and no tool lists: every tool is allowed, and Bash's rail checks its calls.
HOOKS_CONFIG = {
    "models": [],
    "rails": {
        "config": {"regex_detection": {"tool_call": {"Bash": {"patterns": [r"rm\s+-rf"]}}}},
        "tool_call": {"per_tool": {"Bash": ["regex check tool call $argument=command"]}},
    },
}

# The rail raises on a `$argument=` its tool's allowed_tools entry doesn't list, so every `Read` call fails it.
RAIL_FAILURE_CONFIG = {
    "models": [],
    "hooks_server": {"allowed_tools": [{"name": "Read", "arguments": ["file_path"]}]},
    "rails": {"tool_call": {"per_tool": {"Read": ["regex check tool call $argument=x"]}}},
}

# A tool policy alone, with no rails: the server decides every call itself.
ALLOWLIST_CONFIG = {"models": [], "hooks_server": {"allowed_tools": ["Bash"], "allow_unlisted_tools": False}}

# Blocks WebFetch by name, and a Bash call that passes `run_in_background`; no rails.
BLOCKLIST_CONFIG = {
    "models": [],
    "hooks_server": {"blocked_tools": ["WebFetch", {"name": "Bash", "arguments": ["run_in_background"]}]},
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

UNUSED_MAIN_MODEL = (
    "The config has a `main` model, which the hooks server doesn't use: it checks tool calls and never generates a "
    "response, and no tool-call rail names `$model=main`. Remove the `main` model, or give the hooks server a config "
    "of its own."
)
IGNORED_MAIN_MODEL_TOOLS = (
    "The `main` model declares tools in parameters.tools, which the hooks server ignores: it checks each tool call "
    "against hooks_server.allowed_tools, blocked_tools and allow_unlisted_tools only."
)
MAIN_MODEL_TOOLS_WITHOUT_TOOL_LISTS = (
    "The `main` model declares tools in parameters.tools, which the hooks server ignores, and the hooks_server "
    "section sets no tool policy that can deny a call, so every tool call would be allowed. List the tools in "
    "hooks_server.allowed_tools with allow_unlisted_tools: false, or remove parameters.tools."
)
NOTHING_TO_DENY = (
    "The hooks server needs a tool policy in the hooks_server section, or per-tool rails under "
    "rails.tool_call.per_tool; this config has neither, so every tool call would be allowed."
)
ALLOWED_NAMES_DENY_NOTHING = (
    "hooks_server.allowed_tools names tools by name only while hooks_server.allow_unlisted_tools is true, so the tool "
    "policy denies nothing, and with no per-tool rails every tool call would be allowed; set "
    "hooks_server.allow_unlisted_tools: false."
)
TOOL_CALL_VALIDATION_ON_THE_HOOKS_SERVER = (
    "rails.tool_call.flows lists 'tool call validation', which can't deny a call on the hooks server: the server "
    "applies its tool policy and declares the called tool itself, so only per-tool rails check a call. Restrict tools "
    "with hooks_server.allowed_tools and hooks_server.allow_unlisted_tools: false, and remove the flow."
)

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


def _main_model_rail_config(main_model: dict, hooks_server: Optional[dict] = None, **tool_call: Any) -> dict:
    """A config whose per-tool rail on Bash calls *main_model* as `$model=main`, plus *hooks_server* and *tool_call*."""
    per_tool = {"Bash": ["tool safety check tool call $model=main $variant=bash"]}
    return {
        "models": [main_model],
        "hooks_server": hooks_server or {},
        "rails": {"tool_call": {"per_tool": per_tool, **tool_call}},
        "prompts": [
            {
                "task": "tool_safety_check_tool_call $model=main $variant=bash",
                "content": "Is this call to {{ tool_name }} safe? {{ tool_call_arguments }}",
                "output_parser": "parse_tool_safety_verdict",
            }
        ],
    }


def _hooks_server_warnings(caplog) -> list[str]:
    """The warnings the hooks server logged."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == hooks_app.__name__ and record.levelno == logging.WARNING
    ]


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

    @pytest.mark.parametrize("agent", AGENTS)
    @pytest.mark.parametrize("tool_input", ["rm -rf /tmp/build", ["rm", "-rf"], None])
    @pytest.mark.parametrize("config", [HOOKS_CONFIG, ALLOWLIST_CONFIG], ids=["per-tool-rails", "policy-only"])
    def test_input_that_is_not_an_object_is_denied_as_malformed(self, agent, tool_input, config):
        """A `tool_input` that isn't a JSON object is denied by the server, naming the tool, with or without rails."""
        with TestClient(_app(config)) as client:
            response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", tool_input))

        reason = "tool call 'Bash' has arguments that aren't a JSON object"
        assert (response.status_code, response.json()) == (200, _denied(agent, reason))

    @pytest.mark.parametrize(
        ("tool_name", "expected"),
        [("Bash", {}), ("WebFetch", DENY_NOT_ALLOWED)],
        ids=["allowed", "unlisted"],
    )
    def test_policy_only_config_decides_without_the_rails(self, monkeypatch, tool_name, expected):
        """With a tool policy and no rails, the policy alone decides each call, and no check runs."""
        with TestClient(_app(ALLOWLIST_CONFIG)) as client:
            check = AsyncMock(wraps=client.app.state.engine.check_async)
            monkeypatch.setattr(client.app.state.engine, "check_async", check)

            response = client.post("/v1/hooks/claude-code", json=_hook("claude-code", tool_name, {"url": "x"}))

        assert (response.status_code, response.json()) == (200, expected)
        check.assert_not_awaited()

    @pytest.mark.parametrize("agent", AGENTS)
    def test_tool_without_rails_is_allowed(self, client, monkeypatch, agent):
        """A call to a tool with no per-tool rail is allowed without a check, though another tool has one."""
        check = AsyncMock(wraps=client.app.state.engine.check_async)
        monkeypatch.setattr(client.app.state.engine, "check_async", check)

        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Read", {"file_path": "/etc/passwd"}))

        assert (response.status_code, response.json()) == (200, _allowed(agent))
        check.assert_not_awaited()

    @pytest.mark.parametrize("agent", AGENTS)
    @pytest.mark.parametrize(
        "tool_name, tool_input, reason",
        [
            ("WebFetch", {"url": "https://example.com"}, "tool call 'WebFetch' is a blocked tool"),
            (
                "Bash",
                {"command": "npm test", "run_in_background": True},
                "tool call 'Bash' passes argument 'run_in_background', which is blocked",
            ),
            ("Bash", {"command": "npm test"}, None),
        ],
        ids=["tool-blocked", "argument-blocked", "no-blocked-argument"],
    )
    def test_blocked_tools_deny_a_tool_or_one_of_its_arguments(self, agent, tool_name, tool_input, reason):
        """blocked_tools denies a tool by name, or a call passing one of its blocked arguments, naming either."""
        with TestClient(_app(BLOCKLIST_CONFIG)) as client:
            response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, tool_name, tool_input))

        expected = _allowed(agent) if reason is None else _denied(agent, reason)
        assert (response.status_code, response.json()) == (200, expected)

    def test_call_the_tool_policy_denies_is_answered_without_the_rails(self, monkeypatch):
        """A call the tool policy denies gets the agent's deny without a check, so no rail runs on it."""
        with TestClient(_app({**HOOKS_CONFIG, "hooks_server": BLOCKLIST_CONFIG["hooks_server"]})) as client:
            check = AsyncMock(wraps=client.app.state.engine.check_async)
            monkeypatch.setattr(client.app.state.engine, "check_async", check)

            response = client.post(
                "/v1/hooks/claude-code", json=_hook("claude-code", "WebFetch", {"url": "https://example.com"})
            )

        assert (response.status_code, response.json()) == (
            200,
            _denied("claude-code", "tool call 'WebFetch' is a blocked tool"),
        )
        check.assert_not_awaited()

    def test_call_the_tool_policy_denies_logs_the_tool_policy_as_what_decided(self, caplog):
        """A call the tool policy denies logs one line naming the tool policy, and never the tool input."""
        with TestClient(_app(BLOCKLIST_CONFIG)) as client:
            caplog.set_level(logging.INFO)
            client.post(
                "/v1/hooks/claude-code", json=_hook("claude-code", "WebFetch", {"url": "https://secret-7f3a.example"})
            )

        assert [record.getMessage() for record in caplog.records if record.name == hooks_app.__name__] == [
            "Hook check on /v1/hooks/claude-code: event=PreToolUse tool_name='WebFetch' "
            "tool_use_id='toolu_01GHI789...' outcome=deny code=None rail='tool policy'"
        ]
        assert "secret-7f3a" not in caplog.text

    @pytest.mark.parametrize(
        "tool_name, expected",
        [("WebFetch", DENY_NOT_ALLOWED), ("Bash", {})],
        ids=["declared-on-main-but-not-allowed", "allowed-but-not-declared-on-main"],
    )
    def test_main_model_tools_play_no_part_in_the_check(self, tool_name, expected):
        """Only the tool lists decide: a tool the `main` model declares is denied, and one it doesn't is allowed."""
        config = _main_model_rail_config(
            _main_model(_tool("WebFetch", ANY_ARGUMENTS)),
            hooks_server={"allowed_tools": ["Bash", "Read"], "allow_unlisted_tools": False},
            # The rail that uses `main` is on Read, so these checks never call the model.
            per_tool={"Read": ["tool safety check tool call $model=main $variant=bash"]},
        )

        with TestClient(_app(config)) as client:
            response = client.post("/v1/hooks/claude-code", json=_hook("claude-code", tool_name, {"url": "x"}))

        assert (response.status_code, response.json()) == (200, expected)

    def test_check_offers_the_called_tool_on_the_existing_check_arguments(self, monkeypatch):
        """A call is checked with tool-call rails and the called tool's declaration as `tools`, and nothing else."""
        with TestClient(_app({**HOOKS_CONFIG, "hooks_server": ALLOWLIST_CONFIG["hooks_server"]})) as client:
            engine = client.app.state.engine
            check = AsyncMock(wraps=engine.check_async)
            monkeypatch.setattr(engine, "check_async", check)

            client.post("/v1/hooks/claude-code", json=_hook("claude-code", "Bash", {"command": "npm test"}))

        check.assert_awaited_once_with(ANY, rail_types=[RailType.TOOL_CALL], tools=[_tool("Bash", ANY_ARGUMENTS)])

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
        """A tool allowed_tools doesn't list, while unlisted tools are denied, gets `denyNotAllowed`."""
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
        response = failing_check(exc).post("/v1/hooks/codex", json=_hook("codex", "Bash", {"command": "npm test"}))

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
        monkeypatch.setattr(
            "nemoguardrails.guardrails.rails_manager._latest_tool_calls_openai",
            MagicMock(side_effect=RuntimeError("a bug")),
        )

        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", {"command": "npm test"}))

        assert (response.status_code, response.json()) == (500, CHECK_FAILED)

    def test_full_admission_queue_is_retryable_queue_full(self, failing_check):
        """A full admission queue gets the spec's `503 queue_full`, with `Retry-After`."""
        response = failing_check(NonStreamingWorkQueueFullError()).post("/v1/hooks/claude-code", json=CLAUDE_CODE_BASH)

        assert (response.status_code, response.headers["retry-after"], response.json()) == (503, "30", OVERLOADED)

    def test_full_admission_queue_leaves_a_tool_without_rails_allowed(self, failing_check):
        """A call to a tool with no per-tool rail is allowed while the admission queue is full: it isn't queued."""
        client = failing_check(NonStreamingWorkQueueFullError())

        response = client.post("/v1/hooks/claude-code", json=_hook("claude-code", "Read", {"file_path": "README.md"}))

        assert (response.status_code, response.json()) == (200, _allowed("claude-code"))

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("unexpected"), RailTypeNotConfiguredError("Requested rail type 'tool_call' has no rails.")],
        ids=["RuntimeError", "RailTypeNotConfiguredError"],
    )
    def test_unexpected_failure_is_internal_error(self, failing_check, exc):
        """Any other failure, a library `ValueError` included, gets the spec's `500` internal error."""
        response = failing_check(exc).post(
            "/v1/hooks/nemo-relay", json=_hook("nemo-relay", "Bash", {"command": "npm test"})
        )

        assert (response.status_code, response.json()) == (500, INTERNAL_ERROR)


class TestStartup:
    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({"models": []}, NOTHING_TO_DENY),
            ({"models": [], "rails": {"tool_call": {"per_tool": {"Bash": []}}}}, NOTHING_TO_DENY),
            ({"models": [], "hooks_server": {"allowed_tools": ["Bash", "Read"]}}, ALLOWED_NAMES_DENY_NOTHING),
        ],
        ids=["nothing", "per-tool-entry-without-rails", "allowed-names-only"],
    )
    def test_config_that_can_deny_no_call_is_refused_before_the_engine(self, monkeypatch, config, message):
        """A config with no tool policy that can deny and no per-tool rails is refused, since it allows every call."""
        engine = MagicMock()
        monkeypatch.setattr(hooks_app, "Guardrails", engine)

        with pytest.raises(HooksConfigError) as exc_info:
            _app(config)

        assert str(exc_info.value) == message
        engine.assert_not_called()

    def test_allowed_names_beside_per_tool_rails_start(self):
        """Bare allowed_tools names with unlisted tools allowed start when per-tool rails can deny a call."""
        config = {**HOOKS_CONFIG, "hooks_server": {"allowed_tools": ["Bash", "Read"]}}

        assert isinstance(_app(config).state.engine.rails_engine, IORails)

    def test_config_with_tool_call_validation_is_refused_before_the_engine(self, monkeypatch):
        """`tool call validation` is refused, since the server declares the called tool and the flow can't deny."""
        engine = MagicMock()
        monkeypatch.setattr(hooks_app, "Guardrails", engine)

        with pytest.raises(HooksConfigError) as exc_info:
            _app({**ALLOWLIST_CONFIG, "rails": {"tool_call": {"flows": ["tool call validation"]}}})

        assert str(exc_info.value) == TOOL_CALL_VALIDATION_ON_THE_HOOKS_SERVER
        engine.assert_not_called()

    @pytest.mark.parametrize(
        ("hooks", "message"),
        [
            (
                {"blocked_tools": ["Bash"]},
                "rails.tool_call.per_tool has rails for tool 'Bash', which hooks_server.blocked_tools blocks by name, "
                "so they never run; remove its per_tool entry, or unblock it.",
            ),
            (
                {"allowed_tools": ["Read"], "allow_unlisted_tools": False},
                "rails.tool_call.per_tool has rails for tool 'Bash', which hooks_server.allowed_tools doesn't list "
                "while hooks_server.allow_unlisted_tools is false, so they never run; add it to "
                "hooks_server.allowed_tools.",
            ),
        ],
        ids=["blocked-by-name", "unlisted-denied"],
    )
    def test_per_tool_rails_the_tool_policy_never_lets_run_are_refused_before_the_engine(
        self, monkeypatch, hooks, message
    ):
        """per_tool rails on a tool the tool policy denies every call to are refused, since they would never run."""
        engine = MagicMock()
        monkeypatch.setattr(hooks_app, "Guardrails", engine)

        with pytest.raises(HooksConfigError) as exc_info:
            _app({**HOOKS_CONFIG, "hooks_server": hooks})

        assert str(exc_info.value) == message
        engine.assert_not_called()

    def test_per_tool_entry_without_rails_on_a_blocked_tool_starts(self):
        """A per_tool entry with no rails on a tool blocked by name starts, since it has no rails to skip."""
        config = {
            "models": [],
            "hooks_server": {"blocked_tools": ["Bash"]},
            "rails": {"tool_call": {"per_tool": {"Bash": []}}},
        }

        assert isinstance(_app(config).state.engine.rails_engine, IORails)

    def test_per_tool_rails_beside_a_blocked_argument_start(self):
        """per_tool rails on a tool whose blocked_tools entry blocks only an argument start, since they run."""
        config = {
            **HOOKS_CONFIG,
            "hooks_server": {"blocked_tools": [{"name": "Bash", "arguments": ["run_in_background"]}]},
        }

        assert isinstance(_app(config).state.engine.rails_engine, IORails)

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

    @pytest.mark.parametrize(
        "main_model",
        [MAIN_MODEL, _main_model(_tool("Bash", ANY_ARGUMENTS))],
        ids=["main-model", "main-model-with-tools"],
    )
    def test_main_model_no_rail_names_is_refused_before_the_engine(self, monkeypatch, main_model):
        """A `main` model no tool-call rail names is refused, since the server never uses it."""
        engine = MagicMock()
        monkeypatch.setattr(hooks_app, "Guardrails", engine)

        with pytest.raises(HooksConfigError) as exc_info:
            _app({**ALLOWLIST_CONFIG, "models": [main_model]})

        assert str(exc_info.value) == UNUSED_MAIN_MODEL
        engine.assert_not_called()

    @pytest.mark.parametrize("hooks", [None, {"allowed_tools": ["Bash"]}], ids=["no-policy", "policy-denying-nothing"])
    def test_main_model_tools_without_tool_lists_are_refused_before_the_engine(self, monkeypatch, hooks):
        """Tools on a `main` model a rail names, with no policy that can deny, are refused, as every tool is allowed."""
        engine = MagicMock()
        monkeypatch.setattr(hooks_app, "Guardrails", engine)

        with pytest.raises(HooksConfigError) as exc_info:
            _app(_main_model_rail_config(_main_model(_tool("Bash", ANY_ARGUMENTS)), hooks_server=hooks))

        assert str(exc_info.value) == MAIN_MODEL_TOOLS_WITHOUT_TOOL_LISTS
        engine.assert_not_called()

    def test_main_model_a_rail_names_starts_without_a_warning(self, caplog):
        """A `main` model a tool-call rail names with `$model=main` is used, so the server starts without a warning."""
        _app(_main_model_rail_config(MAIN_MODEL))

        assert _hooks_server_warnings(caplog) == []

    def test_tools_on_a_main_model_a_rail_names_are_warned_about_with_tool_lists(self, caplog):
        """With tool lists set, tools on a `main` model a rail names only log that the server ignores them."""
        config = _main_model_rail_config(
            _main_model(_tool("Bash", ANY_ARGUMENTS)),
            hooks_server={"allowed_tools": ["Bash"], "allow_unlisted_tools": False},
        )

        _app(config)

        assert _hooks_server_warnings(caplog) == [IGNORED_MAIN_MODEL_TOOLS]

    @pytest.mark.parametrize("config", [HOOKS_CONFIG, ALLOWLIST_CONFIG], ids=["per-tool", "allowlist"])
    def test_config_without_a_main_model_logs_no_warning(self, caplog, config):
        """A config with no `main` model starts without a warning."""
        _app(config)

        assert _hooks_server_warnings(caplog) == []

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
        """Each example has no model and passes config validation, so the server starts without a warning."""
        _example_app(agent)

        assert _hooks_server_warnings(caplog) == []

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
            ("nemo-relay", "search", NEMO_RELAY_EMPTY_QUERY["tool_input"], "regex check tool call"),
            ("nemo-relay", "search", {}, "regex check tool call"),
            ("nemo-relay", "search", {"query": 7}, "regex check tool call"),
            (
                "nemo-relay",
                "search",
                {"query": "weather", "limit": 3},
                "tool call 'search' passes 1 argument its allowed_tools entry doesn't list",
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
            "nemo-relay-query-not-a-string",
            "nemo-relay-unlisted-argument",
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
