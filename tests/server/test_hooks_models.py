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

import dataclasses
import json
from typing import Any, Optional, get_args

import jsonschema
import pytest
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailsResult, RailStatus, ToolViolation
from nemoguardrails.server.hooks import (
    ADAPTERS,
    HookAgent,
    HookErrorCode,
    HookOutcome,
    UnsupportedHookEventError,
    check_messages,
    check_tools,
    outcome_from_result,
)
from nemoguardrails.server.hooks.checking import ToolInvocation
from nemoguardrails.server.hooks.schemas import (
    ClaudeCodePreToolUseRequest,
    CodexPreToolUseRequest,
    NemoRelayToolConditionalExecutionGuardrailRequest,
)
from nemoguardrails.server.hooks.tool_policy import ToolPolicy
from tests.guardrails.tool_helpers import call_violation

CLAUDE_CODE_BASH = {
    "session_id": "abc123",
    "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
    "transcript_path": "/home/user/.claude/projects/.../transcript.jsonl",
    "cwd": "/home/user/my-project",
    "scratchpad_dir": "/tmp/claude-1000/-home-user-my-project/abc123/scratchpad",
    "permission_mode": "default",
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {
        "command": "npm test",
        "description": "Run test suite",
        "timeout": 120000,
        "run_in_background": False,
    },
    "tool_use_id": "toolu_01ABC123...",
}

CLAUDE_CODE_DESTRUCTIVE_BASH = {
    "session_id": "abc123",
    "transcript_path": "/home/user/.claude/projects/.../transcript.jsonl",
    "cwd": "/home/user/my-project",
    "permission_mode": "default",
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": "rm -rf /tmp/build"},
    "tool_use_id": "toolu_01GHI789...",
}

CLAUDE_CODE_WRITE_WINDOWS = {
    "session_id": "abc123",
    "transcript_path": "C:\\Users\\user\\.claude\\projects\\...\\transcript.jsonl",
    "cwd": "C:\\project",
    "permission_mode": "default",
    "hook_event_name": "PreToolUse",
    "tool_name": "Write",
    "tool_input": {"file_path": "C:\\project\\src\\index.ts", "content": "..."},
    "tool_use_id": "toolu_01JKL012...",
}

CLAUDE_CODE_MCP = {
    "session_id": "abc123",
    "transcript_path": "/home/user/.claude/projects/.../transcript.jsonl",
    "cwd": "/home/user/my-project",
    "permission_mode": "default",
    "hook_event_name": "PreToolUse",
    "tool_name": "mcp__github__search_repositories",
    "tool_input": {"query": "nemo guardrails"},
    "tool_use_id": "toolu_01DEF456...",
    "mcp_server": {"name": "github", "source": "user"},
}

CODEX_APPLY_PATCH = {
    "session_id": "<session id>",
    "transcript_path": None,
    "cwd": "<working directory>",
    "hook_event_name": "PreToolUse",
    "model": "<active model slug>",
    "permission_mode": "default",
    "turn_id": "<turn id>",
    "tool_name": "apply_patch",
    "tool_input": {
        "command": "*** Begin Patch\n"
        "*** Add File: nested/new.txt\n"
        "+created\n"
        "*** Delete File: delete.txt\n"
        "*** Update File: modify.txt\n"
        "@@\n"
        "-line2\n"
        "+changed\n"
        "*** End Patch\n"
    },
    "tool_use_id": "<tool call id>",
}

CODEX_MCP = {
    "session_id": "<session id>",
    "transcript_path": None,
    "cwd": "<working directory>",
    "hook_event_name": "PreToolUse",
    "model": "<active model slug>",
    "permission_mode": "default",
    "turn_id": "<turn id>",
    "tool_name": "mcp__fs__read",
    "tool_input": {"file_path": "src/main.rs", "count": 3},
    "tool_use_id": "<tool call id>",
}

NEMO_RELAY_SEARCH = {
    "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
    "tool_name": "search",
    "tool_input": {"query": "weather"},
}

NEMO_RELAY_EMPTY_QUERY = {
    "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
    "tool_name": "search",
    "tool_input": {"query": ""},
}

CLAUDE_CODE_EXAMPLES = [CLAUDE_CODE_BASH, CLAUDE_CODE_DESTRUCTIVE_BASH, CLAUDE_CODE_WRITE_WINDOWS, CLAUDE_CODE_MCP]
CODEX_EXAMPLES = [CODEX_APPLY_PATCH, CODEX_MCP]

PRE_TOOL_USE_REQUIRED = ["hook_event_name", "tool_name", "tool_input", "tool_use_id"]
NEMO_RELAY_REQUIRED = ["hook_event_name", "tool_name", "tool_input"]

# Codex's generated PreToolUse hook schemas, copied unchanged from openai/codex@e444aa99:
# https://github.com/openai/codex/tree/e444aa99d76beeb8b5b0610b793c0cdefe58da58/codex-rs/hooks/schema/generated
# OpenAI Codex, Copyright 2025 OpenAI, licensed under the Apache License, Version 2.0.
# Refresh both from one Codex commit, and update that commit here.

# pre-tool-use.command.input.schema.json: what Codex sends to a PreToolUse hook.
CODEX_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "additionalProperties": False,
    "definitions": {"NullableString": {"type": ["string", "null"]}},
    "properties": {
        "agent_id": {"type": "string"},
        "agent_type": {"type": "string"},
        "cwd": {"type": "string"},
        "hook_event_name": {"const": "PreToolUse", "type": "string"},
        "model": {"type": "string"},
        "permission_mode": {
            "enum": ["default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"],
            "type": "string",
        },
        "session_id": {"type": "string"},
        "tool_input": True,
        "tool_name": {"type": "string"},
        "tool_use_id": {"type": "string"},
        "transcript_path": {"$ref": "#/definitions/NullableString"},
        "turn_id": {
            "description": "Codex extension: expose the active turn id to internal turn-scoped hooks.",
            "type": "string",
        },
    },
    "required": [
        "cwd",
        "hook_event_name",
        "model",
        "permission_mode",
        "session_id",
        "tool_input",
        "tool_name",
        "tool_use_id",
        "transcript_path",
        "turn_id",
    ],
    "title": "pre-tool-use.command.input",
    "type": "object",
}

# pre-tool-use.command.output.schema.json: what Codex accepts back.
CODEX_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "additionalProperties": False,
    "definitions": {
        "PreToolUseDecisionWire": {"enum": ["approve", "block"], "type": "string"},
        "PreToolUseHookSpecificOutputWire": {
            "additionalProperties": False,
            "properties": {
                "additionalContext": {"default": None, "type": "string"},
                "hookEventName": {"const": "PreToolUse", "type": "string"},
                "permissionDecision": {
                    "allOf": [{"$ref": "#/definitions/PreToolUsePermissionDecisionWire"}],
                    "default": None,
                },
                "permissionDecisionReason": {"default": None, "type": "string"},
                "updatedInput": {"default": None},
            },
            "required": ["hookEventName"],
            "type": "object",
        },
        "PreToolUsePermissionDecisionWire": {"enum": ["allow", "deny", "ask"], "type": "string"},
    },
    "properties": {
        "continue": {"default": True, "type": "boolean"},
        "decision": {"allOf": [{"$ref": "#/definitions/PreToolUseDecisionWire"}], "default": None},
        "hookSpecificOutput": {"allOf": [{"$ref": "#/definitions/PreToolUseHookSpecificOutputWire"}], "default": None},
        "reason": {"default": None, "type": "string"},
        "stopReason": {"default": None, "type": "string"},
        "suppressOutput": {"default": False, "type": "boolean"},
        "systemMessage": {"default": None, "type": "string"},
    },
    "title": "pre-tool-use.command.output",
    "type": "object",
}

REFUSAL = "I'm sorry, I can't respond to that."
INTERNAL_ERROR = "I'm sorry, an internal error has occurred."

UNDECLARED_TOOL = call_violation(
    "tool_not_allowed",
    "tool call 'WebFetch' is not an allowed tool",
    tool_call_id="toolu_01",
    tool_name="WebFetch",
    index=0,
)
PER_TOOL_BLOCK = call_violation(
    "per_tool_rail",
    "regex check tool call",
    tool_call_id="toolu_01",
    tool_name="Bash",
    index=0,
    rail="regex check tool call",
)
MALFORMED_ARGUMENTS = call_violation(
    "malformed_arguments",
    "tool call 'toolu_01' has malformed arguments",
    tool_call_id="toolu_01",
    tool_name="Bash",
    index=0,
)
RAIL_FAILED = call_violation(
    "rail_failed",
    "regex check tool call error",
    tool_call_id="toolu_01",
    tool_name="Bash",
    index=0,
    rail="regex check tool call",
)
EXTRACTION_FAILED = call_violation("extraction_failed", "tool call extraction failed")
INVALID_TOOLSET = call_violation("invalid_toolset", "tool parsing failed")
INVALID_TOOL_SCHEMA = call_violation(
    "invalid_tool_schema",
    "declared schema for tool 'Bash' is not valid JSON Schema",
    tool_call_id="toolu_01",
    tool_name="Bash",
    index=0,
)
EMPTY_REASON_BLOCK = call_violation(
    "per_tool_rail",
    "",
    tool_call_id="toolu_01",
    tool_name="Bash",
    index=0,
    rail="custom rail",
)


def _without(body: dict, field: str) -> dict:
    """*body* with *field* removed."""
    return {key: value for key, value in body.items() if key != field}


def _validation_errors(agent: HookAgent, body: Any) -> list[ErrorDetails]:
    """The pydantic errors *body* raises on *agent*'s path."""
    with pytest.raises(ValidationError) as exc_info:
        ADAPTERS[agent].parse(body)
    return exc_info.value.errors()


def _blocked(
    *violations: ToolViolation, rail: Optional[str] = None, reason: Optional[str] = None, content: str = REFUSAL
) -> RailsResult:
    """A blocked tool-call check carrying *violations*, as IORails reports one."""
    return RailsResult(
        status=RailStatus.BLOCKED,
        content=content,
        rail=rail,
        reason=reason,
        tool_violations=list(violations) or None,
    )


class TestAdapters:
    def test_registry_has_one_adapter_per_agent_naming_that_agent(self):
        """The registry has an adapter for every agent path, and each adapter names the agent it is registered for."""
        assert {agent: adapter.agent for agent, adapter in ADAPTERS.items()} == {
            agent: agent for agent in get_args(HookAgent)
        }

    @pytest.mark.parametrize("agent", get_args(HookAgent))
    def test_adapter_event_is_the_event_its_request_model_takes(self, agent):
        """Each adapter's event is the one `hook_event_name` its request model accepts."""
        adapter = ADAPTERS[agent]

        assert get_args(adapter.request_model.model_fields["hook_event_name"].annotation) == (adapter.event,)


class TestParse:
    @pytest.mark.parametrize(
        "agent, body, call_id",
        [
            *[("claude-code", body, body["tool_use_id"]) for body in CLAUDE_CODE_EXAMPLES],
            *[("codex", body, body["tool_use_id"]) for body in CODEX_EXAMPLES],
            *[("nemo-relay", body, None) for body in [NEMO_RELAY_SEARCH, NEMO_RELAY_EMPTY_QUERY]],
        ],
    )
    def test_spec_request_examples_parse_on_their_own_path(self, agent, body, call_id):
        """Every request example in the spec parses on its own path into the tool call it carries."""
        assert ADAPTERS[agent].parse(body) == ToolInvocation(
            name=body["tool_name"], arguments=body["tool_input"], call_id=call_id
        )

    @pytest.mark.parametrize(
        "agent, body",
        [
            *[("codex", body) for body in CLAUDE_CODE_EXAMPLES],
            *[("claude-code", body) for body in CODEX_EXAMPLES],
        ],
    )
    def test_payload_parses_on_the_other_agents_path(self, agent, body):
        """A Claude Code payload parses on the Codex path and the reverse, since both paths read the same fields."""
        assert ADAPTERS[agent].parse(body) == ToolInvocation(
            name=body["tool_name"], arguments=body["tool_input"], call_id=body["tool_use_id"]
        )

    @pytest.mark.parametrize("agent", ["claude-code", "codex"])
    def test_body_with_only_the_fields_the_server_reads_parses(self, agent):
        """A body with only `hook_event_name`, `tool_name`, `tool_input` and `tool_use_id` parses on either agent path."""
        body = {field: CLAUDE_CODE_BASH[field] for field in PRE_TOOL_USE_REQUIRED}

        assert ADAPTERS[agent].parse(body).call_id == "toolu_01ABC123..."

    @pytest.mark.parametrize(
        "agent, body, field",
        [
            *[("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH, field) for field in PRE_TOOL_USE_REQUIRED],
            *[("codex", CODEX_MCP, field) for field in PRE_TOOL_USE_REQUIRED],
            *[("nemo-relay", NEMO_RELAY_SEARCH, field) for field in NEMO_RELAY_REQUIRED],
        ],
    )
    def test_missing_required_field_fails_at_that_field(self, agent, body, field):
        """A body missing one required field fails with that field, and only that field, as the error."""
        errors = _validation_errors(agent, _without(body, field))

        assert [(error["loc"], error["type"]) for error in errors] == [((field,), "missing")]

    @pytest.mark.parametrize(
        "agent, body, field, value",
        [
            ("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH, "model", "<active model slug>"),
            ("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH, "turn_id", None),
            ("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH, "transcript_path", None),
            ("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH, "effort", "high"),
            ("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH, "mcp_server", "github"),
            ("codex", CODEX_MCP, "prompt_id", "550e8400-e29b-41d4-a716-446655440000"),
            ("codex", CODEX_MCP, "mcp_server", {"name": "github", "source": "user"}),
            ("codex", CODEX_MCP, "permission_mode", None),
            ("codex", CODEX_MCP, "agent_id", 7),
        ],
    )
    def test_field_the_server_does_not_read_parses_with_any_value(self, agent, body, field, value):
        """A field the server doesn't read parses with any value, even another agent's field, and leaves the call alone."""
        assert ADAPTERS[agent].parse({**body, field: value}) == ADAPTERS[agent].parse(body)

    def test_nemo_relay_rejects_unknown_field(self):
        """The NeMo Relay path rejects a field its schema doesn't list."""
        errors = _validation_errors("nemo-relay", {**NEMO_RELAY_SEARCH, "session_id": "abc123"})

        assert [(error["loc"], error["msg"]) for error in errors] == [
            (("session_id",), "Extra inputs are not permitted")
        ]

    @pytest.mark.parametrize("agent, body", [("claude-code", CLAUDE_CODE_DESTRUCTIVE_BASH), ("codex", CODEX_MCP)])
    def test_agent_paths_accept_unknown_fields(self, agent, body):
        """The Claude Code and Codex paths accept a field the spec doesn't list, so a new agent field is no error."""
        unknown_field = {"added_by_a_later_release": {"nested": 1}}

        assert ADAPTERS[agent].parse({**body, **unknown_field}) == ADAPTERS[agent].parse(body)

    @pytest.mark.parametrize("mode", ["python", "json"])
    @pytest.mark.parametrize(
        "model, body",
        [
            *[(ClaudeCodePreToolUseRequest, body) for body in CLAUDE_CODE_EXAMPLES],
            *[(CodexPreToolUseRequest, body) for body in CODEX_EXAMPLES],
            *[
                (NemoRelayToolConditionalExecutionGuardrailRequest, body)
                for body in [NEMO_RELAY_SEARCH, NEMO_RELAY_EMPTY_QUERY]
            ],
        ],
    )
    def test_dumped_request_validates_again_unchanged(self, model, body, mode):
        """A request's `model_dump()` validates again to an equal request, so a saved request reloads."""
        request = model.model_validate(body)

        assert model.model_validate(request.model_dump(mode=mode)) == request

    @pytest.mark.parametrize(
        "agent, body", [("claude-code", CLAUDE_CODE_BASH), ("codex", CODEX_MCP), ("nemo-relay", NEMO_RELAY_SEARCH)]
    )
    def test_empty_tool_name_fails(self, agent, body):
        """An empty `tool_name` fails at `tool_name` on every path."""
        errors = _validation_errors(agent, {**body, "tool_name": ""})

        assert [(error["loc"], error["type"]) for error in errors] == [(("tool_name",), "string_too_short")]

    @pytest.mark.parametrize(
        "agent, event",
        [
            ("claude-code", "PostToolUse"),
            ("codex", "PostToolUse"),
            ("nemo-relay", "PostToolUse"),
            ("claude-code", "NemoRelayToolConditionalExecutionGuardrail"),
            ("nemo-relay", "PreToolUse"),
        ],
    )
    def test_event_the_path_does_not_handle_raises_unsupported_hook_event(self, agent, event):
        """An event the path doesn't handle raises `UnsupportedHookEventError`, before the body is validated."""
        with pytest.raises(UnsupportedHookEventError) as exc_info:
            ADAPTERS[agent].parse({"hook_event_name": event})

        assert (exc_info.value.agent, exc_info.value.event) == (agent, event)

    def test_unsupported_hook_event_message_names_the_path_and_its_events(self):
        """The message names the event, the path and the event to send instead, as in the spec's example."""
        with pytest.raises(UnsupportedHookEventError) as exc_info:
            ADAPTERS["claude-code"].parse({**CLAUDE_CODE_BASH, "hook_event_name": "PostToolUse"})

        assert str(exc_info.value) == (
            "hook_event_name 'PostToolUse' is not handled on /v1/hooks/claude-code; send PreToolUse"
        )

    def test_unsupported_hook_event_message_cuts_and_escapes_the_event(self):
        """A long event name with line breaks reaches the message cut short and escaped."""
        with pytest.raises(UnsupportedHookEventError) as exc_info:
            ADAPTERS["codex"].parse({"hook_event_name": "Evil\n" * 1000})

        message = str(exc_info.value)
        assert "\n" not in message
        assert len(message) < 200

    @pytest.mark.parametrize("body", [[CLAUDE_CODE_BASH], "PreToolUse", None, 1])
    def test_body_that_is_not_an_object_fails_at_the_body(self, body):
        """A body that isn't a JSON object fails with an error at the body itself, naming no field."""
        errors = _validation_errors("claude-code", body)

        assert [error["loc"] for error in errors] == [()]

    @pytest.mark.parametrize("event", [5, None, ["PreToolUse"]])
    def test_non_string_event_fails_at_hook_event_name(self, event):
        """A `hook_event_name` that isn't a string fails at `hook_event_name`."""
        errors = _validation_errors("claude-code", {**CLAUDE_CODE_BASH, "hook_event_name": event})

        assert [(error["loc"], error["type"]) for error in errors] == [(("hook_event_name",), "string_type")]


class TestCheckMessages:
    def test_claude_code_call_becomes_one_assistant_tool_call(self):
        """A Claude Code hook becomes one assistant message whose one tool call carries the hook's id, name and input."""
        messages = check_messages(ADAPTERS["claude-code"].parse(CLAUDE_CODE_DESTRUCTIVE_BASH))

        assert messages == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_01GHI789...",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command": "rm -rf /tmp/build"}'},
                    }
                ],
            }
        ]

    def test_codex_call_carries_its_tool_use_id_and_json_encoded_input(self):
        """A Codex hook's call carries `tool_use_id` as its id, and `tool_input` encoded as JSON."""
        (tool_call,) = check_messages(ADAPTERS["codex"].parse(CODEX_APPLY_PATCH))[0]["tool_calls"]

        assert tool_call["id"] == "<tool call id>"
        assert tool_call["function"]["name"] == "apply_patch"
        assert json.loads(tool_call["function"]["arguments"]) == CODEX_APPLY_PATCH["tool_input"]

    def test_nemo_relay_call_has_no_id(self):
        """A NeMo Relay call has no id, since its guardrail callback receives none."""
        (tool_call,) = check_messages(ADAPTERS["nemo-relay"].parse(NEMO_RELAY_SEARCH))[0]["tool_calls"]

        assert tool_call == {
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "weather"}'},
        }

    def test_non_object_tool_input_is_encoded_as_sent(self):
        """A `tool_input` that isn't an object is encoded as sent, for the engine to deny as malformed arguments."""
        call = ADAPTERS["claude-code"].parse({**CLAUDE_CODE_DESTRUCTIVE_BASH, "tool_input": "rm -rf /"})

        (tool_call,) = check_messages(call)[0]["tool_calls"]

        assert tool_call["function"]["arguments"] == '"rm -rf /"'


def _tool_policy(**hooks: Any) -> ToolPolicy:
    """The hooks server's tool policy for a config with *hooks* as its hooks_server section."""
    return ToolPolicy.from_config(RailsConfig.from_content(config={"models": [], "hooks_server": hooks}).hooks_server)


class TestCheckTools:
    @pytest.mark.parametrize(
        "tool_lists",
        [
            {},
            {"allowed_tools": ["Bash"]},
            {"allowed_tools": ["Read"]},
            {"allowed_tools": ["Read"], "allow_unlisted_tools": False},
            {"blocked_tools": ["Bash"]},
            {"blocked_tools": [{"name": "Bash", "arguments": ["run_in_background"]}]},
        ],
        ids=[
            "no-lists",
            "allowed-by-name",
            "unlisted",
            "unlisted-denied",
            "blocked-by-name",
            "blocked-by-argument",
        ],
    )
    def test_called_tool_without_an_allowed_tools_entry_is_declared_with_any_arguments(self, tool_lists):
        """The called tool is the one tool offered, taking any arguments, unless an allowed_tools entry lists them."""
        call = ToolInvocation(name="Bash", arguments={"command": "npm test"})

        tools = check_tools(call, _tool_policy(**tool_lists))

        assert tools == [
            {
                "type": "function",
                "function": {"name": "Bash", "parameters": {"type": "object", "additionalProperties": True}},
            }
        ]

    def test_called_tool_with_an_allowed_tools_entry_is_declared_with_its_arguments(self):
        """A called tool with an allowed_tools entry is offered taking exactly the arguments the entry lists."""
        policy = _tool_policy(allowed_tools=["Bash", {"name": "search", "arguments": ["query", "limit"]}])
        call = ToolInvocation(name="search", arguments={"query": "weather"})

        tools = check_tools(call, policy)

        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {}, "limit": {}},
                        "additionalProperties": False,
                    },
                },
            }
        ]


class TestHookOutcome:
    @pytest.mark.parametrize(
        "fields",
        [
            {"kind": "allow"},
            {"kind": "deny", "reason": "regex check tool call"},
            {"kind": "error", "message": "rail 'x' failed: x error", "code": HookErrorCode.RAIL_FAILED},
            {"kind": "error", "message": "unexpected check status 'modified'"},
        ],
        ids=["allow", "deny", "error-with-code", "error-without-code"],
    )
    def test_valid_shape_of_each_kind_constructs(self, fields):
        """Each kind's valid shape constructs and keeps the fields it was given."""
        outcome = HookOutcome(**fields)

        assert dataclasses.asdict(outcome) == {"reason": None, "code": None, "message": None, **fields}

    @pytest.mark.parametrize(
        "fields",
        [
            {"kind": "allow", "reason": "regex check tool call"},
            {"kind": "allow", "code": HookErrorCode.RAIL_FAILED},
            {"kind": "allow", "message": "rail 'x' failed: x error"},
            {"kind": "deny"},
            {"kind": "deny", "reason": ""},
            {"kind": "deny", "reason": "regex check tool call", "code": HookErrorCode.RAIL_FAILED},
            {"kind": "deny", "reason": "regex check tool call", "message": "rail 'x' failed: x error"},
            {"kind": "error"},
            {"kind": "error", "message": ""},
            {"kind": "error", "message": "rail 'x' failed: x error", "reason": "regex check tool call"},
            {"kind": "error", "message": "rail 'x' failed: x error", "code": "not_a_hook_error_code"},
            {"kind": "block"},
        ],
        ids=[
            "allow-with-reason",
            "allow-with-code",
            "allow-with-message",
            "deny-without-reason",
            "deny-with-empty-reason",
            "deny-with-code",
            "deny-with-message",
            "error-without-message",
            "error-with-empty-message",
            "error-with-reason",
            "error-with-unknown-code",
            "unknown-kind",
        ],
    )
    def test_invalid_shape_raises_value_error(self, fields):
        """A field the kind doesn't have, a missing or empty reason or message, or an unknown code or kind is refused."""
        with pytest.raises(ValueError):
            HookOutcome(**fields)

    @pytest.mark.parametrize(
        "fields",
        [{"kind": "deny", "reason": 5}, {"kind": "error", "message": ["rail 'x' failed: x error"]}],
        ids=["deny-reason", "error-message"],
    )
    def test_reason_or_message_that_is_not_a_string_raises_type_error(self, fields):
        """A deny's reason or an error's message that isn't a string raises `TypeError`."""
        with pytest.raises(TypeError):
            HookOutcome(**fields)

    def test_code_given_as_its_string_value_becomes_the_enum_member(self):
        """A code given as its string value is stored as its `HookErrorCode` member."""
        outcome = HookOutcome(kind="error", message="rail 'x' failed: x error", code="rail_failed")

        assert outcome.code is HookErrorCode.RAIL_FAILED

    def test_outcome_cannot_be_changed_after_construction(self):
        """An outcome is immutable, so a valid one can't be turned into an invalid one."""
        outcome = HookOutcome.deny("regex check tool call")

        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.reason = ""


class TestOutcomeFromResult:
    def test_passed_is_allow(self):
        """A passed check is an allow."""
        result = RailsResult(status=RailStatus.PASSED, content="")

        assert outcome_from_result(result) == HookOutcome.allow()

    @pytest.mark.parametrize(
        "result, reason",
        [
            (
                _blocked(UNDECLARED_TOOL, rail="tool call validation", reason=UNDECLARED_TOOL.reason),
                "tool call 'WebFetch' is not an allowed tool",
            ),
            (
                _blocked(PER_TOOL_BLOCK, rail="regex check tool call", reason="regex check tool call"),
                "regex check tool call",
            ),
            (
                _blocked(MALFORMED_ARGUMENTS, reason=MALFORMED_ARGUMENTS.reason),
                "tool call 'toolu_01' has malformed arguments",
            ),
        ],
        ids=["tool-call-validation", "per-tool-rail", "malformed-arguments"],
    )
    def test_block_by_a_rail_is_deny_with_the_violation_reason(self, result, reason):
        """A rail's block is a deny carrying the violation's reason."""
        assert outcome_from_result(result) == HookOutcome.deny(reason)

    def test_every_violation_rail_failed_is_rail_failed_error(self):
        """A block where every violation is a rail failure is a `rail_failed` error naming the rail."""
        result = _blocked(RAIL_FAILED, rail="regex check tool call", reason=RAIL_FAILED.reason, content=INTERNAL_ERROR)

        assert outcome_from_result(result) == HookOutcome.error(
            "rail 'regex check tool call' failed: regex check tool call error", code=HookErrorCode.RAIL_FAILED
        )

    def test_rail_failure_without_a_rail_name_is_still_rail_failed_error(self):
        """A rail failure whose violation names no rail is still a `rail_failed` error, with the failure's reason."""
        unnamed = call_violation("rail_failed", "tool call validation error")

        assert outcome_from_result(_blocked(unnamed, content=INTERNAL_ERROR)) == HookOutcome.error(
            "a tool-call rail failed: tool call validation error", code=HookErrorCode.RAIL_FAILED
        )

    @pytest.mark.parametrize(
        "decided, reason",
        [
            (PER_TOOL_BLOCK, "regex check tool call"),
            (MALFORMED_ARGUMENTS, "tool call 'toolu_01' has malformed arguments"),
        ],
        ids=["per-tool-rail", "malformed-arguments"],
    )
    def test_rail_failure_beside_a_decided_violation_is_deny(self, decided, reason):
        """A rail failure beside a decided violation, a rail's block or an unparsable call, is a deny with its reason."""
        result = _blocked(RAIL_FAILED, decided, reason=RAIL_FAILED.reason, content=INTERNAL_ERROR)

        assert outcome_from_result(result) == HookOutcome.deny(reason)

    @pytest.mark.parametrize(
        "failure, message",
        [
            (EXTRACTION_FAILED, "the tool-call check failed: tool call extraction failed"),
            (INVALID_TOOLSET, "the tool-call check failed: tool parsing failed"),
            (
                INVALID_TOOL_SCHEMA,
                "the tool-call check failed: declared schema for tool 'Bash' is not valid JSON Schema",
            ),
        ],
        ids=["extraction-failed", "invalid-toolset", "invalid-tool-schema"],
    )
    def test_check_failure_other_than_a_rail_is_error_without_a_code(self, failure, message):
        """A block whose one violation says the check couldn't finish, though no rail failed, is an error with no code."""
        result = _blocked(failure, reason=failure.reason)

        assert outcome_from_result(result) == HookOutcome.error(message)

    def test_rail_failure_beside_another_check_failure_is_error_without_a_code(self):
        """A block whose violations are all check failures, not all of them a rail's, is an error with no code."""
        result = _blocked(RAIL_FAILED, EXTRACTION_FAILED, reason=RAIL_FAILED.reason, content=INTERNAL_ERROR)

        assert outcome_from_result(result) == HookOutcome.error(
            "the tool-call check failed: tool call extraction failed"
        )

    @pytest.mark.parametrize(
        "failure",
        [EXTRACTION_FAILED, INVALID_TOOLSET, INVALID_TOOL_SCHEMA],
        ids=["extraction-failed", "invalid-toolset", "invalid-tool-schema"],
    )
    def test_check_failure_beside_a_decided_violation_is_deny(self, failure):
        """A check failure beside a rail's block is a deny with the block's reason, since a policy denial wins."""
        result = _blocked(failure, PER_TOOL_BLOCK, reason=failure.reason)

        assert outcome_from_result(result) == HookOutcome.deny("regex check tool call")

    @pytest.mark.parametrize(
        "rail, reason, expected",
        [
            ("tool call validation", "the result's reason", "the result's reason"),
            ("tool call validation", None, "tool call validation"),
            (None, None, "tool call blocked"),
        ],
    )
    def test_block_without_violations_is_deny(self, rail, reason, expected):
        """A block reporting no violations is a deny, with the result's reason, else its rail, else a fixed reason."""
        assert outcome_from_result(_blocked(rail=rail, reason=reason)) == HookOutcome.deny(expected)

    @pytest.mark.parametrize(
        "result, expected",
        [
            (_blocked(EMPTY_REASON_BLOCK, PER_TOOL_BLOCK, reason="custom rail"), "regex check tool call"),
            (_blocked(EMPTY_REASON_BLOCK, rail="custom rail", reason=""), "custom rail"),
        ],
        ids=["next-violation", "rail-name"],
    )
    def test_empty_violation_reason_falls_through_to_the_next_reason(self, result, expected):
        """A violation with an empty reason is skipped, so a deny's reason is never empty."""
        assert outcome_from_result(result) == HookOutcome.deny(expected)

    def test_modified_is_error_without_a_code(self):
        """A modified result, which a tool-call check never gives, is an error with no code."""
        result = RailsResult(status=RailStatus.MODIFIED, content="rewritten")

        assert outcome_from_result(result) == HookOutcome.error("unexpected check status 'modified'")


class TestResponseBody:
    @pytest.mark.parametrize(
        "agent, outcome, expected",
        [
            ("claude-code", HookOutcome.allow(), {}),
            (
                "claude-code",
                HookOutcome.deny("Destructive command blocked by hook"),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "NeMo Guardrails: Destructive command blocked by hook",
                    }
                },
            ),
            (
                "claude-code",
                HookOutcome.deny("tool call 'WebFetch' is not an allowed tool"),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "NeMo Guardrails: tool call 'WebFetch' is not an allowed tool",
                    }
                },
            ),
            (
                "claude-code",
                HookOutcome.deny("tool safety check tool call"),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "NeMo Guardrails: tool safety check tool call",
                    }
                },
            ),
            ("codex", HookOutcome.allow(), {}),
            (
                "codex",
                HookOutcome.deny("Destructive command blocked by hook."),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "NeMo Guardrails: Destructive command blocked by hook.",
                    }
                },
            ),
            ("nemo-relay", HookOutcome.allow(), {"decision": "allow"}),
            (
                "nemo-relay",
                HookOutcome.deny("search.query is required"),
                {"decision": "deny", "reason": "search.query is required"},
            ),
        ],
        ids=[
            "claude-code-allow",
            "claude-code-deny",
            "claude-code-deny-not-allowed",
            "claude-code-deny-per-tool",
            "codex-allow",
            "codex-deny",
            "nemo-relay-allow",
            "nemo-relay-deny",
        ],
    )
    def test_response_equals_spec_example(self, agent, outcome, expected):
        """Each allow's and deny's response body is exactly the spec's response example for that path."""
        assert ADAPTERS[agent].response_body(outcome) == expected

    @pytest.mark.parametrize("agent", ["claude-code", "codex", "nemo-relay"])
    def test_error_outcome_has_no_200_body(self, agent):
        """An error outcome has no `200` body on any path."""
        with pytest.raises(ValueError, match="error outcome"):
            ADAPTERS[agent].response_body(HookOutcome.error("rail 'x' failed: x error", code=HookErrorCode.RAIL_FAILED))


class TestCodexSchemas:
    @pytest.mark.parametrize(
        "outcome", [HookOutcome.allow(), HookOutcome.deny("regex check tool call")], ids=["allow", "deny"]
    )
    @pytest.mark.parametrize("agent", ["claude-code", "codex"])
    def test_agent_responses_validate_against_codex_output_schema(self, agent, outcome):
        """Every Claude Code and Codex response validates against Codex's output schema, which rejects unknown fields."""
        jsonschema.validate(ADAPTERS[agent].response_body(outcome), CODEX_OUTPUT_SCHEMA)

    def test_codex_request_requires_only_fields_codex_always_sends(self):
        """Every field the Codex request model requires is one Codex's input schema requires, so Codex always sends it."""
        required = {name for name, field in CodexPreToolUseRequest.model_fields.items() if field.is_required()}

        assert required <= set(CODEX_INPUT_SCHEMA["required"])

    def test_codex_request_declares_only_codex_fields(self):
        """The Codex request model declares no field outside Codex's input schema."""
        assert set(CodexPreToolUseRequest.model_fields) <= set(CODEX_INPUT_SCHEMA["properties"])

    @pytest.mark.parametrize("body", CODEX_EXAMPLES)
    def test_spec_codex_examples_validate_against_codex_input_schema(self, body):
        """Every Codex request example in the spec is a body Codex's input schema accepts."""
        jsonschema.validate(body, CODEX_INPUT_SCHEMA)
