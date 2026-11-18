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

"""A stop: a rule that ends the run, not only the call, through each agent's hook path."""

import logging
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails import telemetry
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks import app as hooks_app
from nemoguardrails.server.hooks.app import HooksConfigError, create_hooks_app
from nemoguardrails.server.hooks.checking import HookOutcome
from tests.server.test_hooks_models import CODEX_OUTPUT_SCHEMA
from tests.server.test_hooks_server import _hook

AGENTS = ["claude-code", "codex", "nemo-relay"]
EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "configs" / "agent_hooks" / "stop"

# `drop_database` ends the run; Bash ends it when a call passes `dangerously_disable_sandbox`, and when the regex rail
# finds `rm -rf`; WebFetch is only denied.
STOP_CONFIG = {
    "models": [],
    "hooks_server": {
        "blocked_tools": ["WebFetch"],
        "stop_tools": ["drop_database", {"name": "Bash", "arguments": ["dangerously_disable_sandbox"]}],
        "stop_rails": ["regex check tool call"],
    },
    "rails": {
        "config": {"regex_detection": {"tool_call": {"Bash": {"patterns": [r"rm\s+-rf"]}}}},
        "tool_call": {"per_tool": {"Bash": ["regex check tool call $argument=command"]}},
    },
}


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)


def _answer(agent: str, tool_name: str, tool_input: dict, config: dict = STOP_CONFIG) -> dict:
    with TestClient(create_hooks_app(RailsConfig.from_content(config=config))) as client:
        response = client.post(f"/v1/hooks/{agent}", json=_hook(agent, tool_name, tool_input))
    assert response.status_code == 200, response.text
    return response.json()


def _stopped(agent: str, reason: str) -> dict:
    """*agent*'s stop: a deny of the call, and for Claude Code and Relay the stop itself."""
    if agent == "nemo-relay":
        return {"decision": "stop", "reason": reason}
    denied = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"NeMo Guardrails: {reason}",
        }
    }
    if agent == "codex":
        return denied
    return {"continue": False, "stopReason": f"NeMo Guardrails: {reason}", **denied}


class TestStopTools:
    @pytest.mark.parametrize("agent", AGENTS)
    def test_a_stop_tool_by_name_stops_the_run(self, agent):
        assert _answer(agent, "drop_database", {"name": "prod"}) == _stopped(
            agent, "tool call 'drop_database' is a stop tool"
        )

    @pytest.mark.parametrize("agent", AGENTS)
    def test_a_stop_argument_stops_the_run(self, agent):
        answer = _answer(agent, "Bash", {"command": "ls", "dangerously_disable_sandbox": True})
        assert answer == _stopped(
            agent, "tool call 'Bash' passes argument 'dangerously_disable_sandbox', which stops the run"
        )

    def test_a_call_without_the_stop_argument_is_decided_as_before(self):
        assert _answer("nemo-relay", "Bash", {"command": "ls"}) == {"decision": "allow"}

    def test_a_blocked_tool_is_only_denied(self):
        assert _answer("nemo-relay", "WebFetch", {"url": "x"}) == {
            "decision": "deny",
            "reason": "tool call 'WebFetch' is a blocked tool",
        }

    def test_a_stop_wins_over_an_allowlist(self):
        config = {
            "models": [],
            "hooks_server": {
                "allowed_tools": ["drop_database"],
                "allow_unlisted_tools": False,
                "stop_tools": ["drop_database"],
            },
        }
        assert _answer("nemo-relay", "drop_database", {}, config)["decision"] == "stop"

    def test_stop_tools_alone_is_a_policy_that_can_deny(self):
        config = {"models": [], "hooks_server": {"stop_tools": ["drop_database"]}}
        assert _answer("nemo-relay", "ls", {}, config) == {"decision": "allow"}
        assert _answer("nemo-relay", "drop_database", {}, config)["decision"] == "stop"

    def test_a_stop_tool_named_twice_is_refused(self):
        with pytest.raises(ValidationError, match="stop_tools names tool 'x' more than once"):
            RailsConfig.from_content(config={"models": [], "hooks_server": {"stop_tools": ["x", "x"]}})


class TestStopRails:
    @pytest.mark.parametrize("agent", AGENTS)
    def test_a_stop_rail_that_blocks_stops_the_run(self, agent):
        assert _answer(agent, "Bash", {"command": "rm -rf /"}) == _stopped(agent, "regex check tool call")

    def test_a_rail_not_listed_only_denies(self):
        config = {**STOP_CONFIG, "hooks_server": {**STOP_CONFIG["hooks_server"], "stop_rails": []}}
        assert _answer("nemo-relay", "Bash", {"command": "rm -rf /"}, config) == {
            "decision": "deny",
            "reason": "regex check tool call",
        }

    def test_a_stop_rail_that_no_flow_runs_is_refused(self):
        config = {**STOP_CONFIG, "hooks_server": {**STOP_CONFIG["hooks_server"], "stop_rails": ["regex check"]}}
        with pytest.raises(HooksConfigError, match="stop_rails names rail 'regex check'"):
            create_hooks_app(RailsConfig.from_content(config=config))


class TestResponses:
    def test_codex_gets_a_deny_its_schema_accepts(self):
        jsonschema.validate(_answer("codex", "drop_database", {}), CODEX_OUTPUT_SCHEMA)

    def test_a_stop_needs_a_reason(self):
        with pytest.raises(ValueError, match="a stop outcome needs a reason"):
            HookOutcome(kind="stop")

    def test_a_stop_is_logged_as_a_stop(self, caplog):
        caplog.set_level(logging.INFO, logger=hooks_app.__name__)
        _answer("nemo-relay", "drop_database", {})
        lines = [record.getMessage() for record in caplog.records if record.name == hooks_app.__name__]
        assert any("outcome=stop" in line and "rail='tool policy'" in line for line in lines)


class TestExampleConfig:
    def test_the_example_stops_and_denies(self):
        config = RailsConfig.from_path(str(EXAMPLE_CONFIG))
        with TestClient(create_hooks_app(config)) as client:
            ask = lambda name, tool_input: client.post(  # noqa: E731
                "/v1/hooks/nemo-relay",
                json={
                    "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
                    "tool_name": name,
                    "tool_input": tool_input,
                },
            ).json()
            assert ask("search", {"query": "weather"}) == {"decision": "allow"}
            assert ask("search", {"query": ""})["decision"] == "deny"
            assert ask("delete_index", {"index": "docs"})["decision"] == "stop"
            assert ask("search", {"query": "x", "admin_override": True})["decision"] == "stop"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_exported_contract_describes_a_stop(agent):
    from tests.server.test_export_hooks_openapi import validate

    exporter = __import__("runpy").run_path(str(Path(__file__).parents[2] / "scripts" / "export_hooks_openapi.py"))
    spec = exporter["hooks_openapi"]()
    schema = spec["paths"][f"/v1/hooks/{agent}"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    validate(spec, schema, _answer(agent, "drop_database", {}))
