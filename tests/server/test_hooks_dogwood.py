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

"""The hooks server's Dogwood rules: a tool call decided from the call and from what already happened in the run."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails import telemetry
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks import app as hooks_app
from nemoguardrails.server.hooks.app import HooksConfigError, create_hooks_app

pytest.importorskip("cedarpy")
pytest.importorskip("nemoguardrails_dogwood")

EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "configs" / "agent_hooks" / "dogwood"
GATEWAY = "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/gw-1"
SCOPE = 'resource == AgentCore::Gateway::"{gateway}"'
PERMIT_ALL = '@id("permit-all")\npermit(principal, action, resource);\n'
CLAUDE = "/v1/hooks/claude-code"
# The tools these tests call, in AgentCore's vocabulary, with the inputs a policy may read.
SCHEMA = """namespace AgentCore {
  entity OAuthUser;
  entity Gateway;
  action "Bash" appliesTo { principal: [OAuthUser], resource: [Gateway], context: { input: { command: String } } };
  action "ops___get_status", "ops___restart_service", "ops___deploy", "c___approve", "c___disburse",
    "claims___delete_claim", "claims___disburse_payment"
    appliesTo { principal: [OAuthUser], resource: [Gateway], context: { input: {} } };
  action "pay___disburse" appliesTo {
    principal: [OAuthUser], resource: [Gateway], context: { input: { amountCents: Long } }
  };
}
"""


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)


def _policy(effect: str, action: str, temporal: str, *, policy_id: str, gateway: str = GATEWAY) -> str:
    scope = SCOPE.format(gateway=gateway)
    return (
        f'@id("{policy_id}")\n{effect} (\n    principal,\n    action == AgentCore::Action::"{action}",\n    {scope}\n)\n'
        f"when temporal {{\n    {temporal}\n}};\n"
    )


def _count_cap(action: str, window: str, limit: int) -> str:
    return (
        "exists (n: Long).\n        (count for (t: Timepoint).\n"
        f'            where (formerly within {window} (AgentCore::Action::"{action}"::request{{ eventResource: resource }}'
        f" && tp(t)))) == n\n        && n > {limit}"
    )


def _server(
    tmp_path: Path, policies: dict[str, str], gateway: str = GATEWAY, denied_calls: str = "request", **hooks_server
) -> TestClient:
    folder = tmp_path / "policies"
    folder.mkdir(exist_ok=True)
    for name, text in policies.items():
        (folder / name).write_text(text, encoding="utf-8")
    (tmp_path / "hooks.cedarschema").write_text(SCHEMA, encoding="utf-8")
    dogwood = {"policies": str(folder), "schema": str(tmp_path / "hooks.cedarschema"), "gateway": gateway}
    config = {
        "models": [],
        "hooks_server": {
            "decision_record": str(tmp_path / "decisions.jsonl"),
            "dogwood": {**dogwood, "denied_calls": denied_calls},
            **hooks_server,
        },
    }
    return TestClient(create_hooks_app(RailsConfig.from_content(config=config)))


def _call(
    client: TestClient, tool: str, run: Optional[str] = "run-1", agent: str = "release-reviewer", **tool_input: Any
) -> tuple[bool, Optional[str]]:
    """Ask about one tool call the way the lab's `ask-tool.py` does; whether it is allowed, and the deny reason."""
    headers = {"x-nemo-agent-name": agent, "x-nemo-agent-digest": "sha256:abc"}
    if run is not None:
        headers["x-nemo-run-id"] = run
    body = {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input, "tool_use_id": "t"}
    response = client.post(CLAUDE, json=body, headers=headers)
    assert response.status_code == 200, response.text
    verdict = response.json().get("hookSpecificOutput", {})
    return verdict.get("permissionDecision") != "deny", verdict.get("permissionDecisionReason")


def _completed(
    client: TestClient, tool: str, event: str = "PostToolUse", run: str = "run-1", **tool_input: Any
) -> None:
    headers = {"x-nemo-agent-name": "release-reviewer", "x-nemo-agent-digest": "sha256:abc", "x-nemo-run-id": run}
    body = {
        "hook_event_name": event,
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_use_id": "t",
        "tool_response": "secret",
    }
    response = client.post(CLAUDE, json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {}


def _later(monkeypatch, minutes: int) -> None:
    """Move the server's clock forward, as if *minutes* had passed since the record's entries were written."""
    real = datetime

    class Later(real):
        @classmethod
        def now(cls, tz=None):
            return real.now(tz) + timedelta(minutes=minutes)

    monkeypatch.setattr(hooks_app, "datetime", Later)


class TestCountCap:
    POLICIES = {
        "permit.dw": PERMIT_ALL,
        "cap.dw": _policy("forbid", "Bash", _count_cap("Bash", "5m", 2), policy_id="bash/at-most-2-in-5m"),
    }

    def test_the_calls_up_to_the_cap_are_allowed_and_the_next_is_denied(self, tmp_path):
        with _server(tmp_path, self.POLICIES) as client:
            assert _call(client, "Bash", command="ls")[0]
            assert _call(client, "Bash", command="ls")[0]
            allowed, reason = _call(client, "Bash", command="ls")
        assert not allowed
        assert reason == "NeMo Guardrails: tool call 'Bash' is denied by Dogwood policy 'bash/at-most-2-in-5m'"

    def test_when_the_window_has_passed_the_call_is_allowed_again(self, tmp_path, monkeypatch):
        with _server(tmp_path, self.POLICIES) as client:
            for _ in range(2):
                assert _call(client, "Bash", command="ls")[0]
            assert not _call(client, "Bash", command="ls")[0]
            _later(monkeypatch, 6)
            assert _call(client, "Bash", command="ls")[0]

    def test_another_run_or_another_agent_starts_its_own_count(self, tmp_path):
        with _server(tmp_path, self.POLICIES) as client:
            for _ in range(2):
                assert _call(client, "Bash", command="ls")[0]
            assert _call(client, "Bash", run="run-2", command="ls")[0]
            assert _call(client, "Bash", agent="other-agent", command="ls")[0]


def test_an_earlier_action_in_the_same_run_denies_a_later_one(tmp_path):
    mutual = _policy(
        "forbid",
        "claims___delete_claim",
        'formerly within 2m AgentCore::Action::"claims___disburse_payment"::request{ eventResource: resource }',
        policy_id="no-delete-after-disburse",
    )
    with _server(tmp_path, {"permit.dw": PERMIT_ALL, "mutual.dw": mutual}) as client:
        assert _call(client, "mcp__claims__delete_claim", run="run-a")[0]
        assert _call(client, "mcp__claims__disburse_payment", run="run-b")[0]
        allowed, reason = _call(client, "mcp__claims__delete_claim", run="run-b")
    assert not allowed
    assert "no-delete-after-disburse" in reason


class TestHistoryUnavailable:
    POLICIES = {
        "permit.dw": PERMIT_ALL,
        "cap.dw": _policy("forbid", "Bash", _count_cap("Bash", "5m", 2), policy_id="c"),
    }

    def test_a_call_without_a_run_id_is_denied(self, tmp_path):
        with _server(tmp_path, self.POLICIES) as client:
            allowed, reason = _call(client, "Bash", run=None, command="ls")
        assert not allowed
        assert "x-nemo-run-id" in reason and "Dogwood rules need it" in reason

    def test_a_record_that_cannot_be_read_denies(self, tmp_path):
        with _server(tmp_path, self.POLICIES) as client:
            assert _call(client, "Bash", command="ls")[0]
            record = tmp_path / "decisions.jsonl"
            # An earlier line that isn't JSON: the history can't be read, while a new entry can still be written.
            record.write_text("not json\n" + record.read_text(), encoding="utf-8")
            allowed, reason = _call(client, "Bash", command="ls")
        assert not allowed
        assert "can't be read" in reason

    def test_a_config_without_a_decision_record_is_refused(self, tmp_path):
        with pytest.raises(ValidationError, match="no decision_record"):
            RailsConfig.from_content(
                config={"models": [], "hooks_server": {"dogwood": {"policies": str(tmp_path), "schema": "s"}}}
            )


def test_an_agentform_prerequisite_is_met_by_a_completed_call_and_not_by_a_failed_one(tmp_path):
    """The exact text Agentform's Dogwood renderer writes for a `requires` with no gateway ARN given."""
    rendered = (
        "permit (\n    principal,\n"
        '    action == AgentCore::Action::"ops___restart_service",\n'
        '    resource == AgentCore::Gateway::"__GATEWAY_ARN__"\n)\nwhen temporal {\n'
        '    formerly within 15m AgentCore::Action::"ops___get_status"::response{ eventResource: resource }\n};\n'
    )
    status = '@id("allow-status")\npermit (principal, action == AgentCore::Action::"ops___get_status", resource);\n'
    with _server(tmp_path, {"policies.dogwood": rendered, "status.dw": status}, gateway="__GATEWAY_ARN__") as client:
        assert not _call(client, "mcp__ops__restart_service")[0]
        assert _call(client, "mcp__ops__get_status")[0]
        _completed(client, "mcp__ops__get_status", event="PostToolUseFailure")
        assert not _call(client, "mcp__ops__restart_service")[0]
        _completed(client, "mcp__ops__get_status")
        assert _call(client, "mcp__ops__restart_service")[0]


@pytest.mark.parametrize(
    ("denied_calls", "third_allowed"),
    # By default a denied call is an attempt and its amount still counts, as the Dogwood language keeps it; read as
    # AgentCore describes a denied request, it is an error event and its amount does not.
    [("request", False), ("error", True)],
)
def test_a_running_total_is_capped_and_the_record_keeps_only_the_summed_input(tmp_path, denied_calls, third_allowed):
    total = (
        "exists (total: Long).\n        (sum amt for (amt: Long), (t: Timepoint).\n"
        '            where (formerly within 5m (AgentCore::Action::"pay___disburse"::request{ eventResource: resource, '
        "input.amountCents: amt } && tp(t)))) == total\n        && total > 150"
    )
    policies = {"permit.dw": PERMIT_ALL, "sum.dw": _policy("forbid", "pay___disburse", total, policy_id="s")}
    with _server(tmp_path, policies, denied_calls=denied_calls) as client:
        assert _call(client, "mcp__pay__disburse", amountCents=100, note="first")[0]
        assert not _call(client, "mcp__pay__disburse", amountCents=100, note="second")[0]
        assert _call(client, "mcp__pay__disburse", amountCents=50, note="third")[0] == third_allowed
    entries = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [entry["asked"]["inputs"] for entry in entries] == [{"amountCents": 100}] * 2 + [{"amountCents": 50}]
    assert "first" not in (tmp_path / "decisions.jsonl").read_text()


def test_an_approval_is_good_for_one_use(tmp_path):
    once = _policy(
        "permit",
        "c___disburse",
        '!AgentCore::Action::"c___disburse"::response{ eventResource: resource }\n'
        '    since within 1h AgentCore::Action::"c___approve"::response{ eventResource: resource }',
        policy_id="one-use",
    )
    approve = '@id("approve")\npermit (principal, action == AgentCore::Action::"c___approve", resource);\n'
    with _server(tmp_path, {"once.dw": once, "approve.dw": approve}) as client:
        assert not _call(client, "mcp__c__disburse")[0]
        _completed(client, "mcp__c__approve")
        assert _call(client, "mcp__c__disburse")[0]
        _completed(client, "mcp__c__disburse")
        assert not _call(client, "mcp__c__disburse")[0]


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("permit (principal, action, resource) when temporal { formerly within };", "Dogwood refuses the policies"),
        (
            'permit (principal, action == AgentCore::Action::"Bash", resource) when { context.input.nope == "x" };',
            "fail Dogwood's type check",
        ),
    ],
)
def test_a_policy_dogwood_refuses_stops_the_server(tmp_path, policy, message):
    with pytest.raises(HooksConfigError, match=message):
        _server(tmp_path, {"p.dw": policy})


def test_a_call_to_a_tool_the_schema_does_not_declare_is_denied(tmp_path):
    with _server(tmp_path, {"permit.dw": PERMIT_ALL}) as client:
        allowed, reason = _call(client, "WebFetch", url="https://example.com")
    assert not allowed
    assert "declares no action 'WebFetch'" in reason


def test_arguments_the_schema_does_not_declare_are_not_seen_or_recorded(tmp_path):
    with _server(tmp_path, {"permit.dw": PERMIT_ALL}) as client:
        assert _call(client, "Bash", command="ls", description="list the folder")[0]
    entry = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert entry["asked"]["inputs"] == {"command": "ls"}


def test_dogwood_beside_another_tool_rule_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="decided by Dogwood alone"):
        RailsConfig.from_content(
            config={
                "models": [],
                "hooks_server": {
                    "decision_record": "d.jsonl",
                    "dogwood": {"policies": str(tmp_path), "schema": "s.cedarschema"},
                    "blocked_tools": ["Bash"],
                },
            }
        )


def test_the_example_configuration_caps_and_requires(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(create_hooks_app(RailsConfig.from_path(str(EXAMPLE_CONFIG)))) as client:
        assert not _call(client, "mcp__ops__restart_service")[0]
        assert _call(client, "mcp__ops__get_status")[0]
        _completed(client, "mcp__ops__get_status")
        assert _call(client, "mcp__ops__restart_service")[0]
        assert [_call(client, "mcp__ops__deploy")[0] for _ in range(3)] == [True, True, False]
