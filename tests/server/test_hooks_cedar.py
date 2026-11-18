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

"""The hooks server's Cedar rules: an agent start and a tool call decided by Cedar policies alone."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails import telemetry
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks.app import HooksConfigError, create_hooks_app
from nemoguardrails.server.hooks.cedar_policy import cedar_record

EXAMPLES = Path(__file__).parents[2] / "examples"
EXAMPLE_CONFIG = EXAMPLES / "configs" / "agent_hooks" / "cedar"
SCHEMA = EXAMPLE_CONFIG / "hooks.cedarschema"
YAML_START_CONFIG = EXAMPLES / "configs" / "agent_hooks" / "agent_start"
YAML_STOP_CONFIG = EXAMPLES / "configs" / "agent_hooks" / "stop"
APPROVED = "git::https://git.example.com/workshop/agent-parts.git//"
MODEL = "bedrock::nvidia.nemotron-super-3-120b"
DIGEST = "sha256:9afb37ee2535c419d84e11883c8c5e687ef2d729bbfe0dde07e010d3d4814aa8"
PART = {
    "id": "$/skills/release-checklist",
    "type": "skill",
    "source": f"{APPROVED}skills/release-checklist",
    "version": "0.1.0",
    "digest": DIGEST,
    "local": False,
}
START = {
    "hook_event_name": "AgentStart",
    "agent_name": "release-reviewer",
    "agent_digest": "sha256:abc",
    "parts": [PART],
    "models": [MODEL],
    "policies": [{"id": "net", "enforcers": ["openshell"]}],
    "enforcers_present": ["openshell"],
}
RELAY = "/v1/hooks/nemo-relay"
PERMIT_ALL_STARTS = '@id("start/permit")\npermit(principal, action == Hooks::Action::"start", resource);\n'


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)


def _relay(tool_name: str, tool_input: Any) -> dict:
    return {
        "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
        "tool_name": tool_name,
        "tool_input": tool_input,
    }


def _write_policies(folder: Path, **files: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / f"{name}.cedar").write_text(text, encoding="utf-8")
    return folder


def _app(config: dict, folder: Path):
    """The app for *config*, loaded from *folder* so its relative Cedar paths resolve there."""
    (folder / "config.yml").write_text(json.dumps(config), encoding="utf-8")
    return create_hooks_app(RailsConfig.from_path(str(folder)))


@pytest.fixture
def example():
    pytest.importorskip("cedarpy")
    with TestClient(create_hooks_app(RailsConfig.from_path(str(EXAMPLE_CONFIG)))) as client:
        yield client


class TestAgentStart:
    """The example's agent-start policies: the Cedar form of the YAML agent_start example."""

    def test_an_approved_start_is_allowed(self, example):
        assert example.post("/v1/hooks/agent-start", json=START).json() == {"decision": "allow"}

    @pytest.mark.parametrize(
        ("field", "value", "reason"),
        [
            (
                "parts",
                [{**PART, "source": "git::https://git.example.org/community/x.git//skills/x"}],
                "skill '$/skills/release-checklist' is denied by Cedar policy 'parts/approved-source'",
            ),
            (
                "parts",
                [{**PART, "source": "./skills/release-checklist", "local": True}],
                "skill '$/skills/release-checklist' is denied by Cedar policy 'parts/no-local-folder'",
            ),
            (
                "parts",
                [{key: value for key, value in PART.items() if key != "digest"}],
                "skill '$/skills/release-checklist' is denied by Cedar policy 'parts/digest'",
            ),
            ("models", ["bedrock::other"], "model 'bedrock::other' is denied by Cedar policy 'models/listed'"),
            ("enforcers_present", [], "policy 'net' is denied by Cedar policy 'policies/enforcers-present'"),
            (
                "policies",
                [{"id": "net", "enforcers": []}],
                "policy 'net' is denied by Cedar policy 'policies/enforcers-present'",
            ),
        ],
    )
    def test_each_rule_denies_with_its_policy(self, example, field, value, reason):
        answer = example.post("/v1/hooks/agent-start", json={**START, field: value}).json()
        assert answer == {"decision": "deny", "reason": reason}

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("parts", [{**PART, "source": "git::https://git.example.org/community/x.git//skills/x"}]),
            ("parts", [{**PART, "source": "./skills/release-checklist", "local": True}]),
            ("parts", [{key: value for key, value in PART.items() if key != "digest"}]),
            ("models", ["bedrock::other"]),
            ("enforcers_present", []),
            ("policies", [{"id": "net", "enforcers": []}]),
            ("parts", [PART]),
        ],
    )
    def test_cedar_answers_as_the_yaml_example_does(self, example, field, value):
        with TestClient(create_hooks_app(RailsConfig.from_path(str(YAML_START_CONFIG)))) as yaml_server:
            yaml_answer = yaml_server.post("/v1/hooks/agent-start", json={**START, field: value}).json()
        cedar_answer = example.post("/v1/hooks/agent-start", json={**START, field: value}).json()
        assert cedar_answer["decision"] == yaml_answer["decision"]

    def test_nothing_starts_unless_a_policy_permits_it(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(tmp_path / "policies", only="forbid(principal, action, resource is Hooks::Model);")
        app = _app({"models": [], "hooks_server": {"agent_start": {"cedar": {"policies": "policies"}}}}, tmp_path)
        with TestClient(app) as client:
            answer = client.post("/v1/hooks/agent-start", json=START).json()
        assert answer == {"decision": "deny", "reason": "no Cedar policy permits agent 'release-reviewer'"}

    def test_a_policy_that_fails_to_evaluate_denies(self, tmp_path):
        pytest.importorskip("cedarpy")
        # `version` is optional, and the policy reads it without `has`: Cedar skips such a forbid, the server denies.
        _write_policies(
            tmp_path / "policies",
            start=PERMIT_ALL_STARTS
            + '@id("parts/version")\nforbid(principal, action, resource is Hooks::Part) when { resource.version == "0" };',
        )
        app = _app({"models": [], "hooks_server": {"agent_start": {"cedar": {"policies": "policies"}}}}, tmp_path)
        part = {key: value for key, value in PART.items() if key != "version"}
        with TestClient(app) as client:
            answer = client.post("/v1/hooks/agent-start", json={**START, "parts": [part]}).json()
        assert answer == {
            "decision": "deny",
            "reason": "skill '$/skills/release-checklist' is denied: Cedar policy 'parts/version' failed to evaluate",
        }

    def test_a_part_is_in_its_agent(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(
            tmp_path / "policies",
            start=PERMIT_ALL_STARTS
            + '@id("own")\nforbid(principal, action, resource is Hooks::Part) unless { resource in principal };',
        )
        app = _app({"models": [], "hooks_server": {"agent_start": {"cedar": {"policies": "policies"}}}}, tmp_path)
        with TestClient(app) as client:
            assert client.post("/v1/hooks/agent-start", json=START).json() == {"decision": "allow"}


class TestToolCall:
    """The example's tool-call policies: the Cedar form of the YAML stop example."""

    @pytest.mark.parametrize(
        ("tool_name", "tool_input", "answer"),
        [
            ("search", {"query": "gpus"}, {"decision": "allow"}),
            (
                "search",
                {"query": ""},
                {"decision": "deny", "reason": "tool call 'search' is denied by Cedar policy 'search/empty-query'"},
            ),
            (
                "search",
                {"query": "gpus", "limit": 3},
                {"decision": "deny", "reason": "no Cedar policy permits tool call 'search'"},
            ),
            ("fetch", {"url": "x"}, {"decision": "deny", "reason": "no Cedar policy permits tool call 'fetch'"}),
            (
                "delete_index",
                {},
                {
                    "decision": "stop",
                    "reason": "tool call 'delete_index' is stopped by Cedar policy 'stop/delete-index'",
                },
            ),
            (
                "search",
                {"query": "gpus", "admin_override": True},
                {"decision": "stop", "reason": "tool call 'search' is stopped by Cedar policy 'stop/admin-override'"},
            ),
        ],
    )
    def test_the_example_decides_each_call(self, example, tool_name, tool_input, answer):
        assert example.post(RELAY, json=_relay(tool_name, tool_input)).json() == answer

    @pytest.mark.parametrize(
        ("tool_name", "tool_input"),
        [
            ("search", {"query": "gpus"}),
            ("fetch", {"url": "x"}),
            ("delete_index", {}),
            ("search", {"query": "gpus", "admin_override": True}),
            ("search", {"query": "gpus", "limit": 3}),
        ],
    )
    def test_cedar_answers_as_the_yaml_stop_example_does(self, example, tool_name, tool_input):
        with TestClient(create_hooks_app(RailsConfig.from_path(str(YAML_STOP_CONFIG)))) as yaml_server:
            yaml_answer = yaml_server.post(RELAY, json=_relay(tool_name, tool_input)).json()
        assert example.post(RELAY, json=_relay(tool_name, tool_input)).json()["decision"] == yaml_answer["decision"]

    def test_claude_code_gets_a_stop_as_a_deny_that_ends_the_turn(self, example):
        body = {"hook_event_name": "PreToolUse", "tool_name": "delete_index", "tool_input": {}, "tool_use_id": "t1"}
        answer = example.post("/v1/hooks/claude-code", json=body).json()
        assert answer["continue"] is False
        assert answer["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_a_quarantine_is_stored_and_holds_the_agent(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(
            tmp_path / "tools",
            tools="permit(principal, action, resource);\n"
            '@id("exfiltrate")\n@outcome("quarantine")\n'
            'forbid(principal, action, resource == Hooks::Tool::"exfiltrate");',
        )
        _write_policies(tmp_path / "start", start=PERMIT_ALL_STARTS)
        config = {
            "models": [],
            "hooks_server": {
                "cedar": {"policies": "tools"},
                "quarantine_store": str(tmp_path / "quarantine.jsonl"),
                "agent_start": {"cedar": {"policies": "start"}},
            },
        }
        headers = {"X-NeMo-Agent-Name": "release-reviewer", "X-NeMo-Agent-Digest": "sha256:abc"}
        with TestClient(_app(config, tmp_path)) as client:
            answer = client.post(RELAY, json=_relay("exfiltrate", {}), headers=headers).json()
            assert answer["decision"] == "quarantine"
            held = client.post("/v1/hooks/agent-start", json=START).json()
        assert held["decision"] == "deny"
        assert "is quarantined" in held["reason"]

    def test_the_agent_and_its_digest_are_the_principal(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(
            tmp_path / "policies",
            tools='permit(principal, action, resource) when { principal.name == "release-reviewer" && '
            'principal has digest && principal.digest == "sha256:abc" };',
        )
        app = _app({"models": [], "hooks_server": {"cedar": {"policies": "policies"}}}, tmp_path)
        with TestClient(app) as client:
            named = client.post(
                RELAY,
                json=_relay("x", {}),
                headers={"X-NeMo-Agent-Name": "release-reviewer", "X-NeMo-Agent-Digest": "sha256:abc"},
            ).json()
            unnamed = client.post(RELAY, json=_relay("x", {})).json()
        assert named == {"decision": "allow"}
        assert unnamed["decision"] == "deny"

    def test_arguments_that_are_not_an_object_are_denied_before_cedar(self, example):
        answer = example.post(RELAY, json=_relay("search", "gpus")).json()
        assert answer == {"decision": "deny", "reason": "tool call 'search' has arguments that aren't a JSON object"}


class TestArguments:
    def test_cedar_holds_strings_whole_numbers_booleans_lists_and_objects(self):
        values = {"s": "a", "n": 3, "b": True, "l": ["a", 1], "o": {"k": "v"}}
        assert cedar_record(values) == values

    def test_what_cedar_cannot_hold_is_left_out_and_still_named(self):
        assert cedar_record({"f": 1.5, "z": None, "big": 2**70, "l": [1.5], "ok": 1}) == {"ok": 1}

    @pytest.mark.parametrize("key", ["__entity", "__extn", "__expr"])
    def test_a_key_cedar_json_reads_as_an_entity_is_dropped(self, key):
        assert cedar_record({key: {"type": "Hooks::Agent", "id": "admin"}, "o": {key: "x"}}) == {"o": {}}


class TestConfig:
    def test_yaml_rules_beside_cedar_are_refused(self):
        with pytest.raises(ValidationError, match="decided by Cedar alone"):
            RailsConfig.from_content(
                config={"models": [], "hooks_server": {"cedar": {"policies": "p"}, "blocked_tools": ["x"]}}
            )
        with pytest.raises(ValidationError, match="decided by Cedar alone"):
            RailsConfig.from_content(
                config={
                    "models": [],
                    "hooks_server": {"agent_start": {"cedar": {"policies": "p"}, "require_digests": True}},
                }
            )

    def test_a_policy_set_that_does_not_parse_stops_the_server(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(tmp_path / "policies", broken="permit(principal, action")
        with pytest.raises(HooksConfigError, match="'broken.cedar' doesn't parse as Cedar"):
            _app({"models": [], "hooks_server": {"cedar": {"policies": "policies"}}}, tmp_path)

    def test_a_missing_folder_or_an_empty_one_stops_the_server(self, tmp_path):
        pytest.importorskip("cedarpy")
        with pytest.raises(HooksConfigError, match="which is not a folder"):
            _app({"models": [], "hooks_server": {"cedar": {"policies": "nowhere"}}}, tmp_path)
        (tmp_path / "empty").mkdir()
        with pytest.raises(HooksConfigError, match="has no .cedar file"):
            _app({"models": [], "hooks_server": {"cedar": {"policies": "empty"}}}, tmp_path)

    def test_policies_that_fail_the_schema_stop_the_server(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(
            tmp_path / "policies", typo='permit(principal, action, resource) when { resource.nmae == "x" };'
        )
        (tmp_path / "hooks.cedarschema").write_text(SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
        config = {"models": [], "hooks_server": {"cedar": {"policies": "policies", "schema": "hooks.cedarschema"}}}
        with pytest.raises(HooksConfigError, match="fail the check against the schema"):
            _app(config, tmp_path)

    @pytest.mark.parametrize(
        ("policy", "match"),
        [
            ('@outcome("stop")\npermit(principal, action, resource);', "only a forbid selects one"),
            ('@outcome("halt")\nforbid(principal, action, resource);', 'use "stop" or "quarantine"'),
        ],
    )
    def test_a_wrong_outcome_annotation_stops_the_server(self, tmp_path, policy, match):
        pytest.importorskip("cedarpy")
        _write_policies(tmp_path / "policies", bad=policy)
        with pytest.raises(HooksConfigError, match=match):
            _app({"models": [], "hooks_server": {"cedar": {"policies": "policies"}}}, tmp_path)

    def test_an_agent_start_policy_cannot_select_an_outcome(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(tmp_path / "policies", bad='@outcome("stop")\nforbid(principal, action, resource);')
        with pytest.raises(HooksConfigError, match="only a tool call's policy can have"):
            _app({"models": [], "hooks_server": {"agent_start": {"cedar": {"policies": "policies"}}}}, tmp_path)

    def test_a_quarantine_outcome_needs_a_store(self, tmp_path):
        pytest.importorskip("cedarpy")
        _write_policies(tmp_path / "policies", q='@outcome("quarantine")\nforbid(principal, action, resource);')
        with pytest.raises(HooksConfigError, match="no hooks_server.quarantine_store"):
            _app({"models": [], "hooks_server": {"cedar": {"policies": "policies"}}}, tmp_path)

    def test_without_cedarpy_a_cedar_config_fails_at_start_with_the_extra_to_install(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "cedarpy", None)
        _write_policies(tmp_path / "policies", any="permit(principal, action, resource);")
        with pytest.raises(HooksConfigError, match=r"pip install nemoguardrails\[cedar\]"):
            _app({"models": [], "hooks_server": {"cedar": {"policies": "policies"}}}, tmp_path)

    def test_a_yaml_config_does_not_need_cedarpy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cedarpy", None)
        with TestClient(create_hooks_app(RailsConfig.from_path(str(YAML_START_CONFIG)))) as client:
            body = {**START, "policies": [], "enforcers_present": []}
            assert client.post("/v1/hooks/agent-start", json=body).json() == {"decision": "allow"}
