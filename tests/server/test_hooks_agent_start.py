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

"""`/v1/hooks/agent-start` through the hooks server's ASGI app: whether an agent may start, decided by fixed rules."""

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails import telemetry
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks import app as hooks_app
from nemoguardrails.server.hooks.app import HooksConfigError, create_hooks_app

APPROVED = "git::https://git.example.com/workshop/agent-parts.git//"
OTHER = "git::https://git.example.org/community/agent-parts.git//skills/release-checklist"
MODEL = "bedrock::nvidia.nemotron-super-3-120b"
OTHER_MODEL = "bedrock::meta.llama3-70b-instruct-v1:0"
DIGEST = "sha256:9afb37ee2535c419d84e11883c8c5e687ef2d729bbfe0dde07e010d3d4814aa8"

APPROVED_PART = {
    "id": "$/skills/release-checklist",
    "type": "skill",
    "source": f"{APPROVED}skills/release-checklist",
    "version": "0.1.0",
    "digest": DIGEST,
    "local": False,
}
CALLED_AGENT = {
    "id": "$/agents/researcher",
    "type": "agent",
    "source": f"{APPROVED}agents/researcher",
    "version": "1.0.0",
    "digest": DIGEST,
    "local": False,
}
POLICY_PART = {**APPROVED_PART, "id": "$/policies/net", "type": "policy", "source": f"{APPROVED}policies/net"}

START = {
    "hook_event_name": "AgentStart",
    "agent_name": "release-reviewer",
    "parts": [APPROVED_PART],
    "models": [MODEL],
}

RULE = "hooks_server.agent_start"
EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "configs" / "agent_hooks" / "agent_start"
NOT_CONFIGURED = {
    "error": {
        "message": "this hooks server's config has no hooks_server.agent_start section, so it checks no agent starts",
        "type": "not_found_error",
        "param": None,
        "code": "check_not_configured",
    }
}
TOOL_CALLS_NOT_CONFIGURED = {
    "error": {
        "message": "this hooks server's config sets only hooks_server.agent_start, so it checks no tool calls",
        "type": "not_found_error",
        "param": None,
        "code": "check_not_configured",
    }
}


def _start(**fields: Any) -> dict:
    """The approved start, with *fields* replaced."""
    return {**START, **fields}


def _config(**agent_start: Any) -> dict:
    """A config whose hooks_server section sets *agent_start* alone."""
    return {"models": [], "hooks_server": {"agent_start": agent_start}}


def _client(config: dict) -> TestClient:
    return TestClient(create_hooks_app(RailsConfig.from_content(config=config)))


def _ask(config: dict, body: Any):
    with _client(config) as client:
        return client.post("/v1/hooks/agent-start", json=body)


def _decision(config: dict, body: Any) -> dict:
    response = _ask(config, body)
    assert response.status_code == 200, response.text
    return response.json()


def _denied(reason: str) -> dict:
    return {"decision": "deny", "reason": reason}


ALLOWED = {"decision": "allow"}


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    """No engine switch, and the process's deployment type restored after the test."""
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)


class TestSources:
    def test_a_listed_source_is_allowed(self):
        assert _decision(_config(allowed_sources=[APPROVED]), START) == ALLOWED

    def test_an_unlisted_source_is_denied_without_quoting_it(self):
        body = _start(parts=[{**APPROVED_PART, "source": OTHER}])
        assert _decision(_config(allowed_sources=[APPROVED]), body) == _denied(
            f"skill '$/skills/release-checklist' comes from a source {RULE}.allowed_sources doesn't list"
        )

    def test_a_prefix_matches_from_the_start_only(self):
        body = _start(parts=[{**APPROVED_PART, "source": f"mirror+{APPROVED}skills/x"}])
        assert _decision(_config(allowed_sources=[APPROVED]), body)["decision"] == "deny"

    def test_an_empty_list_allows_no_source(self):
        assert _decision(_config(allowed_sources=[]), START)["decision"] == "deny"

    def test_a_called_agent_from_an_unlisted_source_is_denied(self):
        body = _start(parts=[APPROVED_PART, {**CALLED_AGENT, "source": "git::https://elsewhere.example/x.git//a"}])
        assert _decision(_config(allowed_sources=[APPROVED]), body) == _denied(
            f"agent '$/agents/researcher' comes from a source {RULE}.allowed_sources doesn't list"
        )

    @pytest.mark.parametrize("kind", ["skill", "plugin", "connector", "hook", "package", "policy", "agent"])
    def test_every_part_type_is_checked(self, kind):
        body = _start(parts=[{**APPROVED_PART, "type": kind, "source": OTHER}])
        reason = _decision(_config(allowed_sources=[APPROVED]), body)["reason"]
        assert reason.startswith(f"{kind} '$/skills/release-checklist' comes from a source")

    def test_a_called_agent_from_a_listed_source_is_allowed(self):
        body = _start(parts=[APPROVED_PART, CALLED_AGENT])
        assert _decision(_config(allowed_sources=[APPROVED]), body) == ALLOWED

    def test_the_first_denied_part_is_named(self):
        parts = [{**APPROVED_PART, "id": "a", "source": OTHER}, {**APPROVED_PART, "id": "b", "source": OTHER}]
        assert "'a'" in _decision(_config(allowed_sources=[APPROVED]), _start(parts=parts))["reason"]

    def test_a_prefix_must_end_a_path(self):
        with pytest.raises(ValidationError, match="doesn't end in '/'"):
            RailsConfig.from_content(config=_config(allowed_sources=["git::https://git.example.com"]))


class TestLocalSources:
    LOCAL = {**APPROVED_PART, "source": "./skills/release-checklist", "local": True, "version": None}

    def test_a_local_part_is_denied_when_local_sources_are_not_allowed(self):
        assert _decision(_config(allow_local_sources=False), _start(parts=[self.LOCAL])) == _denied(
            f"skill '$/skills/release-checklist' comes from a local folder, and {RULE}.allow_local_sources is false"
        )

    def test_a_local_part_is_allowed_when_local_sources_are(self):
        config = _config(allowed_sources=[APPROVED], allow_local_sources=True)
        assert _decision(config, _start(parts=[self.LOCAL])) == ALLOWED

    def test_a_local_called_agent_is_denied(self):
        part = {**CALLED_AGENT, "source": "./agents/researcher", "local": True}
        reason = _decision(_config(allow_local_sources=False), _start(parts=[APPROVED_PART, part]))["reason"]
        assert reason.startswith("agent '$/agents/researcher' comes from a local folder")


class TestDigests:
    @pytest.mark.parametrize("digest", [None, ""])
    def test_a_part_without_a_digest_is_denied(self, digest):
        body = _start(parts=[{**APPROVED_PART, "digest": digest}])
        assert _decision(_config(require_digests=True), body) == _denied(
            f"skill '$/skills/release-checklist' has no digest, and {RULE}.require_digests is true"
        )

    def test_a_part_with_a_digest_is_allowed(self):
        assert _decision(_config(require_digests=True), START) == ALLOWED

    def test_a_called_agent_without_a_digest_is_denied(self):
        body = _start(parts=[APPROVED_PART, {**CALLED_AGENT, "digest": None}])
        assert _decision(_config(require_digests=True), body)["reason"].startswith("agent '$/agents/researcher'")


class TestModels:
    def test_a_listed_model_is_allowed(self):
        assert _decision(_config(allowed_models=[MODEL]), START) == ALLOWED

    def test_an_unlisted_model_is_denied(self):
        body = _start(models=[MODEL, OTHER_MODEL])
        assert _decision(_config(allowed_models=[MODEL]), body) == _denied(
            f"model '{OTHER_MODEL}' is not in {RULE}.allowed_models"
        )

    def test_an_empty_list_allows_no_model(self):
        assert _decision(_config(allowed_models=[]), START)["decision"] == "deny"

    def test_no_models_named_is_allowed(self):
        assert _decision(_config(allowed_models=[MODEL]), _start(models=[])) == ALLOWED


class TestPolicyEnforcers:
    POLICY = {"id": "$/policies/net", "required": True, "enforcers": ["openshell"]}

    def test_a_required_policy_with_its_enforcer_present_is_allowed(self):
        body = _start(parts=[APPROVED_PART, POLICY_PART], policies=[self.POLICY], enforcers_present=["openshell"])
        assert _decision(_config(require_policy_enforcers=True), body) == ALLOWED

    def test_a_required_policy_without_its_enforcer_is_denied(self):
        body = _start(policies=[self.POLICY], enforcers_present=["nemo-relay"])
        assert _decision(_config(require_policy_enforcers=True), body) == _denied(
            f"policy '$/policies/net' needs enforcer 'openshell', which is not present, and "
            f"{RULE}.require_policy_enforcers is true"
        )

    def test_a_required_policy_that_names_no_enforcer_is_denied(self):
        body = _start(policies=[{**self.POLICY, "enforcers": []}])
        assert _decision(_config(require_policy_enforcers=True), body) == _denied(
            f"policy '$/policies/net' names no enforcer, and {RULE}.require_policy_enforcers is true"
        )

    def test_an_optional_policy_is_not_checked(self):
        body = _start(policies=[{**self.POLICY, "required": False}])
        assert _decision(_config(require_policy_enforcers=True), body) == ALLOWED

    def test_enforcers_are_not_checked_when_the_rule_is_off(self):
        body = _start(policies=[self.POLICY])
        assert _decision(_config(allowed_models=[MODEL]), body) == ALLOWED


class TestEmptyParts:
    def test_no_parts_is_allowed(self):
        config = _config(allowed_sources=[APPROVED], allow_local_sources=False, require_digests=True)
        assert _decision(config, _start(parts=[])) == ALLOWED

    def test_no_parts_still_checks_models(self):
        assert _decision(_config(allowed_models=[MODEL]), _start(parts=[], models=[OTHER_MODEL]))["decision"] == "deny"


class TestMalformedRequests:
    """A body the server can't read is a `422` error, never an allow."""

    @pytest.mark.parametrize(
        "body,param",
        [
            ({k: v for k, v in START.items() if k != "parts"}, "parts"),
            ({k: v for k, v in START.items() if k != "models"}, "models"),
            ({k: v for k, v in START.items() if k != "agent_name"}, "agent_name"),
            (_start(agent_name=""), "agent_name"),
            (_start(parts="all"), "parts"),
            (_start(parts=[{**APPROVED_PART, "type": "model"}]), "parts.0.type"),
            (_start(parts=[{k: v for k, v in APPROVED_PART.items() if k != "local"}]), "parts.0.local"),
            (_start(parts=[{**APPROVED_PART, "local": "no"}]), "parts.0.local"),
            (_start(parts=[{**APPROVED_PART, "local": 0}]), "parts.0.local"),
            (_start(policies=[{"id": "p", "enforcers": [], "required": "false"}]), "policies.0.required"),
            (_start(parts=[{**APPROVED_PART, "source": ""}]), "parts.0.source"),
            (_start(parts=[{**APPROVED_PART, "url": "x"}]), "parts.0.url"),
            (_start(models=[1]), "models.0"),
            (_start(policies=[{"id": "p"}]), "policies.0.enforcers"),
            (_start(session_id="s"), "session_id"),
        ],
    )
    def test_an_invalid_body_is_a_422(self, body, param):
        response = _ask(_config(allowed_models=[MODEL]), body)
        assert response.status_code == 422
        assert response.json()["error"]["param"] == param

    def test_a_body_that_is_not_json_is_a_422(self):
        with _client(_config(allowed_models=[MODEL])) as client:
            response = client.post("/v1/hooks/agent-start", content=b"{not json")
        assert response.status_code == 422

    @pytest.mark.parametrize("body", [[START], "AgentStart", None])
    def test_a_body_that_is_not_an_object_is_a_422(self, body):
        assert _ask(_config(allowed_models=[MODEL]), body).status_code == 422

    def test_another_event_is_unsupported(self):
        response = _ask(_config(allowed_models=[MODEL]), _start(hook_event_name="PreToolUse"))
        assert response.status_code == 422
        assert response.json()["error"] == {
            "message": "hook_event_name 'PreToolUse' is not handled on /v1/hooks/agent-start; send AgentStart",
            "type": "invalid_request_error",
            "param": "hook_event_name",
            "code": "unsupported_hook_event",
        }


class TestConfig:
    def test_an_agent_start_section_with_no_rule_is_refused(self):
        with pytest.raises(HooksConfigError, match="sets no rule that can deny a start"):
            create_hooks_app(RailsConfig.from_content(config=_config()))

    def test_a_config_without_the_section_answers_an_error(self):
        config = {"models": [], "hooks_server": {"allowed_tools": ["Bash"], "allow_unlisted_tools": False}}
        response = _ask(config, START)
        assert response.status_code == 404
        assert response.json() == NOT_CONFIGURED

    @pytest.mark.parametrize("agent", ["claude-code", "codex", "nemo-relay"])
    def test_a_config_for_agent_starts_alone_checks_no_tool_calls(self, agent):
        with _client(_config(allowed_models=[MODEL])) as client:
            response = client.post(f"/v1/hooks/{agent}", json={"hook_event_name": "PreToolUse"})
        assert response.status_code == 404
        assert response.json() == TOOL_CALLS_NOT_CONFIGURED

    def test_one_config_serves_both_checks(self):
        config = {
            "models": [],
            "hooks_server": {
                "allowed_tools": ["search"],
                "allow_unlisted_tools": False,
                "agent_start": {"allowed_models": [MODEL]},
            },
        }
        relay = {
            "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
            "tool_name": "fetch",
            "tool_input": {},
        }
        with _client(config) as client:
            assert client.post("/v1/hooks/agent-start", json=START).json() == ALLOWED
            assert client.post("/v1/hooks/nemo-relay", json=relay).json()["decision"] == "deny"

    def test_a_tool_policy_that_denies_nothing_is_still_refused_beside_agent_start(self):
        config = {
            "models": [],
            "hooks_server": {
                "allowed_tools": ["Bash"],
                "allow_unlisted_tools": True,
                "agent_start": {"allowed_models": []},
            },
        }
        with pytest.raises(HooksConfigError, match="the tool policy denies nothing"):
            create_hooks_app(RailsConfig.from_content(config=config))

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValidationError):
            RailsConfig.from_content(config=_config(allowed_hosts=["x"]))


class TestExampleConfig:
    def test_the_example_allows_the_approved_start_and_denies_the_rest(self):
        with TestClient(create_hooks_app(RailsConfig.from_path(str(EXAMPLE_CONFIG)))) as client:
            ask = lambda body: client.post("/v1/hooks/agent-start", json=body).json()  # noqa: E731
            assert ask(START) == ALLOWED
            assert ask(_start(parts=[{**APPROVED_PART, "source": OTHER}]))["decision"] == "deny"
            assert ask(_start(parts=[{**APPROVED_PART, "local": True}]))["decision"] == "deny"
            assert ask(_start(parts=[{**APPROVED_PART, "digest": None}]))["decision"] == "deny"
            assert ask(_start(models=[OTHER_MODEL]))["decision"] == "deny"
            policy = {"id": "$/policies/net", "enforcers": ["openshell"]}
            assert ask(_start(policies=[policy]))["decision"] == "deny"
            assert ask(_start(policies=[policy], enforcers_present=["openshell"])) == ALLOWED


def test_each_check_logs_one_line_without_sources(caplog):
    caplog.set_level(logging.INFO, logger=hooks_app.__name__)
    _decision(_config(allowed_sources=[APPROVED]), _start(parts=[{**APPROVED_PART, "source": OTHER}]))
    lines = [record.getMessage() for record in caplog.records if "/v1/hooks/agent-start" in record.getMessage()]
    assert len(lines) == 1
    assert "agent_name='release-reviewer' parts=1 models=1 policies=0 outcome=deny" in lines[0]
    assert OTHER not in lines[0]
