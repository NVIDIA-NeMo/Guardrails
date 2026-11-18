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

"""The decision record, chained by hash, and the check of a model call during a run."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from nemoguardrails import telemetry
from nemoguardrails.cli import app as cli_app
from nemoguardrails.cli import hooks_record
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks import decision_record
from nemoguardrails.server.hooks.app import HooksConfigError, create_hooks_app
from tests.server.test_hooks_server import _hook

MODEL = "bedrock::nvidia.nemotron-super-3-120b"
OTHER = "bedrock::meta.llama3-70b-instruct-v1:0"
AGENT = {"x-nemo-agent-name": "release-reviewer", "x-nemo-agent-digest": "sha256:aaaa", "x-nemo-run-id": "run-1"}
START = {"hook_event_name": "AgentStart", "agent_name": "release-reviewer", "agent_digest": "sha256:aaaa",
         "parts": [], "models": [MODEL]}  # fmt: skip


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)


def _config(tmp_path: Path, **hooks_server) -> dict:
    return {
        "models": [],
        "hooks_server": {
            "decision_record": str(tmp_path / "decisions.jsonl"),
            "quarantine_store": str(tmp_path / "quarantine.jsonl"),
            "agent_start": {"allowed_models": [MODEL]},
            "stop_tools": ["delete_index"],
            "blocked_tools": ["WebFetch"],
            **hooks_server,
        },
    }


def _client(tmp_path: Path, **hooks_server) -> TestClient:
    return TestClient(create_hooks_app(RailsConfig.from_content(config=_config(tmp_path, **hooks_server))))


def _model_call(client: TestClient, model: str, **extra) -> dict:
    body = {"hook_event_name": "ModelCall", "agent_name": "release-reviewer", "agent_digest": "sha256:aaaa",
            "model": model, **extra}  # fmt: skip
    response = client.post("/v1/hooks/model-call", json=body, headers={"x-nemo-run-id": "run-1"})
    assert response.status_code == 200, response.text
    return response.json()


class TestModelCall:
    def test_the_allowed_models_rule_decides_each_call(self, tmp_path):
        with _client(tmp_path) as client:
            assert _model_call(client, MODEL) == {"decision": "allow", "rule": "hooks_server.agent_start"}
            answer = _model_call(client, OTHER)
            assert answer["decision"] == "deny" and OTHER in answer["reason"]
            assert answer["rule"] == "hooks_server.agent_start"

    def test_cedar_decides_each_call(self, tmp_path):
        pytest.importorskip("cedarpy")
        policies = tmp_path / "start"
        policies.mkdir()
        (policies / "models.cedar").write_text(
            'permit(principal, action == Hooks::Action::"start", resource);\n'
            '@id("model-must-be-listed")\n'
            'forbid(principal, action == Hooks::Action::"start", resource is Hooks::Model)\n'
            f'unless {{ ["{MODEL}"].contains(resource.name) }};\n'
        )
        config = _config(tmp_path, agent_start={"cedar": {"policies": str(policies)}})
        with TestClient(create_hooks_app(RailsConfig.from_content(config=config))) as client:
            assert _model_call(client, MODEL)["decision"] == "allow"
            answer = _model_call(client, OTHER)
            assert answer["decision"] == "deny" and "model-must-be-listed" in answer["reason"]
            assert answer["rule"] == "cedar policy"

    def test_a_quarantined_agent_may_call_no_model(self, tmp_path):
        with _client(tmp_path) as client:
            client.post(
                "/v1/quarantines",
                json={
                    "agent_name": "release-reviewer",
                    "agent_digest": "sha256:aaaa",
                    "reason": "r",
                    "requested_by": "p",
                },
            )
            answer = _model_call(client, MODEL)
            assert answer["decision"] == "deny" and answer["rule"] == "quarantine"

    def test_another_event_is_refused(self, tmp_path):
        with _client(tmp_path) as client:
            response = client.post("/v1/hooks/model-call", json={**START, "model": MODEL})
            assert response.status_code == 422

    def test_without_agent_start_rules_it_answers_an_error(self, tmp_path):
        config = {"models": [], "hooks_server": {"stop_tools": ["x"]}}
        with TestClient(create_hooks_app(RailsConfig.from_content(config=config))) as client:
            response = client.post("/v1/hooks/model-call", json={"hook_event_name": "ModelCall", "agent_name": "a",
                                                                  "model": MODEL})  # fmt: skip
            assert response.status_code == 404


class TestDecisionRecord:
    def _decide_four(self, client: TestClient) -> None:
        assert client.post("/v1/hooks/agent-start", json=START).json() == {"decision": "allow"}
        assert client.post("/v1/hooks/agent-start", json={**START, "models": [OTHER]}).json()["decision"] == "deny"
        client.post("/v1/hooks/claude-code", json=_hook("claude-code", "delete_index", {"index": "x"}), headers=AGENT)
        _model_call(client, OTHER, id="model-2")

    def test_every_decision_is_recorded_with_who_what_and_why(self, tmp_path):
        with _client(tmp_path) as client:
            self._decide_four(client)
            entries = client.get("/v1/decisions").json()["decisions"]
        assert [(e["seq"], e["kind"], e["outcome"]) for e in entries] == [
            (1, "agent_start", "allow"),
            (2, "agent_start", "deny"),
            (3, "tool_call", "stop"),
            (4, "model_call", "deny"),
        ]
        stop = entries[2]
        assert (stop["agent_name"], stop["agent_digest"], stop["run_id"], stop["rule"]) == (
            "release-reviewer",
            "sha256:aaaa",
            "run-1",
            "tool policy",
        )
        # The tool's input stays out of the record: only its argument names.
        assert stop["asked"] == {"hook": "claude-code", "tool_name": "delete_index", "argument_names": ["index"],
                                 "call_id": stop["asked"]["call_id"]}  # fmt: skip
        assert entries[3]["asked"] == {"model": OTHER, "id": "model-2"}
        assert entries[1]["asked"]["models"] == [OTHER] and entries[1]["reason"]
        assert all(e["by"] == "nemoguardrails hooks-server" and e["at"] for e in entries)
        assert entries[0]["prev"] == decision_record.GENESIS
        assert [e["prev"] for e in entries[1:]] == [e["hash"] for e in entries[:-1]]

    def test_the_endpoints_filter_and_check(self, tmp_path):
        with _client(tmp_path) as client:
            self._decide_four(client)
            denied = client.get("/v1/decisions", params={"outcome": "deny", "kind": "model_call"}).json()
            assert [e["seq"] for e in denied["decisions"]] == [4]
            assert [e["seq"] for e in client.get("/v1/decisions", params={"run_id": "run-1"}).json()["decisions"]] == [
                3,
                4,
            ]
            result = client.get("/v1/decisions/check").json()
            assert (result["entries"], result["intact"], result["problems"]) == (4, True, [])
            assert client.get("/v1/decisions/check", params={"head": result["head"]}).json()["intact"] is True
            assert client.get("/v1/decisions/check", params={"head": "sha256:00"}).json()["intact"] is False

    def test_a_changed_line_breaks_the_chain_where_it_was_changed(self, tmp_path):
        with _client(tmp_path) as client:
            self._decide_four(client)
        path = tmp_path / "decisions.jsonl"
        lines = path.read_text().splitlines()
        lines[1] = lines[1].replace('"deny"', '"allow"')
        path.write_text("\n".join(lines) + "\n")
        result = decision_record.check(path)
        assert result["intact"] is False
        assert result["problems"] == ["line 2: the entry was changed after it was written"]

    def test_a_removed_line_breaks_the_chain(self, tmp_path):
        with _client(tmp_path) as client:
            self._decide_four(client)
        path = tmp_path / "decisions.jsonl"
        lines = path.read_text().splitlines()
        del lines[1]
        path.write_text("\n".join(lines) + "\n")
        assert decision_record.check(path)["problems"] == ["line 2: prev is not the hash of the line before it"]

    def test_the_record_survives_a_restart_and_goes_on_from_its_last_entry(self, tmp_path):
        with _client(tmp_path) as client:
            client.post("/v1/hooks/agent-start", json=START)
        with _client(tmp_path) as restarted:
            restarted.post("/v1/hooks/agent-start", json=START)
            result = restarted.get("/v1/decisions/check").json()
        assert (result["entries"], result["intact"]) == (2, True)

    def test_a_broken_record_refuses_to_start_the_server(self, tmp_path):
        (tmp_path / "decisions.jsonl").write_text('{"seq": 1, "prev": "x", "hash": "y"}\n')
        with pytest.raises(HooksConfigError, match="is not intact"):
            _client(tmp_path)

    def test_a_record_that_cannot_be_written_answers_an_error_not_an_allow(self, tmp_path):
        with _client(tmp_path) as client:
            (tmp_path / "decisions.jsonl").unlink(missing_ok=True)
            (tmp_path / "decisions.jsonl").mkdir()
            response = client.post("/v1/hooks/agent-start", json=START)
            assert response.status_code == 500
            assert response.json()["error"]["code"] == "decision_record_failed"

    def test_a_config_for_agent_starts_alone_can_keep_a_record(self, tmp_path):
        config = {
            "models": [],
            "hooks_server": {
                "decision_record": str(tmp_path / "decisions.jsonl"),
                "agent_start": {"allowed_models": [MODEL]},
            },
        }
        with TestClient(create_hooks_app(RailsConfig.from_content(config=config))) as client:
            assert client.post("/v1/hooks/agent-start", json=START).json() == {"decision": "allow"}
            assert client.post("/v1/hooks/nemo-relay", json=_hook("nemo-relay", "x", {})).status_code == 404
            assert client.get("/v1/decisions/check").json()["entries"] == 1

    def test_without_a_record_the_endpoints_answer_an_error(self, tmp_path):
        config = {"models": [], "hooks_server": {"stop_tools": ["x"]}}
        with TestClient(create_hooks_app(RailsConfig.from_content(config=config))) as client:
            assert client.get("/v1/decisions").status_code == 404
            assert client.get("/v1/decisions/check").status_code == 404


class TestCommands:
    def test_show_and_check(self, tmp_path):
        with _client(tmp_path) as client:
            TestDecisionRecord()._decide_four(client)
        path = str(tmp_path / "decisions.jsonl")
        runner = CliRunner()
        shown = runner.invoke(cli_app, ["hooks-record", "show", path, "--run", "run-1", "--outcome", "deny"])
        assert shown.exit_code == 0
        assert [json.loads(line)["kind"] for line in shown.stdout.splitlines()] == ["model_call"]
        checked = runner.invoke(cli_app, ["hooks-record", "check", path])
        assert checked.exit_code == 0 and json.loads(checked.stdout)["intact"] is True
        Path(path).write_text(Path(path).read_text().replace('"stop"', '"allow"'))
        broken = runner.invoke(cli_app, ["hooks-record", "check", path])
        assert broken.exit_code == 1 and json.loads(broken.stdout)["intact"] is False

    def test_decide_asks_each_model_call_of_a_file(self, tmp_path, monkeypatch):
        package = tmp_path / "package"
        package.mkdir()
        (package / "agent.json").write_text(json.dumps({"name": "release-reviewer"}))
        (package / "oadl.lock.json").write_text("{}")
        digest = "sha256:" + hashlib.sha256(b"{}").hexdigest()
        actions = tmp_path / "model-calls.json"
        actions.write_text(json.dumps([
            {"id": "model-1", "kind": "call_model", "agent": "$", "depth": 0, "model": MODEL, "run": "govern-lab-3"},
            {"id": "model-2", "kind": "call_model", "agent": "$", "depth": 0, "model": OTHER, "run": "govern-lab-3"},
        ]))  # fmt: skip
        with _client(tmp_path) as client:

            def urlopen(request, timeout):
                answer = client.post(
                    request.full_url.replace("http://127.0.0.1:8100", ""),
                    content=request.data,
                    headers=dict(request.header_items()),
                )
                return io.BytesIO(answer.content)

            monkeypatch.setattr(hooks_record.urllib.request, "urlopen", urlopen)
            result = CliRunner().invoke(cli_app, ["hooks-decide", str(package), "--action", str(actions)])
            entries = client.get("/v1/decisions", params={"run_id": "govern-lab-3"}).json()["decisions"]
        assert result.exit_code == 2
        answer = json.loads(result.stdout)
        assert (answer["decision"], answer["agent_digest"]) == ("deny", digest)
        assert [(d["id"], d["subject"], d["decision"]) for d in answer["decisions"]] == [
            ("model-1", MODEL, "allow"),
            ("model-2", OTHER, "deny"),
        ]
        assert [(e["kind"], e["outcome"], e["agent_digest"]) for e in entries] == [
            ("model_call", "allow", digest),
            ("model_call", "deny", digest),
        ]
