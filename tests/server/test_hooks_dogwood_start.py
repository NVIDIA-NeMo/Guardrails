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

"""Dogwood at agent start: a start decided from events recorded from outside any run, such as a scan of a skill."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemoguardrails import telemetry
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks import app as hooks_app
from nemoguardrails.server.hooks.app import create_hooks_app

pytest.importorskip("cedarpy")
pytest.importorskip("nemoguardrails_dogwood")

EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "configs" / "agent_hooks" / "dogwood_start"
SCANNED = "sha256:" + "a" * 64
UNSCANNED = "sha256:" + "b" * 64
TOKEN = "scan-stage-test-token"
RULE = "skills/scanned-within-6h"


@pytest.fixture(autouse=True)
def _hooks_server_env(monkeypatch):
    monkeypatch.delenv("NEMO_GUARDRAILS_IORAILS_ENGINE", raising=False)
    monkeypatch.setattr(telemetry, "_deployment_type_override", None)
    monkeypatch.setenv("SCAN_STAGE_TOKEN", TOKEN)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(create_hooks_app(RailsConfig.from_path(str(EXAMPLE_CONFIG)))) as client:
        yield client


def _start(client: TestClient, digest: Optional[str], part_type: str = "skill") -> dict:
    part = {"id": "$/skills/release-checklist", "type": part_type, "source": "git::https://x/y.git//s", "local": False}
    if digest:
        part["digest"] = digest
    else:
        part["local"] = True
    body = {"hook_event_name": "AgentStart", "agent_name": "release-reviewer", "parts": [part], "models": []}
    response = client.post("/v1/hooks/agent-start", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _scan(client: TestClient, digest: str, result: str = "passed", token: Optional[str] = TOKEN):
    headers = {} if token is None else {"Authorization": f"Bearer {token}"}
    body = {"action": "scan", "kind": "response", "input": {"digest": digest, "result": result}}
    return client.post("/v1/events", json=body, headers=headers)


def _later(monkeypatch, hours: float) -> None:
    real = datetime

    class Later(real):
        @classmethod
        def now(cls, tz=None):
            return real.now(tz) + timedelta(hours=hours)

    monkeypatch.setattr(hooks_app, "datetime", Later)


def test_a_skill_scanned_in_the_window_starts_and_one_never_scanned_is_refused_by_the_rule(client):
    assert _scan(client, SCANNED).status_code == 200
    assert _start(client, SCANNED) == {"decision": "allow"}
    refused = _start(client, UNSCANNED)
    assert refused["decision"] == "deny"
    assert refused["reason"] == f"skill '$/skills/release-checklist' is denied by Dogwood policy '{RULE}'"


def test_when_the_scan_is_older_than_the_window_the_start_is_refused(client, monkeypatch):
    _scan(client, SCANNED)
    _later(monkeypatch, 5.9)
    assert _start(client, SCANNED)["decision"] == "allow"
    _later(monkeypatch, 6.1)
    assert _start(client, SCANNED)["decision"] == "deny"


def test_a_failed_scan_does_not_count(client):
    _scan(client, SCANNED, result="failed")
    assert _start(client, SCANNED)["decision"] == "deny"


def test_a_skill_without_a_digest_is_refused_and_another_part_type_needs_no_scan(client):
    assert _start(client, None)["decision"] == "deny"
    assert _start(client, UNSCANNED, part_type="plugin") == {"decision": "allow"}


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_only_a_named_writer_with_its_token_can_record_a_scan(client, tmp_path, token):
    response = _scan(client, SCANNED, token=token)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert _start(client, SCANNED)["decision"] == "deny"
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert "event" not in kinds


def test_the_record_names_the_writer_of_each_event(client, tmp_path):
    _scan(client, SCANNED)
    entry = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert (entry["kind"], entry["by"], entry["asked"]["input"]["digest"]) == ("event", "scan-stage", SCANNED)


def test_a_writer_whose_token_variable_is_unset_cannot_write(client, monkeypatch):
    monkeypatch.delenv("SCAN_STAGE_TOKEN")
    assert _scan(client, SCANNED, token="").status_code == 401


def test_a_record_that_cannot_be_read_refuses_the_start(client, tmp_path):
    _scan(client, SCANNED)
    record = tmp_path / "decisions.jsonl"
    record.write_text("not json\n" + record.read_text(), encoding="utf-8")
    refused = _start(client, SCANNED)
    assert refused["decision"] == "deny"
    assert "can't be read" in refused["reason"]


def test_a_server_without_writers_records_no_event(tmp_path):
    config = {"models": [], "hooks_server": {"decision_record": str(tmp_path / "d.jsonl"), "blocked_tools": ["x"]}}
    with TestClient(create_hooks_app(RailsConfig.from_content(config=config))) as client:
        assert _scan(client, SCANNED).status_code == 404


@pytest.mark.parametrize(
    ("hooks_server", "message"),
    [
        ({"agent_start": {"dogwood": {"policies": "p", "schema": "s"}}}, "no decision_record"),
        (
            {
                "decision_record": "d",
                "agent_start": {"dogwood": {"policies": "p", "schema": "s"}, "allowed_models": []},
            },
            "Dogwood alone",
        ),
        ({"event_writers": {"scan-stage": "SCAN_STAGE_TOKEN"}}, "event_writers but no decision_record"),
    ],
)
def test_a_config_that_could_not_read_or_keep_events_is_refused(hooks_server, message):
    with pytest.raises(ValidationError, match=message):
        RailsConfig.from_content(config={"models": [], "hooks_server": hooks_server})
