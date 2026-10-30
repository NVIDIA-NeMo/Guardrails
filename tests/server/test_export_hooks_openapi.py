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

"""Verify the generated hooks contract against the running server."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import jsonschema
import pytest
import yaml
from fastapi.testclient import TestClient

from nemoguardrails.exceptions import NonStreamingWorkQueueFullError
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks.app import create_hooks_app
from tests.server.test_export_openapi import OFFLINE_RUNNER
from tests.server.test_hooks_server import HOOKS_CONFIG, _hook

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/export_hooks_openapi.py"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    output = tmp_path_factory.mktemp("hooks-openapi") / "spec.yml"
    env = {**os.environ, "OPENAI_API_KEY": "", "NVIDIA_API_KEY": "", "NVIDIA_INFERENCE_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-c", OFFLINE_RUNNER, str(SCRIPT), "--output", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output, yaml.safe_load(output.read_text())


def validate(spec, schema, body):
    jsonschema.validate(body, {**schema, "components": spec["components"]})


def test_export_is_fresh_and_routes_are_complete(exported):
    output, spec = exported
    assert set(spec["paths"]) == {
        "/v1/hooks/claude-code",
        "/v1/hooks/codex",
        "/v1/hooks/nemo-relay",
        "/v1/health",
        "/healthz",
    }
    before = output.read_bytes()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--check"], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes() == before
    output.write_text("stale")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--check"], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 1
    assert output.read_text() == "stale"


@pytest.mark.parametrize("agent", ["claude-code", "codex", "nemo-relay"])
def test_generated_schemas_describe_real_requests_and_responses(exported, agent, monkeypatch):
    _, spec = exported
    operation = spec["paths"][f"/v1/hooks/{agent}"]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    config = RailsConfig.from_content(config=HOOKS_CONFIG)
    with TestClient(create_hooks_app(config)) as client:
        for value in (None, "text", 1, False, ["item"], {"command": "npm test"}):
            validate(spec, request_schema, _hook(agent, "Bash", value))
        non_object = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", None))
        assert non_object.status_code == 200
        validate(spec, operation["responses"]["200"]["content"]["application/json"]["schema"], non_object.json())
        for command in ("npm test", "rm -rf /tmp/build"):
            body = _hook(agent, "Bash", {"command": command})
            validate(spec, request_schema, body)
            response = client.post(f"/v1/hooks/{agent}", json=body)
            assert response.status_code == 200
            validate(spec, operation["responses"]["200"]["content"]["application/json"]["schema"], response.json())
        invalid = client.post(f"/v1/hooks/{agent}", json={})
        assert invalid.status_code == 422
        validate(spec, operation["responses"]["422"]["content"]["application/json"]["schema"], invalid.json())
        monkeypatch.setattr(client.app.state.engine, "check_async", AsyncMock(side_effect=RuntimeError("failure")))
        failed = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", {"command": "npm test"}))
        assert failed.status_code == 500
        validate(spec, operation["responses"]["500"]["content"]["application/json"]["schema"], failed.json())
        monkeypatch.setattr(
            client.app.state.engine, "check_async", AsyncMock(side_effect=NonStreamingWorkQueueFullError())
        )
        overloaded = client.post(f"/v1/hooks/{agent}", json=_hook(agent, "Bash", {"command": "npm test"}))
        assert overloaded.status_code == 503 and overloaded.headers["retry-after"] == "30"
        validate(spec, operation["responses"]["503"]["content"]["application/json"]["schema"], overloaded.json())


@pytest.mark.parametrize("path", ["/v1/health", "/healthz"])
def test_health_contract_and_no_live_openapi_route(exported, path):
    _, spec = exported
    with TestClient(create_hooks_app(RailsConfig.from_content(config=HOOKS_CONFIG))) as client:
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/health+json"
        schema = spec["paths"][path]["get"]["responses"]["200"]["content"]["application/health+json"]["schema"]
        validate(spec, schema, response.json())
        assert client.get("/openapi.json").status_code == 404


def test_generated_health_pages_match_real_responses(tmp_path):
    output = tmp_path / "spec.yml"
    pages = tmp_path / "health"
    command = [sys.executable, str(SCRIPT), "--output", str(output), "--health-docs-dir", str(pages)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    with TestClient(create_hooks_app(RailsConfig.from_content(config=HOOKS_CONFIG))) as client:
        for path, name in [("/v1/health", "get-health.mdx"), ("/healthz", "get-healthz.mdx")]:
            response = client.get(path)
            page = (pages / name).read_text()
            assert f"GET http://localhost:8100{path}" in page
            assert f"`{response.status_code}`" in page
            assert f"`{response.headers['content-type']}`" in page
            assert json.loads(page.split("```json\n")[1].split("\n```")[0]) == response.json()
    assert subprocess.run([*command, "--check"], capture_output=True, text=True, timeout=90).returncode == 0
    (pages / "get-health.mdx").write_text("stale")
    assert subprocess.run([*command, "--check"], capture_output=True, text=True, timeout=90).returncode == 1
    assert (pages / "get-health.mdx").read_text() == "stale"


@pytest.mark.parametrize(
    "agent,folder", [("claude-code", "claude_code"), ("codex", "codex"), ("nemo-relay", "nemo_relay")]
)
def test_documented_examples_match_example_config(agent, folder):
    overlay = yaml.safe_load((ROOT / "fern/hooks-openapi-overlay.yml").read_text())
    target = f"$.paths['/v1/hooks/{agent}'].post"
    operation = next(action["update"] for action in overlay["actions"] if action["target"] == target)
    requests = operation["requestBody"]["content"]["application/json"]["examples"]
    responses = operation["responses"]["200"]["content"]["application/json"]["examples"]
    config = RailsConfig.from_path(str(ROOT / "examples/configs/agent_hooks" / folder))
    with TestClient(create_hooks_app(config)) as client:
        for name in ("allow", "deny"):
            response = client.post(f"/v1/hooks/{agent}", json=requests[name]["value"])
            assert (response.status_code, response.json()) == (200, responses[name]["value"])
