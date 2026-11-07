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

"""Exercise the exporter CLI against the real application without network access."""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_openapi.py"
OFFLINE_RUNNER = """
import runpy
import socket
import sys

def reject_connection(*args, **kwargs):
    raise AssertionError("OpenAPI generation must not access the network")

socket.socket.connect = reject_connection
socket.socket.connect_ex = reject_connection
socket.create_connection = reject_connection
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_export(output: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "NVIDIA_INFERENCE_KEY"):
        env.pop(key, None)
    # The script must produce the same contract even when chat UI is enabled.
    env["NEMO_GUARDRAILS_DISABLE_CHAT_UI"] = "false"
    return subprocess.run(
        [sys.executable, "-c", OFFLINE_RUNNER, str(SCRIPT), "--output", str(output), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    output = tmp_path_factory.mktemp("openapi") / "openapi.yml"
    result = run_export(output)
    assert result.returncode == 0, result.stdout + result.stderr
    return output.read_text(encoding="utf-8")


def test_export_tracks_tool_rail_rename_and_current_server_contract(exported):
    spec = yaml.safe_load(exported)
    schemas = spec["components"]["schemas"]
    rails = schemas["GenerationRailsOptions"]["properties"]
    assert rails["tool_call"]["default"] is True
    assert rails["tool_result"]["default"] is True
    assert rails["tool_output"]["deprecated"] is True
    assert rails["tool_input"]["deprecated"] is True
    assert "/v1/checks" in spec["paths"]
    request = schemas["GuardrailsChatCompletionRequest"]["properties"]
    assert {"tools", "tool_choice", "parallel_tool_calls"} <= request.keys()


def test_freshness_check_accepts_current_output_without_writing(tmp_path, exported):
    output = tmp_path / "openapi.yml"
    output.write_text(exported, encoding="utf-8")
    before = output.stat().st_mtime_ns
    result = run_export(output, "--check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.stat().st_mtime_ns == before
    assert output.read_text(encoding="utf-8") == exported


@pytest.mark.parametrize("existing", [None, "openapi: 3.0.3\npaths: {}\n"])
def test_freshness_check_rejects_missing_or_stale_output_without_writing(tmp_path, existing):
    output = tmp_path / "openapi.yml"
    if existing is not None:
        output.write_text(existing, encoding="utf-8")
    result = run_export(output, "--check")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "uv run --locked python scripts/export_openapi.py" in result.stdout
    if existing is None:
        assert not output.exists()
    else:
        assert output.read_text(encoding="utf-8") == existing


def test_export_is_identical_across_processes(tmp_path, exported):
    output = tmp_path / "openapi.yml"
    result = run_export(output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == exported


def test_generated_spec_is_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "fern/openapi.yml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "fern/openapi.yml"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "fern/openapi.yml"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode != 0


@pytest.mark.parametrize("status", [0, 1])
def test_breaking_check_compares_generated_base_instead_of_git(monkeypatch, tmp_path, exported, status):
    check_breaking = runpy.run_path(str(ROOT / "scripts/openai_coverage.py"))["_check_breaking"]
    base = tmp_path / "base.yml"
    current = tmp_path / "current.yml"
    base.write_text(exported, encoding="utf-8")
    current.write_text(exported, encoding="utf-8")
    calls = []

    def compare(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, status, stdout="comparison result", stderr="")

    monkeypatch.setattr(subprocess, "run", compare)
    assert check_breaking(str(current), base_spec=base) is (status == 0)
    assert len(calls) == 1
    assert calls[0][:4] == ["oasdiff", "breaking", str(base), str(current)]


def test_breaking_check_fails_when_generated_baseline_is_missing(tmp_path):
    check_breaking = runpy.run_path(str(ROOT / "scripts/openai_coverage.py"))["_check_breaking"]
    with pytest.raises(FileNotFoundError, match="Generated comparison specification is missing"):
        check_breaking(str(tmp_path / "current.yml"), base_spec=tmp_path / "missing.yml")


def test_documented_checks_example_reaches_the_engine(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    from nemoguardrails import RailsConfig
    from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType
    from nemoguardrails.server import api

    overlay = yaml.safe_load((ROOT / "fern/openapi-overlay.yml").read_text(encoding="utf-8"))
    operation = next(a["update"] for a in overlay["actions"] if a["target"] == "$.paths['/v1/checks'].post")
    request = operation["requestBody"]["content"]["application/json"]["examples"]["input"]["value"]
    expected = operation["responses"]["200"]["content"]["application/json"]["examples"]["passed"]["value"]
    rails = MagicMock()
    rails.config = RailsConfig.from_content(yaml_content="models: []")
    rails.check_async = AsyncMock(return_value=RailsResult(status=RailStatus.PASSED, content=expected["content"]))
    load_rails = AsyncMock(return_value=rails)
    monkeypatch.setattr(api, "_get_rails", load_rails)
    client = TestClient(api.app)
    try:
        response = client.post("/v1/checks", json=request)
    finally:
        client.close()
    assert response.status_code == 200
    assert response.json() == expected
    load_rails.assert_awaited_once_with(["nemotron-3.5-content-safety"], model_name=request["model"])
    rails.check_async.assert_awaited_once_with(messages=request["messages"], rail_types=[RailType.INPUT], tools=None)


@pytest.mark.parametrize("upstream_changed", [False, True])
@pytest.mark.parametrize("base_has_exporter", [False, True])
@pytest.mark.parametrize("runtime_changed", [False, True])
def test_conformance_baseline_initialization_requires_unchanged_runtime(
    tmp_path, base_has_exporter, runtime_changed, upstream_changed
):
    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=OpenAPI test",
                "-c",
                "user.email=openapi-test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *args,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    git("init", "--quiet")
    (tmp_path / "nemoguardrails").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "fern").mkdir()
    (tmp_path / "fern/fern.config.json").write_text('{"version": "5.123.0"}\n', encoding="utf-8")
    runtime = tmp_path / "nemoguardrails" / "api.py"
    runtime.write_text("original\n", encoding="utf-8")
    exporter = tmp_path / "scripts" / "export_openapi.py"
    if base_has_exporter:
        exporter.write_text("original exporter\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "base")
    base = git("rev-parse", "HEAD").stdout.strip()
    git("branch", "base")
    git("checkout", "--quiet", "-b", "pull-request")
    exporter.write_text("updated exporter\n", encoding="utf-8")
    if runtime_changed:
        runtime.write_text("changed\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "change")
    git("checkout", "--quiet", "base")
    if upstream_changed:
        (tmp_path / "nemoguardrails/upstream.py").write_text("upstream change\n", encoding="utf-8")
        git("add", ".")
        git("commit", "--quiet", "-m", "upstream change after PR event")
    git("merge", "--quiet", "--no-ff", "pull-request", "-m", "GitHub test merge")

    # Stand in for the compatibility command; this test exercises the real CI guard.
    executable = tmp_path / "uv"
    executable.write_text("#!/bin/sh\necho compatibility-check-ran\n", encoding="utf-8", newline="\n")
    executable.chmod(0o755)
    npx = tmp_path / "npx"
    npx.write_text("#!/bin/sh\necho fern-export-ran\n", encoding="utf-8", newline="\n")
    npx.chmod(0o755)
    workflow = yaml.safe_load((ROOT / ".github/workflows/api-conformance-docs.yml").read_text(encoding="utf-8"))
    step = next(s for s in workflow["jobs"]["docs"]["steps"] if s.get("name") == "Check for breaking changes")
    bash = "bash"
    if sys.platform == "win32":
        # Use Git Bash rather than the Windows System32 WSL launcher.
        git_root = Path(git("--exec-path").stdout.strip()).parents[2]
        bash = str(git_root / "bin" / "bash.exe")
    result = subprocess.run(
        [bash, "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "BASE_REF": base,
            "RUNNER_TEMP": (tmp_path / "runner").as_posix(),
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    initialize = not base_has_exporter and not runtime_changed
    assert ("Initializing generated OpenAPI baseline" in result.stdout) is initialize
    assert ("compatibility-check-ran" in result.stdout) is not initialize
