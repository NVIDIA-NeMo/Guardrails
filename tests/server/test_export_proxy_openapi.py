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

"""Check the documentation export against the registered proxy contract."""

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/export_proxy_openapi.py"
OFFLINE_RUNNER = """
import runpy
import socket
import sys

def reject_connection(*args, **kwargs):
    raise AssertionError("Proxy reference generation must not access the network")

socket.socket.connect = reject_connection
socket.socket.connect_ex = reject_connection
socket.create_connection = reject_connection
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_export(output, *args):
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "NVIDIA_INFERENCE_KEY"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-c", OFFLINE_RUNNER, str(SCRIPT), "--output", str(output), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_export_matches_registered_proxy_routes(tmp_path):
    from nemoguardrails.server.experimental._application import create_proxy_application
    from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration

    output = tmp_path / "proxy.yml"
    result = run_export(output)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = yaml.safe_load(output.read_text())
    app = create_proxy_application(providers=[create_openai_integration(base_url="https://api.openai.com/v1")])
    assert actual == app.openapi()
    assert set(actual["paths"]) == {"/v1/chat/completions", "/v1/responses"}
    for path in actual["paths"]:
        operation = actual["paths"][path]["post"]
        assert operation["requestBody"]["required"] is True
        assert operation["requestBody"]["content"]["application/json"]["schema"]["properties"]
        assert set(operation["responses"]["200"]["content"]) == {"application/json", "text/event-stream"}
        assert {"400", "405", "413", "415", "422", "502"} <= operation["responses"].keys()
    assert (
        actual["paths"]["/v1/responses"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"][
            "store"
        ]["const"]
        is False
    )


def test_export_is_deterministic_and_check_is_read_only(tmp_path):
    output = tmp_path / "proxy.yml"
    assert run_export(output).returncode == 0
    original = output.read_bytes()
    assert run_export(output, "--check").returncode == 0
    assert output.read_bytes() == original
    output.write_text("stale\n")
    assert run_export(output, "--check").returncode == 1
    assert output.read_text() == "stale\n"
    output.unlink()
    assert run_export(output, "--check").returncode == 1
    assert not output.exists()
    assert run_export(output).returncode == 0
    assert output.read_bytes() == original


def test_selected_operations_and_mount_change_registered_routes():
    from nemoguardrails.server.experimental._application import create_proxy_application
    from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration

    provider = create_openai_integration(
        base_url="https://api.openai.com/v1", mount_path="/provider", operations=["responses"]
    )
    app = create_proxy_application(providers=[provider], path_prefix="/proxy")
    assert set(app.openapi()["paths"]) == {"/proxy/provider/v1/responses"}


def test_overlay_examples_match_guarded_bindings():
    from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
    from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT

    overlay = yaml.safe_load((ROOT / "fern/proxy-openapi-overlay.yml").read_text())
    actions = {action["target"]: action["update"] for action in overlay["actions"]}
    for path, endpoint in (("/v1/chat/completions", CHAT_COMPLETIONS_ENDPOINT), ("/v1/responses", RESPONSES_ENDPOINT)):
        operation = actions[f"$.paths['{path}'].post"]
        request = operation["requestBody"]["content"]["application/json"]["examples"]["buffered"]["value"]
        response_examples = operation["responses"]["200"]["content"]["application/json"]["examples"]
        response = next(iter(response_examples.values()))["value"]
        paired = operation["x-fern-examples"][0]
        assert paired["request"] == request
        assert paired["response"]["body"] == response
        assert '"model":"example-model"' in paired["code-samples"][0]["code"]
        endpoint.guarded_request_model.validate_payload(request).locate_guarded_message(request)
        endpoint.guarded_response_model.validate_payload(response).locate_guarded_message(response)
