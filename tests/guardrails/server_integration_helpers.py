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

"""Shared harness for real-subprocess server integration tests.

Spawns the mock tool LLM server (tests/mock_tool_llm_server) and a real, unmodified
`nemoguardrails server` instance as two real subprocesses over real HTTP, and drives
them with real network calls, matching the benchmark/ suite's operational pattern
rather than the tests/server/ TestClient convention. Readiness polling is modeled on
benchmark/locust/run_locust.py's `_check_service`/`_get`, the only existing
health-check polling logic in the repo.

Used by test_per_tool_regex_rails_server.py and test_tool_safety_check_server.py, which
differ only in the config.yml they write and the mock server's environment.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, NamedTuple, Optional

import httpx

STARTUP_TIMEOUT_SECONDS = 20
POLL_INTERVAL_SECONDS = 0.25


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, process: subprocess.Popen) -> None:
    """Poll *url* until it answers, raising if *process* exits first or time runs out."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited early (code {process.returncode}) before {url} became healthy")
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code == httpx.codes.OK:
                return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"{url} did not become healthy within {STARTUP_TIMEOUT_SECONDS}s")


class Servers(NamedTuple):
    guardrails_url: str
    config_id: str
    mock_url: str


def spawn_servers(
    tmp_path_factory,
    config_id: str,
    write_config: Callable[[Path, str, int], None],
    mock_env: Optional[Dict[str, str]] = None,
) -> Iterator[Servers]:
    """Write *config_id*'s config via *write_config*, then spawn both servers.

    *write_config* receives (config_dir, config_id, mock_port) and must create
    config_dir/config_id/config.yml. *mock_env* is merged into the mock server
    subprocess's environment, for tests that need its MOCK_TOOL_LLM_* switches.
    """
    mock_port = _free_port()
    guardrails_port = _free_port()
    mock_url = f"http://127.0.0.1:{mock_port}"
    config_dir = tmp_path_factory.mktemp(f"{config_id}_configs")
    write_config(config_dir, config_id, mock_port)

    mock_process = subprocess.Popen(
        [sys.executable, "-m", "tests.mock_tool_llm_server.run_server", "--port", str(mock_port)],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, **(mock_env or {})},
    )
    guardrails_process = None
    try:
        _wait_for_health(f"{mock_url}/health", mock_process)

        guardrails_process = subprocess.Popen(
            [
                "nemoguardrails",
                "server",
                "--config",
                str(config_dir),
                "--default-config-id",
                config_id,
                "--port",
                str(guardrails_port),
                "--disable-chat-ui",
            ],
            env={**os.environ, "NEMO_GUARDRAILS_IORAILS_ENGINE": "1"},
        )
        guardrails_url = f"http://127.0.0.1:{guardrails_port}"
        _wait_for_health(f"{guardrails_url}/v1/health", guardrails_process)

        yield Servers(guardrails_url=guardrails_url, config_id=config_id, mock_url=mock_url)
    finally:
        for process in (guardrails_process, mock_process):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
