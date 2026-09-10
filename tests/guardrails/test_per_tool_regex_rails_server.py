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

"""Real-process integration tests for per-tool regex rails against the actual server.

Unlike every other server test in this repo (in-process fastapi.testclient.TestClient),
this spawns two real subprocesses over real HTTP -- a `nemoguardrails server` instance
(the genuine, unmodified server binary) and the mock tool LLM server
(tests/mock_tool_llm_server) it points at as its main model's base_url -- and drives them
with real network calls, matching the benchmark/ suite's operational pattern rather than
the tests/server/ TestClient convention. Readiness polling is modeled on
benchmark/locust/run_locust.py's `_check_service`/`_get`, the only existing health-check
polling logic in the repo.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import httpx
import pytest
import yaml

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


class _Servers(NamedTuple):
    guardrails_url: str
    config_id: str


def _write_config(config_dir: Path, config_id: str, mock_port: int) -> None:
    config = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "mock-tool-model",
                "parameters": {"base_url": f"http://127.0.0.1:{mock_port}/v1", "api_key": "unused"},
            }
        ],
        "rails": {
            "config": {
                "regex_detection": {
                    "tool_output": {
                        "run_sql": {"patterns": [r"DROP\s+TABLE"]},
                        "other_tool": {"patterns": [r"SECRET"]},
                    },
                    "tool_input": {"run_sql": {"patterns": [r"ssn:\s*\d{3}-\d{2}-\d{4}"]}},
                }
            },
            "tool_output": {
                "per_tool": {
                    "run_sql": ["regex check tool call"],
                    "other_tool": ["regex check tool call"],
                }
            },
            "tool_input": {"per_tool": {"run_sql": ["regex check tool result"]}},
        },
    }
    (config_dir / config_id).mkdir(parents=True)
    (config_dir / config_id / "config.yml").write_text(yaml.safe_dump(config))


@pytest.fixture(scope="module")
def servers(tmp_path_factory) -> Iterator[_Servers]:
    """Spawn the mock tool LLM server and a real `nemoguardrails server`, both on free ports."""
    mock_port = _free_port()
    guardrails_port = _free_port()
    config_id = "per_tool_regex"
    config_dir = tmp_path_factory.mktemp("per_tool_regex_configs")
    _write_config(config_dir, config_id, mock_port)

    mock_process = subprocess.Popen(
        [sys.executable, "-m", "tests.mock_tool_llm_server.run_server", "--port", str(mock_port)],
        cwd=Path(__file__).resolve().parents[2],
    )
    guardrails_process = None
    try:
        _wait_for_health(f"http://127.0.0.1:{mock_port}/health", mock_process)

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

        yield _Servers(guardrails_url=guardrails_url, config_id=config_id)
    finally:
        for process in (guardrails_process, mock_process):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def _chat(servers: _Servers, content: str, *, tool_result: bool = False) -> dict:
    if tool_result:
        messages = [
            {"role": "user", "content": "run a query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": content},
        ]
    else:
        messages = [{"role": "user", "content": content}]

    response = httpx.post(
        f"{servers.guardrails_url}/v1/chat/completions",
        json={"model": servers.config_id, "messages": messages},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


class TestPerToolCallRegexAgainstRealServer:
    def test_matching_pattern_blocked(self, servers):
        message = _chat(servers, "DROP TABLE users")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_non_matching_pattern_passes(self, servers):
        message = _chat(servers, "SELECT 1")
        assert message["tool_calls"][0]["function"]["name"] == "run_sql"

    def test_multiple_tool_calls_all_safe_pass(self, servers):
        """The fan-out loop evaluates every tool call, not just the first."""
        message = _chat(servers, "SELECT 1||SELECT 2")
        assert len(message["tool_calls"]) == 2
        assert [tc["function"]["arguments"] for tc in message["tool_calls"]] == [
            '{"query": "SELECT 1"}',
            '{"query": "SELECT 2"}',
        ]

    def test_multiple_tool_calls_second_one_blocks(self, servers):
        """A match on any tool call in the batch blocks the whole response, not just that call."""
        message = _chat(servers, "SELECT 1||DROP TABLE users")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_multiple_tool_calls_first_one_blocks(self, servers):
        """Order doesn't matter: a match anywhere in the batch blocks."""
        message = _chat(servers, "DROP TABLE users||SELECT 1")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_second_tools_pattern_does_not_cross_match_first_tools_content(self, servers):
        """run_sql and other_tool each have their own configured patterns (DROP TABLE
        vs SECRET). run_sql's pattern matching other_tool's content must not leak across
        -- other_tool's call only blocks on its own pattern, not run_sql's."""
        message = _chat(servers, "run_sql:SELECT 1||other_tool:DROP TABLE users")
        assert len(message["tool_calls"]) == 2
        names = [tc["function"]["name"] for tc in message["tool_calls"]]
        assert names == ["run_sql", "other_tool"]

    def test_second_tool_blocks_on_its_own_pattern(self, servers):
        """other_tool's own configured pattern (SECRET) still blocks when it matches,
        proving its check genuinely runs rather than being silently skipped."""
        message = _chat(servers, "run_sql:SELECT 1||other_tool:the SECRET is out")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."


class TestPerToolResultRegexAgainstRealServer:
    def test_matching_pattern_blocked(self, servers):
        message = _chat(servers, "ssn: 123-45-6789", tool_result=True)
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_non_matching_pattern_passes(self, servers):
        message = _chat(servers, "no sensitive data", tool_result=True)
        assert message["content"] == "ok"
