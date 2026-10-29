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

"""Fixtures that run Claude Code and Codex against a hooks server, each agent from its published wheel."""

# Neither agent is a Guardrails dependency: `make test-hooks-agents` installs both wheels, pinned with hashes in
# requirements.txt, into a directory of their own (AGENTS_DIR), then runs this suite in .venv. The suite sits outside
# pytest.ini's testpaths, so `make test` doesn't collect it.
#
# Each agent runs with its own home directory and only PATH from this process's environment. Managed settings still
# apply to Claude Code: an `allowedHttpHookUrls` that doesn't allow localhost would stop its hook from running.
#
# The hooks' URLs say `localhost`, as the docs' do, while every server here listens on 127.0.0.1 only, as the hooks
# server does by default; so an agent that didn't fall back from ::1 would fail these tests.

import os
import socket
import sys
import threading
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest
import uvicorn

from integration_tests.hooks_agents.agents import Agent, ClaudeCode, Codex
from nemoguardrails import RailsConfig
from nemoguardrails.server.hooks.app import create_hooks_app

REPO_ROOT = Path(__file__).resolve().parents[2]
# Where `make test-hooks-agents` installs the agents' wheels: HOOKS_AGENTS_DIR when it's set, against the repository root.
AGENTS_DIR = REPO_ROOT / (os.environ.get("HOOKS_AGENTS_DIR") or ".cache/hooks_agents")

# Engine startup runs before the server answers, so its first health check can wait in the socket's backlog.
SERVER_START_TIMEOUT_SECONDS = 60
# How long a stopping server may wait for open connections, and then for its thread.
SERVER_STOP_TIMEOUT_SECONDS = 5


@pytest.fixture(params=[ClaudeCode.name, Codex.name])
def agent(request: pytest.FixtureRequest, tmp_path: Path) -> Agent:
    """Each agent, from its wheel in AGENTS_DIR, working in the test's own directory."""
    if sys.platform == "win32":
        pytest.skip("the docs' hooks and the scripted command need a POSIX shell")
    if not AGENTS_DIR.is_dir():
        pytest.fail(f"{AGENTS_DIR} doesn't exist. From the repository root, run: make test-hooks-agents")
    # Both paths are the packages' own, not public: the SDK runs its bundled binary itself, and Codex's path is the one
    # its package's bundled_codex_path() returns. Neither package is on this process's import path.
    if request.param == ClaudeCode.name:
        return ClaudeCode(AGENTS_DIR / "claude_agent_sdk" / "_bundled" / "claude", tmp_path)
    return Codex(AGENTS_DIR / "codex_cli_bin" / "bin" / "codex", tmp_path)


@pytest.fixture
def hooks_server() -> Iterator[Callable[[RailsConfig], str]]:
    """Serves a config with the hooks server, in this process, and returns its URL; stops each one at teardown."""
    with ExitStack() as servers:
        yield lambda config: servers.enter_context(_serving(config))


@contextmanager
def _serving(config: RailsConfig) -> Iterator[str]:
    """Serves *config* on a loopback port until the block exits, and yields the URL the docs' hooks use."""
    app = create_hooks_app(config)
    # Bound before the server starts, so no other process can take the port, and a hook that connects early waits in
    # the backlog until the server answers.
    listener = socket.create_server(("127.0.0.1", 0))
    url = f"http://localhost:{listener.getsockname()[1]}"
    config = uvicorn.Config(
        app, log_config=None, access_log=False, timeout_graceful_shutdown=SERVER_STOP_TIMEOUT_SECONDS
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=_serve, args=(server, listener), daemon=True)
    thread.start()
    try:
        _check_health(url)
        yield url
    finally:
        server.should_exit = True
        thread.join(2 * SERVER_STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            pytest.fail(f"The hooks server at {url} didn't stop")


def _serve(server: uvicorn.Server, listener: socket.socket) -> None:
    """Runs *server* on *listener*, then closes it, so a server that fails to start resets the connections waiting."""
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()


def _check_health(url: str) -> None:
    """Fails the test unless the hooks server at *url* answers its health check in time."""
    # No proxy handler: urllib otherwise sends even a loopback request to the environment's HTTP_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/v1/health", timeout=SERVER_START_TIMEOUT_SECONDS):
            pass
    except OSError as error:
        pytest.fail(f"The hooks server at {url} didn't pass its health check: {error}")


@pytest.fixture
def refused_url() -> str:
    """The URL of a loopback port nothing listens on, so a connection to it is refused."""
    # Port 1 rather than a freed ephemeral port, which another test's server could be handed before the agent connects.
    return "http://localhost:1"


@pytest.fixture
def silent_url() -> Iterator[str]:
    """The URL of a loopback port that takes connections and never answers them."""
    with socket.create_server(("127.0.0.1", 0)) as listener:
        yield f"http://localhost:{listener.getsockname()[1]}"
