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

"""Fixtures that run Claude Code and Codex against a hooks server, each agent from its package in `agent_hooks`."""

# The agents come from the `agent_hooks` dependency group, outside `dev`, so the tests skip unless it is installed:
#
#   uv sync --group dev --group agent_hooks
#
# Each agent runs with its own home directory and only PATH from this process's environment. Managed settings still
# apply to Claude Code: an `allowedHttpHookUrls` that doesn't allow localhost would stop its hook from running.
#
# The hooks' URLs say `localhost`, as the docs' do, while every server here listens on 127.0.0.1 only, as the hooks
# server does by default; so an agent that didn't fall back from ::1 would fail these tests.

import socket
import sys
import threading
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest
import uvicorn

from nemoguardrails import RailsConfig
from nemoguardrails.server.hooks.app import create_hooks_app
from tests.server.hooks_agents.agents import Agent, ClaudeCode, Codex

# Engine startup runs before the server answers, so its first health check can wait in the socket's backlog.
SERVER_START_TIMEOUT_SECONDS = 60
# How long a stopping server may wait for open connections, and then for its thread.
SERVER_STOP_TIMEOUT_SECONDS = 5

SKIP_REASON = "the agent_hooks dependency group isn't installed: uv sync --group dev --group agent_hooks"


@pytest.fixture(params=[ClaudeCode.name, Codex.name])
def agent(request: pytest.FixtureRequest, tmp_path: Path) -> Agent:
    """Each agent, from its package, working in the test's own directory."""
    if sys.platform == "win32":
        pytest.skip("the docs' hooks and the scripted command need a POSIX shell")
    if request.param == ClaudeCode.name:
        claude_agent_sdk = pytest.importorskip("claude_agent_sdk", reason=SKIP_REASON)
        # The SDK runs this bundled binary itself; the path is the SDK's, not public.
        return ClaudeCode(Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude", tmp_path)
    codex_cli_bin = pytest.importorskip("codex_cli_bin", reason=SKIP_REASON)
    return Codex(codex_cli_bin.bundled_codex_path(), tmp_path)


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
    try:
        with urllib.request.urlopen(f"{url}/v1/health", timeout=SERVER_START_TIMEOUT_SECONDS):
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
