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

"""Fixtures that load the built NeMo Relay plugin into Relay's Python plugin host, on Linux or macOS."""

# The suite runs in its own environment, without Guardrails installed (see requirements.in). From the repository root,
# `make test-relay-plugin` syncs .venv and runs it, or directly:
#
#   uv run --no-config --no-project --python 3.11 --with-requirements tests_relay_plugin/requirements.txt \
#     pytest tests_relay_plugin

import asyncio
import json
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from nemo_relay import Json, plugin

REPO_ROOT = Path(__file__).resolve().parents[1]
CRATE_DIR = REPO_ROOT / "crates" / "nemo-guardrails-relay-plugin"
LIBRARY_TARGET = "nemo_guardrails_relay_plugin"
LIBRARY_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
# Relay merges this file over the plugins.toml a host passes, so it could load or reconfigure plugins in the tests.
SYSTEM_PLUGINS_TOML = Path("/etc/nemo-relay/plugins.toml")

VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SERVE_SCRIPT = Path(__file__).with_name("serve.py")
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "configs" / "agent_hooks" / "nemo_relay"
# The server imports Guardrails before it answers, which takes seconds, and longer on a busy machine.
SERVER_START_TIMEOUT_SECONDS = 60
SERVER_STOP_TIMEOUT_SECONDS = 10

Activate = Callable[[dict[str, Json]], plugin.PluginHostActivation]


@pytest.fixture(scope="session")
def plugin_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The manifest of the plugin bundle, built in release and packaged as the release pipeline does."""
    if SYSTEM_PLUGINS_TOML.exists():
        pytest.fail(f"{SYSTEM_PLUGINS_TOML} exists, and Relay would merge it into every test's plugin host")
    bundle = tmp_path_factory.mktemp("bundle")
    package = [sys.executable, str(CRATE_DIR / "scripts" / "package_bundle.py")]
    result = subprocess.run(
        [*package, "--library", str(_build_library()), "--output", str(bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"package_bundle.py failed:\n{result.stderr}")
    return bundle / "relay-plugin.toml"


def _build_library() -> Path:
    """Builds the plugin library in release, and returns the path Cargo reports for it."""
    result = subprocess.run(
        [
            *("cargo", "build", "--release", "--locked", "-p", "nemo-guardrails-relay-plugin"),
            "--message-format=json-render-diagnostics",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"cargo build failed:\n{result.stderr}")
    # Cargo reports where it wrote the library, which accounts for a target set by CARGO_BUILD_TARGET or build.target.
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") != "compiler-artifact" or message["target"]["name"] != LIBRARY_TARGET:
            continue
        for filename in message["filenames"]:
            if filename.endswith(LIBRARY_SUFFIX):
                return Path(filename)
    pytest.fail(f"cargo build didn't report the plugin library:\n{result.stdout}")


@pytest.fixture
def relay_plugin(plugin_manifest: Path, tmp_path: Path) -> Iterator[Activate]:
    """Activates the plugin in Relay's plugin host with a component config, and closes the host at teardown."""
    activations: list[plugin.PluginHostActivation] = []

    def activate(config: dict[str, Json]) -> plugin.PluginHostActivation:
        """A fresh plugin host that loads only the plugin, with *config* as its component config."""
        plugins_toml = _write_plugins_toml(tmp_path / "plugins.toml", plugin_manifest, config)
        activation = asyncio.run(plugin.initialize(plugin.PluginConfig(), plugins_toml))
        activations.append(activation)
        # From Relay 0.10.4 a plugins.toml the host passes replaces the user's; only the system file would join it.
        assert activation.report["config_paths"] == [str(plugins_toml)]
        return activation

    yield activate
    for activation in activations:
        asyncio.run(activation.close())


def _write_plugins_toml(path: Path, manifest: Path, config: dict[str, Json]) -> Path:
    """Writes a plugins.toml that loads the bundle, with a component config of scalar values."""
    # A JSON string, number or boolean is also valid TOML, as Relay's own host tests write them.
    entries = ", ".join(f"{key} = {json.dumps(value)}" for key, value in config.items())
    path.write_text(
        'version = 1\n\n[plugins.policy.defaults]\nstartup = "required"\nattestation = "integrity_only"\n\n'
        f"[[plugins.dynamic]]\nmanifest = {json.dumps(str(manifest))}\nconfig = {{ {entries} }}\n"
    )
    return path


@pytest.fixture(scope="session")
def example_hooks_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """The URL of a hooks server for Guardrails' NeMo Relay example config."""
    with _running_hooks_server(EXAMPLE_CONFIG, tmp_path_factory.mktemp("example_hooks_server")) as url:
        yield url


@contextmanager
def _running_hooks_server(config: Path, log_dir: Path) -> Iterator[str]:
    """Serves *config* from a child process of the repository's .venv, and yields its URL once its health check passes."""
    if not VENV_PYTHON.exists():
        pytest.fail(f"{VENV_PYTHON} doesn't exist. From the repository root, run: uv sync --group dev --extra server")
    log_path = log_dir / "server.log"
    with socket.create_server(("127.0.0.1", 0)) as listener, log_path.open("w") as log:
        host, port = listener.getsockname()
        server = subprocess.Popen(
            [str(VENV_PYTHON), str(SERVE_SCRIPT), "--config", str(config), "--fd", str(listener.fileno())],
            pass_fds=[listener.fileno()],
            env=_server_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    # From here the server holds the only copy of the socket, so the socket closes when the server exits.
    try:
        _check_health(f"http://{host}:{port}", log_path)
        yield f"http://{host}:{port}"
    finally:
        server.terminate()
        try:
            server.wait(SERVER_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def _server_environment() -> dict[str, str]:
    """The server's whole environment: none of the test process's variables, so it has no provider keys."""
    # Unbuffered, so the log holds uvicorn's access lines, which go to stdout, while the server runs.
    return {"NEMO_GUARDRAILS_NO_USAGE_STATS": "1", "PYTHONUNBUFFERED": "1"}


def _check_health(url: str, log_path: Path) -> None:
    """Fails with the server's log unless the server at *url* answers its health check in time."""
    # The socket listened before the server started, so the request waits in its backlog until the server answers,
    # and is reset as soon as a server that fails to start exits.
    try:
        with urllib.request.urlopen(f"{url}/v1/health", timeout=SERVER_START_TIMEOUT_SECONDS):
            pass
    except OSError as error:
        pytest.fail(f"The hooks server didn't pass its health check ({error}). Its log:\n{log_path.read_text()}")


@pytest.fixture
def refused_endpoint() -> str:
    """The URL of a local port that was just free, so it refuses connections unless another process takes it."""
    # A port held by a socket that doesn't listen isn't refused on macOS: the connection times out instead.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    return f"http://{host}:{port}"
