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

"""Claude Code and Codex, from their published Python wheels, run headless with the hook their docs configure."""

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pytest

from integration_tests.hooks_agents.fake_model import FakeModel, ScriptedToolCall

# The scripted shell command writes this file in the agent's working directory, so a test can tell whether it ran.
TOOL_RAN_FILE = "tool-ran.txt"

# A deadline for one agent run, so a hung agent fails the test instead of stalling the suite.
AGENT_TIMEOUT_SECONDS = 120

# The fake model ignores the prompt: it calls the scripted tool whatever the prompt says.
PROMPT = "Run the scripted command."

CLAUDE_CODE_HOOK_PATH = "/v1/hooks/claude-code"
CODEX_HOOK_PATH = "/v1/hooks/codex"


@dataclass(frozen=True)
class AgentRun:
    """One agent run: whether the scripted tool ran, and the tool result the agent sent its model."""

    tool_ran: bool
    tool_result: str
    process: subprocess.CompletedProcess[str]

    def __str__(self) -> str:
        """The run's outcome and the agent's output, for an assertion message."""
        return (
            f"tool_ran={self.tool_ran} tool_result={self.tool_result!r} exit={self.process.returncode}\n"
            f"stdout:\n{self.process.stdout[-4000:]}\nstderr:\n{self.process.stderr[-4000:]}"
        )


class Agent(ABC):
    """An agent run headless in its own directory, against the fake model, with a `PreToolUse` hook on the server."""

    name: str
    hook_path: str
    other_hook_path: str
    # The directory of the agent's example config under examples/configs/agent_hooks.
    example: str
    # What the agent's tool result to its model says when its hook stopped the call, a deny or a failure alike.
    blocked_by_hook: str

    def __init__(self, binary: Path, work_dir: Path):
        if not binary.is_file():
            pytest.fail(f"The {self.name} binary isn't at {binary}; did its package move it?")
        self.binary = binary
        self.work_dir = work_dir

    def run(self, command: str, hook_url: str, hook_timeout: int = 25) -> AgentRun:
        """Run the agent once, its model calling the shell with *command*, and its hook waiting *hook_timeout* seconds."""
        cwd = self.work_dir / "cwd"
        cwd.mkdir()
        with FakeModel(self.tool_call(command)) as model:
            process = self.start(cwd, model.url, hook_url, hook_timeout)
            results = model.tool_results()
        run = AgentRun((cwd / TOOL_RAN_FILE).exists(), results[0] if results else "", process)
        # Every case needs the agent to have made the call and reported back: otherwise a "never runs" passes vacuously.
        if len(results) != 1:
            pytest.fail(f"{self.name} sent its model {len(results)} tool results, not 1:\n{run}")
        return run

    @abstractmethod
    def tool_call(self, command: str) -> ScriptedToolCall:
        """The model's shell call running *command*, named as the agent offers its shell tool."""

    @abstractmethod
    def start(self, cwd: Path, model_url: str, hook_url: str, hook_timeout: int) -> subprocess.CompletedProcess[str]:
        """Run the agent to completion in *cwd*, its model at *model_url* and its `PreToolUse` hook at *hook_url*."""

    def _environment(self, **variables: str) -> dict[str, str]:
        """The agent's whole environment: its own home and temporary directories, and only PATH from this process."""
        # Nothing else is inherited, so no provider key, agent setting or user config reaches the agent.
        home, tmp = self.work_dir / "home", self.work_dir / "tmp"
        home.mkdir(exist_ok=True)
        tmp.mkdir(exist_ok=True)
        return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home), "TMPDIR": str(tmp), **variables}

    def _run(self, args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        """Run the agent's binary with *args* until it exits, failing the test if it runs past the deadline."""
        try:
            return subprocess.run(
                [str(self.binary), *args],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=AGENT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as timeout:
            pytest.fail(
                f"{self.name} didn't exit within {AGENT_TIMEOUT_SECONDS} seconds.\n"
                f"stdout:\n{_tail(timeout.stdout)}\nstderr:\n{_tail(timeout.stderr)}"
            )


class ClaudeCode(Agent):
    """Claude Code in print mode, with the `http` hook from the hooks server's Claude Code docs."""

    name = "claude-code"
    hook_path = CLAUDE_CODE_HOOK_PATH
    other_hook_path = CODEX_HOOK_PATH
    example = "claude_code"
    blocked_by_hook = "PreToolUse:Bash hook error"

    def tool_call(self, command: str) -> ScriptedToolCall:
        return ScriptedToolCall("Bash", {"command": command, "description": "Run the scripted command"})

    def start(self, cwd: Path, model_url: str, hook_url: str, hook_timeout: int) -> subprocess.CompletedProcess[str]:
        env = self._environment(
            ANTHROPIC_BASE_URL=model_url,
            ANTHROPIC_API_KEY="fake-model-key",
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
            DISABLE_AUTOUPDATER="1",
        )
        # Marks onboarding as done, which a headless run can't answer.
        (Path(env["HOME"]) / ".claude.json").write_text(json.dumps({"hasCompletedOnboarding": True}))
        hook = {"type": "http", "url": hook_url, "timeout": hook_timeout, "onFailure": "block"}
        settings = self.work_dir / "settings.json"
        settings.write_text(json.dumps({"hooks": {"PreToolUse": [{"hooks": [hook]}]}}))
        # A call the hook allows still needs Claude Code's own permission, which nobody can give in print mode.
        return self._run(
            ["-p", PROMPT, "--settings", str(settings), "--allowedTools", "Bash(echo:*)"],
            cwd,
            env,
        )


class Codex(Agent):
    """`codex exec`, with the `curl` command hook from the hooks server's Codex docs."""

    name = "codex"
    hook_path = CODEX_HOOK_PATH
    other_hook_path = CLAUDE_CODE_HOOK_PATH
    example = "codex"
    blocked_by_hook = "Command blocked by PreToolUse hook"

    def tool_call(self, command: str) -> ScriptedToolCall:
        # Codex's default model calls tools from code-mode cells. The cell's own `exec` call skips the hook; its
        # `exec_command` call reaches it as `Bash`, with `cmd` as `command`.
        cell = f"const output = await tools.exec_command({{ cmd: {json.dumps(command)} }});\ntext(output.output);\n"
        return ScriptedToolCall("exec", cell)

    def start(self, cwd: Path, model_url: str, hook_url: str, hook_timeout: int) -> subprocess.CompletedProcess[str]:
        codex_home = self.work_dir / "codex_home"
        codex_home.mkdir()
        (codex_home / "config.toml").write_text(_codex_config(model_url, hook_url, hook_timeout))
        env = self._environment(CODEX_HOME=str(codex_home), CODEX_API_KEY="fake-model-key")
        # The bypass is the docs' flag for automated tests: without it, Codex skips a hook nobody has reviewed.
        return self._run(
            [
                "exec",
                "--skip-git-repo-check",
                "--dangerously-bypass-hook-trust",
                "--sandbox",
                "workspace-write",
                PROMPT,
            ],
            cwd,
            env,
        )


def _codex_config(model_url: str, hook_url: str, hook_timeout: int) -> str:
    """A Codex `config.toml` for the fake model, with the docs' hook: `curl` gives up 5 seconds before Codex would."""
    hook_command = (
        f"curl -sSf --max-time {hook_timeout} --data-binary @- -H 'Content-Type: application/json' {hook_url} || exit 2"
    )
    # A JSON string is a valid TOML basic string.
    return f"""
model_provider = "fake_model"

# A provider of its own: the built-in OpenAI provider tries the Responses API over a WebSocket first.
[model_providers.fake_model]
name = "Fake model"
base_url = {json.dumps(f"{model_url}/v1")}
wire_api = "responses"
env_key = "CODEX_API_KEY"

[[hooks.PreToolUse]]

[[hooks.PreToolUse.hooks]]
type = "command"
command = {json.dumps(hook_command)}
timeout = {hook_timeout + 5}
"""


def _tail(output: Optional[Union[str, bytes]]) -> str:
    """The end of an agent's output captured before a timeout, which Python may hand over as bytes."""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return (output or "")[-4000:]
