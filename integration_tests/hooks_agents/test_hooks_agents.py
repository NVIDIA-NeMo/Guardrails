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

"""Claude Code and Codex obey the hooks server's answers through the hooks their docs configure."""

from pathlib import Path

import pytest

from integration_tests.hooks_agents.agents import TOOL_RAN_FILE, Agent
from nemoguardrails import RailsConfig

pytestmark = pytest.mark.slow

EXAMPLE_CONFIGS = Path(__file__).resolve().parents[2] / "examples" / "configs" / "agent_hooks"

ALLOWED_COMMAND = f"echo hello > {TOOL_RAN_FILE}"
# The example configs' Bash rail matches `rm -rf`, which only echoing it keeps harmless, if the agent runs it anyway.
DENIED_COMMAND = f"echo 'rm -rf build' > {TOOL_RAN_FILE}"
DENY_REASON = "NeMo Guardrails: regex check tool call"

# Every Bash call fails its rail: the rail reads argument `x`, which the declaration the policy builds doesn't have.
RAIL_FAILURE_CONFIG = """
models: []
hooks_server:
  allowed_tools:
    - name: Bash
      arguments: [command, description]
rails:
  tool_call:
    per_tool:
      Bash:
        - regex check tool call $argument=x
"""


def _example_config(agent: Agent) -> RailsConfig:
    """The config the docs give for *agent*, from examples/configs/agent_hooks."""
    return RailsConfig.from_path(str(EXAMPLE_CONFIGS / agent.example))


def test_allowed_call_runs(agent, hooks_server):
    """A call the agent's example config allows runs."""
    run = agent.run(ALLOWED_COMMAND, hooks_server(_example_config(agent)) + agent.hook_path)

    assert run.tool_ran, run


def test_denied_call_never_runs_and_the_model_gets_the_reason(agent, hooks_server):
    """A call the agent's example config denies never runs, and the agent tells its model the server's reason."""
    run = agent.run(DENIED_COMMAND, hooks_server(_example_config(agent)) + agent.hook_path)

    assert not run.tool_ran, run
    assert agent.blocked_by_hook in run.tool_result, run
    assert DENY_REASON in run.tool_result, run


def test_allowed_call_on_the_other_agents_path_runs(agent, hooks_server):
    """A hook pointed at the other agent's path still lets an allowed call run, since both agents read the answer."""
    run = agent.run(ALLOWED_COMMAND, hooks_server(_example_config(agent)) + agent.other_hook_path)

    assert run.tool_ran, run


def test_denied_call_on_the_other_agents_path_never_runs(agent, hooks_server):
    """A hook pointed at the other agent's path still stops a denied call, with the server's reason, not an error."""
    run = agent.run(DENIED_COMMAND, hooks_server(_example_config(agent)) + agent.other_hook_path)

    assert not run.tool_ran, run
    assert agent.blocked_by_hook in run.tool_result, run
    assert DENY_REASON in run.tool_result, run


def test_call_never_runs_when_its_rail_fails(agent, hooks_server, caplog):
    """A call whose rail fails never runs: the server answers `500 rail_failed`, and the agent's hook blocks it."""
    url = hooks_server(RailsConfig.from_content(yaml_content=RAIL_FAILURE_CONFIG))
    run = agent.run(ALLOWED_COMMAND, url + agent.hook_path)

    assert not run.tool_ran, run
    assert agent.blocked_by_hook in run.tool_result and "500" in run.tool_result, run
    assert any(
        f"{agent.hook_path}:" in record.getMessage() and "code=rail_failed" in record.getMessage()
        for record in caplog.records
    ), run


def test_call_never_runs_when_the_hooks_server_is_down(agent, refused_url):
    """A call never runs when the agent's hook can't connect to the hooks server."""
    run = agent.run(ALLOWED_COMMAND, refused_url + agent.hook_path)

    assert not run.tool_ran, run
    assert agent.blocked_by_hook in run.tool_result, run


def test_call_never_runs_when_the_hooks_server_never_answers(agent, silent_url):
    """A call never runs when the hooks server doesn't answer within the hook's timeout."""
    run = agent.run(ALLOWED_COMMAND, silent_url + agent.hook_path, hook_timeout=1)

    assert not run.tool_ran, run
    assert agent.blocked_by_hook in run.tool_result and "timed out" in run.tool_result, run
