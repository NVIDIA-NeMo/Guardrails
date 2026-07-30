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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.utils import FakeLLMModel

COLANG_REFUSE = """
define bot refuse tool call
  "I'm sorry, I can't allow that tool call."

define subflow check tool call
  $outcome = execute check_tool_call(check=$check)
  if $outcome["is_blocked"]
    bot refuse tool call
    stop
"""


def _sql_tool_call() -> ToolCall:
    return ToolCall(
        id="call_sql",
        type="function",
        function=ToolCallFunction(name="run_sql", arguments={"query": "SELECT * FROM users"}),
    )


def _other_tool_call() -> ToolCall:
    return ToolCall(
        id="call_other",
        type="function",
        function=ToolCallFunction(name="list_tables", arguments={}),
    )


def _make_rails(yaml_config: str, colang: str, llm_responses: list) -> LLMRails:
    config = RailsConfig.from_content(colang, yaml_config)
    fake_llm = FakeLLMModel(llm_responses=llm_responses)
    return LLMRails(config, llm=fake_llm)


@pytest.mark.asyncio
async def test_per_tool_output_matching_tool_is_blocked():
    """A per_tool rail for run_sql blocks when the LLM returns BLOCK."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    per_tool:
      run_sql:
        - check tool call $check=check_sql
prompts:
  - task: check_sql
    content: |
      Evaluate this tool call: {{ tool_call }}
      BLOCK
"""
    rails = _make_rails(
        yaml_config,
        COLANG_REFUSE,
        [LLMResponse(content="", tool_calls=[_sql_tool_call()]), LLMResponse(content="BLOCK")],
    )
    result = await rails.generate_async(messages=[{"role": "user", "content": "run a query"}])
    assert "can't allow" in result["content"]


@pytest.mark.asyncio
async def test_per_tool_output_non_matching_tool_passes():
    """A per_tool rail scoped to run_sql does not fire for list_tables."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    per_tool:
      run_sql:
        - check tool call $check=check_sql
prompts:
  - task: check_sql
    content: |
      Evaluate: {{ tool_call }}
      BLOCK
"""
    rails = _make_rails(
        yaml_config,
        COLANG_REFUSE,
        [LLMResponse(content="", tool_calls=[_other_tool_call()])],
    )
    result = await rails.generate_async(messages=[{"role": "user", "content": "list tables"}])
    assert result.get("tool_calls") is not None
    assert result["tool_calls"][0]["function"]["name"] == "list_tables"


@pytest.mark.asyncio
async def test_per_tool_output_multiple_checks_first_blocks():
    """With two checks, the first BLOCK stops evaluation without calling the second."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    per_tool:
      run_sql:
        - check tool call $check=check_sql_1
        - check tool call $check=check_sql_2
prompts:
  - task: check_sql_1
    content: |
      Check 1: {{ tool_call }}
      BLOCK
  - task: check_sql_2
    content: |
      Check 2: {{ tool_call }}
      ALLOW
"""
    rails = _make_rails(
        yaml_config,
        COLANG_REFUSE,
        [LLMResponse(content="", tool_calls=[_sql_tool_call()]), LLMResponse(content="BLOCK")],
    )
    result = await rails.generate_async(messages=[{"role": "user", "content": "run query"}])
    assert "can't allow" in result["content"]
    assert rails.llm.inference_count == 2


@pytest.mark.asyncio
async def test_per_tool_output_global_flows_still_run():
    """Global flows run even when per_tool rails are also configured."""
    call_log = []

    from nemoguardrails.actions import action

    @action(is_system_action=True)
    async def audit_all_calls(context=None, **kwargs):
        call_log.append("audit")
        return True

    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    flows:
      - audit all calls
    per_tool:
      run_sql:
        - check tool call $check=check_sql
prompts:
  - task: check_sql
    content: |
      {{ tool_call }}
      ALLOW
"""
    colang = (
        COLANG_REFUSE
        + """
define subflow audit all calls
  $ok = execute audit_all_calls
"""
    )
    rails = _make_rails(
        yaml_config,
        colang,
        [LLMResponse(content="", tool_calls=[_sql_tool_call()]), LLMResponse(content="ALLOW")],
    )
    rails.register_action(audit_all_calls, name="audit_all_calls")
    await rails.generate_async(messages=[{"role": "user", "content": "run query"}])
    assert "audit" in call_log


@pytest.mark.asyncio
async def test_per_tool_output_only_config_no_global_flows():
    """Gate fix: per_tool-only config (no flows list) still fires the rails."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    per_tool:
      run_sql:
        - check tool call $check=check_sql
prompts:
  - task: check_sql
    content: |
      {{ tool_call }}
      BLOCK
"""
    rails = _make_rails(
        yaml_config,
        COLANG_REFUSE,
        [LLMResponse(content="", tool_calls=[_sql_tool_call()]), LLMResponse(content="BLOCK")],
    )
    result = await rails.generate_async(messages=[{"role": "user", "content": "run a query"}])
    assert "can't allow" in result["content"]


@pytest.mark.asyncio
async def test_per_tool_input_matching_tool_is_blocked():
    """A per_tool rail on tool_input blocks a result from run_sql."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_input:
    per_tool:
      run_sql:
        - check tool call $check=check_result
prompts:
  - task: check_result
    content: |
      Evaluate result: {{ tool_message }}
      BLOCK
"""
    colang = (
        COLANG_REFUSE
        + """
define subflow handle tool result
  bot refuse tool call
  stop
"""
    )
    messages = [
        {"role": "user", "content": "run a query"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_sql",
                    "type": "function",
                    "function": {"name": "run_sql", "arguments": '{"query": "SELECT 1"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "sensitive data row",
            "name": "run_sql",
            "tool_call_id": "call_sql",
        },
    ]

    config = RailsConfig.from_content(colang, yaml_config)
    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="BLOCK")])
    rails = LLMRails(config, llm=fake_llm)

    result = await rails.generate_async(messages=messages)
    assert "can't allow" in result["content"]


@pytest.mark.asyncio
async def test_per_tool_input_only_config_gate_fix():
    """Gate fix: per_tool-only tool_input config emits UserToolMessages."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_input:
    per_tool:
      run_sql:
        - check tool call $check=check_result
prompts:
  - task: check_result
    content: |
      {{ tool_message }}
      BLOCK
"""
    messages = [
        {"role": "user", "content": "run a query"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_sql",
                    "type": "function",
                    "function": {"name": "run_sql", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "result data",
            "name": "run_sql",
            "tool_call_id": "call_sql",
        },
    ]

    config = RailsConfig.from_content(COLANG_REFUSE, yaml_config)
    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="BLOCK")])
    rails = LLMRails(config, llm=fake_llm)

    result = await rails.generate_async(messages=messages)
    assert "can't allow" in result["content"]


@pytest.mark.asyncio
async def test_parallel_tool_output_rails_any_block():
    """With parallel=true, all global flows run concurrently; a BLOCK from any stops the turn."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    flows:
      - check tool call $check=check_a
      - check tool call $check=check_b
    parallel: true
prompts:
  - task: check_a
    content: |
      Check A: {{ tool_call }}
      ALLOW
  - task: check_b
    content: |
      Check B: {{ tool_call }}
      BLOCK
"""
    rails = _make_rails(
        yaml_config,
        COLANG_REFUSE,
        [
            LLMResponse(content="", tool_calls=[_sql_tool_call()]),
            LLMResponse(content="ALLOW"),
            LLMResponse(content="BLOCK"),
        ],
    )
    result = await rails.generate_async(messages=[{"role": "user", "content": "run query"}])
    assert "can't allow" in result["content"]


@pytest.mark.asyncio
async def test_parallel_tool_output_rails_all_allow():
    """With parallel=true and all flows allowing, the tool call passes through."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_output:
    flows:
      - check tool call $check=check_a
      - check tool call $check=check_b
    parallel: true
prompts:
  - task: check_a
    content: |
      Check A: {{ tool_call }}
      ALLOW
  - task: check_b
    content: |
      Check B: {{ tool_call }}
      ALLOW
"""
    rails = _make_rails(
        yaml_config,
        COLANG_REFUSE,
        [
            LLMResponse(content="", tool_calls=[_sql_tool_call()]),
            LLMResponse(content="ALLOW"),
            LLMResponse(content="ALLOW"),
        ],
    )
    result = await rails.generate_async(messages=[{"role": "user", "content": "run query"}])
    assert result.get("tool_calls") is not None
    assert result["tool_calls"][0]["function"]["name"] == "run_sql"


@pytest.mark.asyncio
async def test_parallel_tool_input_rails_any_block():
    """With parallel=true on tool_input, a BLOCK from any flow stops the turn."""
    yaml_config = """
models: []
passthrough: true
rails:
  tool_input:
    flows:
      - check tool call $check=check_a
      - check tool call $check=check_b
    parallel: true
prompts:
  - task: check_a
    content: |
      Check A: {{ tool_message }}
      ALLOW
  - task: check_b
    content: |
      Check B: {{ tool_message }}
      BLOCK
"""
    messages = [
        {"role": "user", "content": "run a query"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_sql",
                    "type": "function",
                    "function": {"name": "run_sql", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "result data",
            "name": "run_sql",
            "tool_call_id": "call_sql",
        },
    ]
    config = RailsConfig.from_content(COLANG_REFUSE, yaml_config)
    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="ALLOW"), LLMResponse(content="BLOCK")])
    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=messages)
    assert "can't allow" in result["content"]


def test_per_tool_load_time_validation_output():
    """A mistyped flow name in tool_output.per_tool raises at load time."""
    from nemoguardrails.exceptions import InvalidRailsConfigurationError

    yaml_config = """
models: []
rails:
  tool_output:
    per_tool:
      run_sql:
        - nonexistent flow
"""
    with pytest.raises(InvalidRailsConfigurationError, match="nonexistent flow"):
        config = RailsConfig.from_content(COLANG_REFUSE, yaml_config)
        LLMRails(config)


def test_per_tool_load_time_validation_input():
    """A mistyped flow name in tool_input.per_tool raises at load time."""
    from nemoguardrails.exceptions import InvalidRailsConfigurationError

    yaml_config = """
models: []
rails:
  tool_input:
    per_tool:
      run_sql:
        - another missing flow
"""
    with pytest.raises(InvalidRailsConfigurationError, match="another missing flow"):
        config = RailsConfig.from_content(COLANG_REFUSE, yaml_config)
        LLMRails(config)
