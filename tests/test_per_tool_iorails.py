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

"""Tests for per-tool LLM-based rails on IORails (PerToolCheckAction + RailsManager wiring)."""

from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails import Guardrails
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.guardrails.actions.per_tool_check_action import _parse_verdict
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.guardrails.tool_schema import Tool, Toolset
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction

MESSAGES = [{"role": "user", "content": "Run a SQL query"}]

_BASE_CONFIG = {
    "models": [{"type": "main", "engine": "openai", "model": "gpt-4o"}],
}

_PROMPT_TASK = "blocked_tables"
_PROMPT_YAML = f"""
prompts:
  - task: {_PROMPT_TASK}
    content: |
      Block if SQL reads from users or payments.
      Tool call: {{{{ tool_call }}}}
      Respond with ALLOW or BLOCK on the last line.
"""


def _make_config(per_tool_output=None, per_tool_input=None, flows=None):
    rails: dict = {}
    if per_tool_output is not None or flows is not None:
        rails["tool_output"] = {}
        if per_tool_output is not None:
            rails["tool_output"]["per_tool"] = per_tool_output
        if flows is not None:
            rails["tool_output"]["flows"] = flows
    if per_tool_input is not None:
        rails["tool_input"] = {"per_tool": per_tool_input}
    cfg = {**_BASE_CONFIG}
    if rails:
        cfg["rails"] = rails
    return RailsConfig.from_content(yaml_content=_PROMPT_YAML, config=cfg)


def _make_manager(config: RailsConfig) -> RailsManager:
    registry = EngineRegistry(config.models, config.rails.config)
    return RailsManager(
        engine_registry=registry,
        task_manager=LLMTaskManager(config),
        input_flows=config.rails.input.flows,
        output_flows=config.rails.output.flows,
        tool_call_flows=config.rails.tool_output.flows,
        tool_result_flows=config.rails.tool_input.flows,
        per_tool_output=dict(config.rails.tool_output.per_tool),
        per_tool_input=dict(config.rails.tool_input.per_tool),
    )


def _sql_call(query: str = "SELECT * FROM users") -> ToolCall:
    return ToolCall(
        id="call_1",
        type="function",
        function=ToolCallFunction(name="run_sql", arguments={"query": query}),
    )


def _other_call() -> ToolCall:
    return ToolCall(
        id="call_2",
        type="function",
        function=ToolCallFunction(name="list_tables", arguments={}),
    )


@pytest.mark.asyncio
async def test_per_tool_output_matching_tool_blocked():
    """Matching tool triggers the check and BLOCK verdict blocks the call."""
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert not result.is_safe
    assert result.reason is not None
    assert "block" in result.reason.lower()


@pytest.mark.asyncio
async def test_per_tool_output_matching_tool_allowed():
    """Matching tool with ALLOW verdict passes through."""
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ALLOW"))

    result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert result.is_safe


@pytest.mark.asyncio
async def test_per_tool_output_non_matching_tool_passes():
    """Tool not in per_tool skips the check and passes."""
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    result = await mgr.are_tool_calls_safe([_other_call()], {}, messages=MESSAGES)

    assert result.is_safe
    mgr.engine_registry.model_call.assert_not_called()


@pytest.mark.asyncio
async def test_per_tool_output_multiple_checks_first_blocks():
    """First BLOCK in a two-check sequence stops evaluation."""
    config = _make_config(
        per_tool_output={
            "run_sql": [
                f"check tool call $check={_PROMPT_TASK}",
                f"check tool call $check={_PROMPT_TASK}",
            ]
        }
    )
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert not result.is_safe
    mgr.engine_registry.model_call.assert_called_once()


@pytest.mark.asyncio
async def test_per_tool_output_only_config_no_global_flows():
    """per_tool-only config (no global flows) works correctly."""
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    assert mgr.tool_call_flows == []
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert not result.is_safe


@pytest.mark.asyncio
async def test_per_tool_output_global_flows_still_run():
    """Global tool_call_validation flow runs before per-tool check."""
    config = _make_config(
        flows=["tool call validation"],
        per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]},
    )
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ALLOW"))

    with patch.object(
        mgr.engine_registry,
        "parse_tools",
        return_value=Toolset(
            [Tool(name="run_sql", arguments_schema={"type": "object", "properties": {"query": {"type": "string"}}})]
        ),
    ):
        result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert result.is_safe
    assert len(result.records) == 2


@pytest.mark.asyncio
async def test_per_tool_input_matching_tool_blocked():
    """per_tool_input check blocks a matching tool result."""
    config = _make_config(per_tool_input={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    messages = [
        {"role": "user", "content": "Run a query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "run_sql", "content": "sensitive data"},
    ]

    result = await mgr.are_tool_results_safe(messages)

    assert not result.is_safe


@pytest.mark.asyncio
async def test_per_tool_input_resolves_missing_name_from_prior_call():
    config = _make_config(per_tool_input={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))
    messages = [
        {"role": "user", "content": "Run a query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sensitive data"},
    ]

    result = await mgr.are_tool_results_safe(messages)

    assert not result.is_safe
    mgr.engine_registry.model_call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["output", "input"])
async def test_disabled_boundary_skips_per_tool_checks(boundary):
    flow = f"check tool call $check={_PROMPT_TASK}"
    config = _make_config(
        per_tool_output={"run_sql": [flow]} if boundary == "output" else None,
        per_tool_input={"run_sql": [flow]} if boundary == "input" else None,
    )
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    if boundary == "output":
        result = await mgr.are_tool_calls_safe([_sql_call()], {}, enabled=False, messages=MESSAGES)
    else:
        result = await mgr.are_tool_results_safe(
            [
                {"role": "user", "content": "Run a query"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "run_sql", "content": "sensitive data"},
            ],
            enabled=False,
        )

    assert result.is_safe
    mgr.engine_registry.model_call.assert_not_called()


@pytest.mark.asyncio
async def test_enabled_list_filters_per_tool_checks():
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="BLOCK"))

    result = await mgr.are_tool_calls_safe(
        [_sql_call()],
        {},
        enabled=["tool call validation"],
        messages=MESSAGES,
    )

    assert result.is_safe
    mgr.engine_registry.model_call.assert_not_called()


def test_invalid_per_tool_action_raises_configuration_error():
    config = _make_config(per_tool_output={"run_sql": ["typo tool check"]})

    with pytest.raises(InvalidRailsConfigurationError, match="typo tool check"):
        Guardrails(config)


def test_structural_validator_in_per_tool_raises_configuration_error():
    config = _make_config(per_tool_output={"run_sql": ["tool call validation"]})

    with pytest.raises(InvalidRailsConfigurationError, match="tool call validation"):
        IORails.unsupported_reason(config)


def test_global_check_action_routes_away_from_iorails():
    config = _make_config(flows=[f"check tool call $check={_PROMPT_TASK}"])

    assert IORails.unsupported_reason(config) is not None


def test_parse_verdict_requires_verdict_on_final_non_empty_line():
    assert _parse_verdict("Reasoning\nALLOW")
    assert not _parse_verdict("ALLOW\nUncertain")


@pytest.mark.asyncio
async def test_per_tool_provider_http_error_propagates():
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(
        side_effect=ModelEngineError("unauthorized", model_name="gpt-4o", status=401)
    )

    with pytest.raises(ModelEngineError, match="unauthorized"):
        await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)


@pytest.mark.asyncio
async def test_per_tool_missing_check_param():
    """Flow string without $check= fails closed without calling the model."""
    config = _make_config(per_tool_output={"run_sql": ["check tool call"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ALLOW"))

    result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert not result.is_safe
    mgr.engine_registry.model_call.assert_not_called()


@pytest.mark.asyncio
async def test_per_tool_unknown_prompt_task():
    """Missing prompt task fails closed without crashing."""
    config = _make_config(per_tool_output={"run_sql": ["check tool call $check=nonexistent_task"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="ALLOW"))

    with patch.object(
        mgr._per_tool_check_actions["check tool call $check=nonexistent_task"].task_manager,
        "render_task_prompt",
        side_effect=Exception("task not found"),
    ):
        result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert not result.is_safe
    mgr.engine_registry.model_call.assert_not_called()


@pytest.mark.asyncio
async def test_per_tool_model_call_failure():
    """Model call failure fails closed."""
    config = _make_config(per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK}"]})
    mgr = _make_manager(config)
    mgr.engine_registry.model_call = AsyncMock(side_effect=RuntimeError("connection timeout"))

    result = await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert not result.is_safe
    assert result.reason is not None


@pytest.mark.asyncio
async def test_per_tool_model_override():
    """$model= override is passed to the model call instead of 'main'."""
    config = _make_config(
        per_tool_output={"run_sql": [f"check tool call $check={_PROMPT_TASK} $model=safety_classifier"]}
    )
    mgr = _make_manager(config)

    captured_model_types: list[str] = []

    async def _capture(model_type, messages, **kwargs):
        captured_model_types.append(model_type)
        return LLMResponse(content="ALLOW")

    with patch.object(mgr.engine_registry, "model_call", side_effect=_capture):
        await mgr.are_tool_calls_safe([_sql_call()], {}, messages=MESSAGES)

    assert captured_model_types == ["safety_classifier"]
