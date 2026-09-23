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

"""Tests for per-tool regex rails: config schema, RailsManager dispatch, IORails wiring.

The regex-based per-tool check is model-free, so unlike test_tool_rails_e2e.py this
module needs no HTTP mocking -- a "main" model entry is only required because
EngineRegistry.parse_tools / extract_tool_exchanges need an engine to parse message
shape, not because any model call happens.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.guardrails.tool_schema import ToolExchange, ToolResult
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.tracing.constants import GuardrailsAttributes
from nemoguardrails.types import ToolCall, ToolCallFunction
from tests.guardrails.tool_helpers import assert_result_blocked

STACK_CONFIG = {"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}

RUN_SQL_PATTERN_CONFIG = {
    "tool_output": {"run_sql": {"patterns": [r"DROP\s+TABLE"]}},
}
RUN_SQL_RESULT_PATTERN_CONFIG = {
    "tool_input": {"run_sql": {"patterns": [r"ssn:\s*\d{3}-\d{2}-\d{4}"]}},
}


def _sql_call(query: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, type="function", function=ToolCallFunction(name="run_sql", arguments={"query": query}))


def _other_call(call_id: str = "call_2") -> ToolCall:
    return ToolCall(
        id=call_id, type="function", function=ToolCallFunction(name="list_tables", arguments={"query": "DROP TABLE x"})
    )


def _llm_params(*tool_names: str) -> dict:
    """Declare *tool_names* with a schema permissive enough to accept any arguments.

    @tool_output_validation blocks a call whose tool isn't declared here, so per-tool
    tests that expect the regex check itself to run (not the schema gate) need their
    tool declared.
    """
    return {
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object", "additionalProperties": True}},
            }
            for name in tool_names
        ]
    }


def _build_manager(
    *,
    per_tool_call_flows=None,
    per_tool_result_flows=None,
    tool_call_flows=None,
    tool_result_flows=None,
    regex_detection=None,
) -> RailsManager:
    config_dict = dict(STACK_CONFIG)
    if regex_detection is not None:
        config_dict = {**config_dict, "rails": {"config": {"regex_detection": regex_detection}}}
    config = RailsConfig.from_content(config=config_dict)
    engine_registry = EngineRegistry(config.models)
    return RailsManager(
        engine_registry=engine_registry,
        task_manager=LLMTaskManager(config),
        input_flows=[],
        output_flows=[],
        tool_call_flows=tool_call_flows or [],
        tool_result_flows=tool_result_flows or [],
        per_tool_call_flows=per_tool_call_flows or {},
        per_tool_result_flows=per_tool_result_flows or {},
    )


class TestConfigSchema:
    def test_per_tool_field_defaults_to_empty(self):
        config = RailsConfig.from_content(config=STACK_CONFIG)
        assert config.rails.tool_output.per_tool == {}
        assert config.rails.tool_input.per_tool == {}

    def test_per_tool_field_round_trips_from_yaml(self):
        config = RailsConfig.from_content(
            config={
                **STACK_CONFIG,
                "rails": {
                    "config": {"regex_detection": RUN_SQL_PATTERN_CONFIG},
                    "tool_output": {"per_tool": {"run_sql": ["regex check tool output"]}},
                },
            }
        )
        assert config.rails.tool_output.per_tool == {"run_sql": ["regex check tool output"]}


class TestAreToolCallsSafe:
    @pytest.mark.asyncio
    async def test_matching_tool_and_pattern_blocks(self):
        manager = _build_manager(
            per_tool_call_flows={"run_sql": ["regex check tool output"]}, regex_detection=RUN_SQL_PATTERN_CONFIG
        )
        result = await manager.are_tool_calls_safe([_sql_call("DROP TABLE users")], _llm_params("run_sql"))
        assert result.is_safe is False
        assert result.records[0].tool_name == "run_sql"
        assert result.records[0].flow == "regex check tool output"
        assert result.records[0].rail_type == "tool_output"
        assert result.records[0].return_value["detections"] == [r"DROP\s+TABLE"]

    @pytest.mark.asyncio
    async def test_matching_tool_non_matching_pattern_allows(self):
        manager = _build_manager(
            per_tool_call_flows={"run_sql": ["regex check tool output"]}, regex_detection=RUN_SQL_PATTERN_CONFIG
        )
        result = await manager.are_tool_calls_safe([_sql_call("SELECT 1")], _llm_params("run_sql"))
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_argument_scoping_excludes_unscoped_field_match(self):
        """$argument= scopes the check away from a matching field it doesn't name."""
        manager = _build_manager(
            per_tool_call_flows={"run_sql": ["regex check tool output $argument=query"]},
            regex_detection=RUN_SQL_PATTERN_CONFIG,
        )
        call = ToolCall(
            id="call_1",
            type="function",
            function=ToolCallFunction(
                name="run_sql", arguments={"query": "SELECT 1", "request_id": "DROP TABLE users"}
            ),
        )
        result = await manager.are_tool_calls_safe([call], _llm_params("run_sql"))
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_argument_scoping_includes_scoped_field_match(self):
        """$argument= still catches a match in the field it does name."""
        manager = _build_manager(
            per_tool_call_flows={"run_sql": ["regex check tool output $argument=query"]},
            regex_detection=RUN_SQL_PATTERN_CONFIG,
        )
        call = ToolCall(
            id="call_1",
            type="function",
            function=ToolCallFunction(name="run_sql", arguments={"query": "DROP TABLE users", "request_id": "abc"}),
        )
        result = await manager.are_tool_calls_safe([call], _llm_params("run_sql"))
        assert result.is_safe is False
        assert result.records[0].return_value["text"] == '{"query": "DROP TABLE users"}'

    @pytest.mark.asyncio
    async def test_tool_not_listed_in_per_tool_skips_check(self):
        manager = _build_manager(
            per_tool_call_flows={"run_sql": ["regex check tool output"]}, regex_detection=RUN_SQL_PATTERN_CONFIG
        )
        result = await manager.are_tool_calls_safe([_other_call()], {})
        assert result.is_safe
        assert result.records == ()

    @pytest.mark.asyncio
    async def test_no_per_tool_configured_is_unaffected(self):
        """REQ 2: a manager with no per_tool config behaves exactly as before."""
        manager = _build_manager()
        result = await manager.are_tool_calls_safe([_sql_call("DROP TABLE users")], {})
        assert result.is_safe
        assert result.records == ()

    @pytest.mark.asyncio
    async def test_undeclared_tool_blocks_per_tool_regex_check(self):
        """@tool_output_validation blocks a call whose tool isn't declared in llm_params,
        before the per-tool regex check itself ever runs."""
        manager = _build_manager(
            per_tool_call_flows={"run_sql": ["regex check tool output"]}, regex_detection=RUN_SQL_PATTERN_CONFIG
        )
        result = await manager.are_tool_calls_safe([_sql_call("SELECT 1")], {})
        assert_result_blocked(result)

    @pytest.mark.asyncio
    async def test_global_flow_blocks_before_per_tool_runs(self):
        manager = _build_manager(
            tool_call_flows=["tool call validation"],
            per_tool_call_flows={"run_sql": ["regex check tool output"]},
            regex_detection=RUN_SQL_PATTERN_CONFIG,
        )
        # No declared tools in llm_params -> the global allowlist check blocks first.
        result = await manager.are_tool_calls_safe([_sql_call("SELECT 1")], {})
        assert_result_blocked(result)
        assert result.records[0].rail_type == "tool_output"
        assert result.records[0].flow == "tool call validation"


class TestAreToolResultsSafe:
    def _messages(self, content: str, *, name: str | None = "run_sql") -> list:
        tool_message = {"role": "tool", "tool_call_id": "call_1", "content": content}
        if name is not None:
            tool_message["name"] = name
        return [
            {"role": "user", "content": "run a query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}
                ],
            },
            tool_message,
        ]

    @pytest.mark.asyncio
    async def test_matching_tool_and_pattern_blocks(self):
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        result = await manager.are_tool_results_safe(self._messages("ssn: 123-45-6789"))
        assert result.is_safe is False
        assert result.records[0].tool_name == "run_sql"
        assert result.records[0].rail_type == "tool_input"

    @pytest.mark.asyncio
    async def test_non_matching_pattern_allows(self):
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        result = await manager.are_tool_results_safe(self._messages("no sensitive data"))
        assert result.is_safe

    @pytest.mark.asyncio
    async def test_tool_name_resolved_from_prior_call_when_missing(self):
        """A tool message with no `name` still resolves via the matching prior call's id."""
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        result = await manager.are_tool_results_safe(self._messages("ssn: 123-45-6789", name=None))
        assert result.is_safe is False
        assert result.records[0].tool_name == "run_sql"

    @pytest.mark.asyncio
    async def test_tool_result_not_listed_in_per_tool_skips_check(self):
        """A result for a tool with no per_tool_result_flows entry resolves fine via
        call_id, but has nothing configured to check it against, so it must not block."""
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        messages = [
            {"role": "user", "content": "run a query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "other_tool", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "other_tool", "content": "ssn: 123-45-6789"},
        ]
        result = await manager.are_tool_results_safe(messages)
        assert result.is_safe
        assert result.records == ()

    @pytest.mark.asyncio
    async def test_no_per_tool_configured_is_unaffected(self):
        manager = _build_manager()
        result = await manager.are_tool_results_safe(self._messages("ssn: 123-45-6789"))
        assert result.is_safe
        assert result.records == ()

    @pytest.mark.asyncio
    async def test_extraction_skipped_when_disabled_despite_per_tool_configured(self):
        """enabled=False must disable per-tool flows too, not just the global ones.

        Proven by making extraction itself blow up: if the early-exit doesn't account for
        `enabled`, extraction still runs and its failure would incorrectly block the
        request even though the caller asked for no tool-result checking at all.
        """
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        with patch.object(manager.engine_registry, "extract_tool_exchanges", side_effect=RuntimeError("boom")):
            result = await manager.are_tool_results_safe(self._messages("ssn: 123-45-6789"), enabled=False)
        assert result.is_safe
        assert result.records == ()

    @pytest.mark.asyncio
    async def test_conflicting_supplied_name_does_not_override_call_id(self):
        """A result whose call_id identifies run_sql, but whose name claims a different
        tool, still resolves to run_sql. A spoofed name cannot steer the result to a
        different (or no) tool's policy and dodge the check that actually applies."""
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        result = await manager.are_tool_results_safe(self._messages("ssn: 123-45-6789", name="list_tables"))
        assert result.is_safe is False
        assert result.records[0].tool_name == "run_sql"

    @pytest.mark.asyncio
    async def test_unresolvable_call_id_blocks_when_a_per_tool_policy_is_enabled(self):
        """No `name` and no matching prior call, with a per-tool policy enabled: fails closed.

        An unresolvable identity must not silently skip every per-tool check, since that
        would let a malformed or unlinked result dodge whichever policy should have
        applied to it, the same class of bypass as trusting a spoofed name.
        """
        manager = _build_manager(
            per_tool_result_flows={"run_sql": ["regex check tool input"]},
            regex_detection=RUN_SQL_RESULT_PATTERN_CONFIG,
        )
        messages = self._messages("no sensitive data", name=None)
        messages[-1]["tool_call_id"] = "call_unknown"
        result = await manager.are_tool_results_safe(messages)
        assert result.is_safe is False

    @pytest.mark.asyncio
    async def test_global_only_config_does_not_require_result_linkage(self):
        """A global-only config (no per-tool result policy enabled) must not impose its
        own identity-resolution requirement: a custom global rail may intentionally
        allow an orphan/unresolved result, and the per-tool machinery must not override
        that just because it happens to run in the same request."""
        manager = _build_manager(tool_result_flows=["tool result validation"])
        with patch.object(manager, "_run_tool_result_rail", AsyncMock(return_value=RailResult.allow())):
            messages = self._messages("no sensitive data", name=None)
            messages[-1]["tool_call_id"] = "call_unknown"
            result = await manager.are_tool_results_safe(messages)
        assert result.is_safe


class TestResolveToolCallForResult:
    def test_returns_none_for_ambiguous_call_id(self):
        """Zero or multiple matching calls: resolves to None rather than guessing."""
        exchange = ToolExchange(calls=[], results=[])
        tool_result = ToolResult(call_id="call_unknown", name=None, content="hi")

        assert RailsManager._resolve_tool_call_for_result(exchange, tool_result) is None

    def test_ignores_supplied_name_when_call_id_has_no_match(self):
        """An unresolved call_id resolves to None, even with a name supplied.

        The supplied name is never trusted on its own: without a verified call_id match,
        there is nothing to corroborate it against, so it cannot select a policy.
        """
        exchange = ToolExchange(calls=[], results=[])
        tool_result = ToolResult(call_id="call_unknown", name="run_sql", content="hi")

        assert RailsManager._resolve_tool_call_for_result(exchange, tool_result) is None

    def test_returns_matching_call_by_id(self):
        """A unique call_id match resolves to that ToolCall."""
        call = _sql_call("SELECT 1")
        exchange = ToolExchange(calls=[call], results=[])
        tool_result = ToolResult(call_id=call.id, name="run_sql", content="hi")

        assert RailsManager._resolve_tool_call_for_result(exchange, tool_result) is call

    def test_returns_matching_call_by_id_despite_conflicting_name(self):
        """A conflicting supplied name does not change the call_id-resolved match.

        The supplied name is never trusted for selection, so it is not even compared
        against the resolved call's own name here; see
        test_conflicting_supplied_name_does_not_override_call_id for the end-to-end
        behavior this enables.
        """
        call = _sql_call("SELECT 1")
        exchange = ToolExchange(calls=[call], results=[])
        tool_result = ToolResult(call_id=call.id, name="list_tables", content="hi")

        assert RailsManager._resolve_tool_call_for_result(exchange, tool_result) is call


class TestIORailsWiring:
    def test_per_tool_flows_reach_rails_manager(self):
        config = RailsConfig.from_content(
            config={
                **STACK_CONFIG,
                "rails": {
                    "config": {"regex_detection": RUN_SQL_PATTERN_CONFIG},
                    "tool_output": {"per_tool": {"run_sql": ["regex check tool output"]}},
                },
            }
        )
        rails = IORails(config)
        assert rails.rails_manager.per_tool_call_flows == {"run_sql": ["regex check tool output"]}
        assert (SurfaceDirection.TOOL_OUTPUT, "regex check tool output") in rails.rails_manager._per_tool_rails

    def test_unsupported_reason_rejects_nonexistent_per_tool_flow(self):
        """A per-tool flow naming no catalog surface must be caught here, not raise
        RailCompilationError later when RailsManager tries to compile it."""
        config = RailsConfig.from_content(
            config={
                **STACK_CONFIG,
                "rails": {"tool_output": {"per_tool": {"run_sql": ["this flow does not exist"]}}},
            }
        )
        assert IORails.unsupported_reason(config) is not None


def _capture_per_tool_manager():
    """Build (manager, exporter) with a real tracer and content capture on, per-tool regex wired."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    config = RailsConfig.from_content(
        config={
            **STACK_CONFIG,
            "rails": {
                "config": {
                    "regex_detection": {
                        **RUN_SQL_PATTERN_CONFIG,
                        **RUN_SQL_RESULT_PATTERN_CONFIG,
                    }
                }
            },
        }
    )
    engine_registry = EngineRegistry(config.models)
    manager = RailsManager(
        engine_registry=engine_registry,
        task_manager=LLMTaskManager(config),
        input_flows=[],
        output_flows=[],
        per_tool_call_flows={"run_sql": ["regex check tool output"]},
        per_tool_result_flows={"run_sql": ["regex check tool input"]},
        tracer=provider.get_tracer("test"),
        content_capture_enabled=True,
    )
    return manager, exporter


class TestPerToolContentCapture:
    @pytest.mark.asyncio
    async def test_per_tool_call_span_captures_tool_name_and_arguments(self):
        manager, exporter = _capture_per_tool_manager()
        await manager.are_tool_calls_safe([_sql_call("SELECT 1")], {})
        spans = [s for s in exporter.get_finished_spans() if GuardrailsAttributes.RAIL_INPUT in s.attributes]
        assert len(spans) == 1
        payload = json.loads(spans[0].attributes[GuardrailsAttributes.RAIL_INPUT])
        assert payload["tool_name"] == "run_sql"
        assert payload["tool_call"]["function"]["arguments"]["query"] == "SELECT 1"

    @pytest.mark.asyncio
    async def test_per_tool_result_span_captures_tool_name_and_content(self):
        manager, exporter = _capture_per_tool_manager()
        messages = [
            {"role": "user", "content": "run a query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "no sensitive data"},
        ]
        await manager.are_tool_results_safe(messages)
        spans = [s for s in exporter.get_finished_spans() if GuardrailsAttributes.RAIL_INPUT in s.attributes]
        assert len(spans) == 1
        payload = json.loads(spans[0].attributes[GuardrailsAttributes.RAIL_INPUT])
        assert payload["tool_name"] == "run_sql"
        assert payload["tool_call"]["id"] == "call_1"
        assert payload["tool_result"]["content"] == "no sensitive data"
