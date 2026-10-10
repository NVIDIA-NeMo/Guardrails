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

import functools
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult, tool_output_validation
from nemoguardrails.library.cel.rail_config import CelOptions, compile_expression
from nemoguardrails.types import ToolCall

if TYPE_CHECKING:
    from cel import Program

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1024)
def _program(expression: str) -> "Program":
    """The compiled program for *expression*, cached outside the config (see CelOptions.compile_expressions)."""
    return compile_expression(expression)


def _tool_options(config: RailsConfig, source: str, tool_name: str) -> Optional[CelOptions]:
    """Return the tool's CEL options for *source*, or None when none are configured."""
    cel_config = config.rails.config.cel
    if cel_config is None:
        return None
    return getattr(cel_config, source).get(tool_name)


def _evaluate(source: str, tool_name: str, variables: Dict[str, Any], options: Optional[CelOptions]) -> RailOutcome:
    """Evaluate all of a tool's CEL expressions, blocking when any is true.

    A failing expression (an error or a non-boolean result) raises even if another expression
    matched, so the rail fails closed.
    """
    metadata: Dict[str, Any] = {"source": source}
    if options is None:
        log.debug("No CEL expressions configured for tool %r under source: %s", tool_name, source)
        return RailOutcome.allow(metadata=metadata)

    # Already imported when the expressions compiled at config load.
    import cel
    from cel.stdlib import add_stdlib_to_context

    context = cel.Context()
    add_stdlib_to_context(context)
    context.update(variables)

    matched: List[str] = []
    failures: List[Tuple[str, Exception]] = []
    for expression in options.expressions:
        try:
            result = _program(expression).execute(context)
            if not isinstance(result, bool):
                raise TypeError(f"returned {type(result).__name__}, not bool")
        except Exception as e:
            failures.append((expression, e))
            continue
        if result:
            log.info("CEL expression matched: %s", expression)
            matched.append(expression)

    if failures:
        # Only the exception type: its message can quote the checked arguments or result.
        details = "; ".join(f"{expression!r}: {type(error).__name__}" for expression, error in failures)
        raise RuntimeError(f"CEL expressions for tool {tool_name!r} failed: {details}")
    if matched:
        return RailOutcome.block(metadata={**metadata, "matched_expressions": matched})
    return RailOutcome.allow(metadata=metadata)


@action(is_system_action=True)
@tool_output_validation
async def evaluate_tool_output_cel(
    source: str,
    tool_call: ToolCall,
    tool_definition: Optional[Tool],
    config: RailsConfig,
    **kwargs,
) -> RailOutcome:
    """Checks a tool call's arguments against that tool's CEL expressions.

    Args:
        source: Fixed per surface, always "tool_output".
        tool_call: The tool call to check. Its arguments are `args` and its name is `tool`.
        tool_definition: The declared tool, used only by @tool_output_validation to check
            the call's arguments against its schema before this function runs.
        config: The rails configuration object.

    Returns:
        RailOutcome that blocks when any expression is true, with the matched expressions in
        metadata.
    """
    if source != "tool_output":
        raise ValueError("source must be 'tool_output'")

    tool_name = tool_call.function.name or tool_call.type
    options = _tool_options(config, source, tool_name)
    variables = {"args": tool_call.function.arguments, "tool": tool_name}
    return _evaluate(source, tool_name, variables, options)


@action(is_system_action=True)
async def evaluate_tool_input_cel(
    source: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
    config: RailsConfig,
    **kwargs,
) -> RailOutcome:
    """Checks a tool result's content against that tool's CEL expressions.

    Args:
        source: Fixed per surface, always "tool_input".
        tool_call: The prior call this result answers. Its arguments are `args` and its name
            is `tool`.
        tool_result: The tool result to check. Its content is `result`, as the tool returned it.
        config: The rails configuration object.

    Returns:
        RailOutcome that blocks when any expression is true, with the matched expressions in
        metadata.
    """
    if source != "tool_input":
        raise ValueError("source must be 'tool_input'")

    tool_name = tool_call.function.name or tool_call.type
    options = _tool_options(config, source, tool_name)
    variables = {"result": tool_result.content, "args": tool_call.function.arguments, "tool": tool_name}
    return _evaluate(source, tool_name, variables, options)
