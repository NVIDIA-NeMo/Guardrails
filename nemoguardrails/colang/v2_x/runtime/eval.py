# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set

import simpleeval
from simpleeval import EvalWithCompoundTypes

from nemoguardrails.colang.v2_x.lang.colang_ast import Element
from nemoguardrails.colang.v2_x.runtime import system_functions
from nemoguardrails.colang.v2_x.runtime.errors import ColangValueError
from nemoguardrails.colang.v2_x.runtime.flows import FlowState, State
from nemoguardrails.colang.v2_x.runtime.utils import (
    AttributeDict,
    escape_special_string_characters,
)
from nemoguardrails.logging.simplify_formatter import SimplifyFormatter
from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)


class ComparisonExpression:
    """An expression to compare to values."""

    def __init__(self, operator: Callable[[Any], bool], value: Any) -> None:
        if not isinstance(value, (int, float)):
            raise ColangValueError(
                f"Comparison operators don't support values of type '{type(value)}'"
            )
        self.value = value
        self.operator = operator

    def compare(self, value: Any) -> bool:
        """Compare given value with the expression's value."""
        if not isinstance(value, type(self.value)):
            raise ColangValueError(
                "Comparing variables of different types is not supported!"
            )

        return self.operator(value)


def eval_expression(expr: str, context: dict) -> Any:
    """Evaluates the provided expression in the given."""
    # If it's not a string, we should return it as such
    if expr is None:
        return None

    if not isinstance(expr, str):
        assert isinstance(expr, bool) or isinstance(expr, int)

        return expr

    # We search for all expressions in strings within curly brackets and evaluate them first
    # Find first all strings
    string_pattern = (
        r'("""|\'\'\')((?:\\\1|(?!\1)[\s\S])*?)\1|("|\')((?:\\\3|(?!\3).)*?)\3'
    )
    string_expressions_matches = re.findall(string_pattern, expr)
    string_expression_values = []
    for string_expression_match in string_expressions_matches:
        character = string_expression_match[0] or string_expression_match[2]
        string_expression = (
            character
            + (string_expression_match[1] or string_expression_match[3])
            + character
        )
        if string_expression:
            # Find expressions within curly brackets, ignoring double curly brackets
            expression_pattern = r"{(?!\{)([^{}]+)\}(?!\})"
            inner_expressions = re.findall(expression_pattern, string_expression)
            if inner_expressions:
                inner_expression_values = []
                for inner_expression in inner_expressions:
                    try:
                        value = eval_expression(inner_expression, context)
                    except Exception as ex:
                        raise ColangValueError(
                            f"Error evaluating inner expression: '{inner_expression}'"
                        ) from ex

                    value = str(value)

                    # Escape special characters
                    value = escape_special_string_characters(value)

                    inner_expression_values.append(value)
                string_expression = re.sub(
                    expression_pattern,
                    lambda x: inner_expression_values.pop(0),
                    string_expression,
                )
            string_expression = string_expression.replace("{{", "{").replace("}}", "}")
            string_expression_values.append(string_expression)
    if string_expression_values:
        expr = re.sub(
            string_pattern,
            lambda x: string_expression_values.pop(0),
            expr,
        )

    # We search for all variable names starting with $, remove the $ and add
    # the value in the dict for eval
    expr_locals = {}
    regex_pattern = r"\$([a-zA-Z_][a-zA-Z0-9_]*)"
    var_names = re.findall(regex_pattern, expr)
    updated_expr = re.sub(regex_pattern, r"var_\1", expr)

    for var_name in var_names:
        # if we've already computed the value, we skip
        if f"var_{var_name}" in expr_locals:
            continue

        # Check if it is a global variable
        global_var_name = f"_global_{var_name}"
        if global_var_name in context:
            val = context.get(global_var_name, None)
        else:
            val = context.get(var_name, None)

        # We transform dicts to AttributeDict so we can access their keys as attributes
        # e.g. write things like $speaker.name
        if isinstance(val, dict):
            val = AttributeDict(val)

        expr_locals[f"var_{var_name}"] = val

    # Finally, just evaluate the expression
    try:
        # TODO: replace this with something even more restrictive.
        functions = simpleeval.DEFAULT_FUNCTIONS.copy()
        functions.update(
            {
                "len": len,
                "flow": system_functions.flow,  # TODO: Consider this to remove
                "action": system_functions.action,  # TODO: Consider this to remove
                "regex": _create_regex,
                "search": _regex_search,
                "find_all": _regex_findall,
                "uid": new_uuid,
                "pretty_str": _pretty_str,
                "escape": _escape_string,
                "is_int": _is_int,
                "is_float": _is_float,
                "is_bool": _is_bool,
                "is_str": _is_str,
                "is_regex": _is_regex,
                "type": _get_type,
                "less_than": _less_than_operator,
                "equal_less_than": _equal_or_less_than_operator,
                "greater_than": _greater_than_operator,
                "equal_greater_than": _equal_or_greater_than_operator,
                "not_equal_to": _not_equal_to_operator,
                "list": list,
            }
        )
        if "system" in context and "state" in context["system"]:
            functions.update(
                {"flows_info": partial(_flows_info, context["system"]["state"])}
            )

        # TODO: replace this with something even more restrictive.
        s = EvalWithCompoundTypes(
            functions=functions,
            names=expr_locals,
        )

        result = s.eval(updated_expr)

        # Assign back changed values to dictionary variables
        for var_name, val in expr_locals.items():
            if isinstance(val, AttributeDict):
                var_name = var_name[4:]
                global_var_name = f"_global_{var_name}"
                if global_var_name in context:
                    context[global_var_name].clear()
                    context[global_var_name].update(val)
                else:
                    context[var_name].clear()
                    context[var_name].update(val)

        return result
    except Exception as e:
        raise ColangValueError(f"Error evaluating '{expr}', {e}")


def _create_regex(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def _regex_search(pattern: str, string: str) -> bool:
    return bool(re.search(pattern, string))


def _regex_findall(pattern: str, string: str) -> List[str]:
    return re.findall(pattern, string)


def _pretty_str(data: Any) -> str:
    if isinstance(data, (dict, list, set)):
        string = json.dumps(data, indent=4)
        return SimplifyFormatter().format(string)
    return str(data)


def _escape_string(string: str) -> str:
    """Escape a string and inner expressions."""
    return (
        string.replace("\\", "\\\\")
        .replace("{{", "\\{")
        .replace("}}", "\\}")
        .replace("'", "\\'")
        .replace('"', '\\"')
    )


def _is_int(val: Any) -> bool:
    """Check if it is an integer."""
    return isinstance(val, int)


def _is_float(val: Any) -> bool:
    """Check if it is an integer."""
    return isinstance(val, float)


def _is_bool(val: Any) -> bool:
    """Check if it is an integer."""
    return isinstance(val, bool)


def _is_str(val: Any) -> bool:
    """Check if it is an integer."""
    return isinstance(val, str)


def _is_regex(val: Any) -> bool:
    """Check if it is an integer."""
    return isinstance(val, re.Pattern)


def _get_type(val: Any) -> str:
    """ "Return type of variable as a string."""
    return type(val).__name__


def _less_than_operator(v_ref: Any) -> ComparisonExpression:
    """Create less then comparison expression."""
    return ComparisonExpression(lambda val, v_ref=v_ref: val < v_ref, v_ref)


def _equal_or_less_than_operator(v_ref: Any) -> ComparisonExpression:
    """Create equal or less than comparison expression."""
    return ComparisonExpression(lambda val, val_ref=v_ref: val <= val_ref, v_ref)


def _greater_than_operator(v_ref: Any) -> ComparisonExpression:
    """Create less then comparison expression."""
    return ComparisonExpression(lambda val, val_ref=v_ref: val > val_ref, v_ref)


def _equal_or_greater_than_operator(v_ref: Any) -> ComparisonExpression:
    """Create equal or less than comparison expression."""
    return ComparisonExpression(lambda val, val_ref=v_ref: val >= val_ref, v_ref)


def _not_equal_to_operator(v_ref: Any) -> ComparisonExpression:
    """Create a not equal comparison expression."""
    return ComparisonExpression(lambda val, val_ref=v_ref: val != val_ref, v_ref)


def _flows_info(state: State, flow_instance_uid: Optional[str] = None) -> dict:
    """Return a summary of the provided state, or all states by default."""
    if flow_instance_uid is not None and flow_instance_uid in state.flow_states:
        summary = {"flow_instance_uid": flow_instance_uid}
        summary.update(
            _flow_state_related_to_source(state, state.flow_states[flow_instance_uid])
        )

        return summary
    else:
        summary = {}
        for flow_state in state.flow_states.values():
            summary.update(
                {flow_state.uid: _flow_state_related_to_source(state, flow_state)}
            )
        return summary


def _flow_state_related_to_source(state: State, flow_state: FlowState):
    flow_config = state.flow_configs[flow_state.flow_id]
    flow_head_source_lines: Set[int] = set()
    for head in flow_state.active_heads.values():
        element = flow_config.elements[head.position]
        if isinstance(element, Element) and element._source is not None:
            flow_head_source_lines.add(element._source.line)
    summary: dict = {
        "flow_id": flow_state.flow_id,
        "loop_id": flow_state.loop_id,
        "status": flow_state.status.value,
        "flow_hierarchy": _get_flow_state_hierarchy(state, flow_state.uid)[:-1],
        "active_statement_at_lines": list(flow_head_source_lines),
        "meta_tags": flow_config.decorators.get("meta", []),
    }

    if flow_state.action_uids:
        summary.update({"action_uids": flow_state.action_uids})

    if flow_state.child_flow_uids:
        summary.update({"child_flow_uids": flow_state.child_flow_uids})

    return summary


def _get_flow_state_hierarchy(state: State, flow_state_uid: str) -> List[str]:
    if flow_state_uid not in state.flow_states:
        return []
    flow_state = state.flow_states[flow_state_uid]
    if flow_state.parent_uid is None:
        return [flow_state.uid]
    else:
        result = _get_flow_state_hierarchy(state, flow_state.parent_uid)
        result.append(flow_state.uid)
        return result
