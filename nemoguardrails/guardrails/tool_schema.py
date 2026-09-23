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

"""Internal tool types for IORails tool-calling rails.

These engine-internal dataclasses are the normalized, provider-neutral shape the
tool rails validate against. They are NOT part of the public API and NOT carried
on ``GenerationOptions``: the request surface stays the provider-native
``llm_params`` block, which ``ModelEngine`` parses into a ``Toolset`` (and incoming
tool results into ``ToolResult`` objects) per inference call.

``Tool`` is a declared tool definition (what the caller offers); ``ToolCall`` (in
``nemoguardrails.types``) is an invocation the model emitted. Field names are
provider-neutral so the per-provider adapters all produce the same shape:
``arguments_schema`` is OpenAI ``parameters`` / Anthropic ``input_schema`` / Gemini
``parameters``; ``ToolResult.call_id`` is the OpenAI ``tool_call_id`` / Responses
``call_id`` / Anthropic ``tool_use_id`` / Gemini function-call ``id``. OpenAI Chat
Completions is the engine implemented today.
"""

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jsonschema

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.types import ToolCall


@dataclass(frozen=True, slots=True)
class Tool:
    """A declared tool definition the caller offered to the model.

    ``name`` is set for function tools and ``None`` for hosted/server tools that
    are identified only by ``type`` (e.g. web_search). ``arguments_schema`` is the
    JSON Schema for the call arguments, or ``None`` when none is declared. A hosted
    tool with no schema is allowlist-only (the provider owns the call shape); a
    function tool that declares no parameters accepts no arguments, so a call that
    supplies any is rejected.
    """

    name: str | None = None
    type: str = "function"
    description: str | None = None
    arguments_schema: dict | None = None
    strict: bool | None = None

    @property
    def key(self) -> str:
        """Allowlist / lookup identifier: the function ``name``, else the ``type``."""
        return self.name or self.type


class Toolset:
    """The set of tools declared on a request, indexed by tool key.

    Backed by a single ``key -> Tool`` mapping built at construction, so there is
    no separate list that can drift out of sync with the index. Look tools up with
    :meth:`get` (``toolset.get(name)``); iterate or count via the read-only
    :attr:`tools`.
    """

    __slots__ = ("_by_key",)

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        """Index *tools* by their ``key``, rejecting duplicates.

        A toolset must not declare the same function name (or hosted-tool type)
        twice, so a repeated ``key`` raises ``ValueError``. A tool with an empty
        ``key`` (no name and no type) has no lookup identifier and is dropped.
        """
        self._by_key: dict[str, Tool] = {}
        for tool in tools or []:
            if not tool.key:
                continue
            if tool.key in self._by_key:
                raise ValueError(f"duplicate tool '{tool.key}' in toolset")
            self._by_key[tool.key] = tool

    def get(self, key: str) -> Tool | None:
        """Return the declared tool registered under *key* (function name or hosted-tool type), or None."""
        return self._by_key.get(key)

    @property
    def tools(self) -> tuple[Tool, ...]:
        """The declared tools in declaration order (read-only view of the index)."""
        return tuple(self._by_key.values())


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A normalized tool result extracted from incoming messages by the engine.

    The ToolResultRail consumes a list of these; the per-provider extraction (e.g.
    OpenAI ``role:"tool"`` messages) lives in the engine adapter, so the rail never
    sees provider wire shapes. ``content`` is a string or a list of content blocks
    (the latter covers multimodal results). ``is_error`` flags a failed result
    where the provider exposes one (e.g. Anthropic ``is_error`` / Bedrock
    ``status:"error"``).
    """

    call_id: str | None = None
    name: str | None = None
    content: str | list[dict] | None = None
    is_error: bool = False

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
        }


class ToolExchange(NamedTuple):
    """One assistant turn's tool calls paired with the tool results that answer them."""

    calls: list[ToolCall]
    results: list[ToolResult]


def _schema_accepts_no_arguments(schema: dict) -> bool:
    """Whether an argument schema declares no way to supply arguments.

    True for an empty schema (``{}``) or an object schema that names no ``properties``
    and opens no other input channel (``additionalProperties``, ``patternProperties``,
    or a composition/reference keyword). Such a schema describes a tool that takes no
    arguments; jsonschema treats it as permissive, so callers enforce emptiness directly.
    """
    if not schema:
        return True
    if schema.get("properties"):
        return False
    if schema.get("additionalProperties"):
        return False
    if schema.get("patternProperties"):
        return False
    return not any(keyword in schema for keyword in ("anyOf", "oneOf", "allOf", "$ref"))


def _schema_rejects_argument(schema: dict, argument_name: str) -> bool:
    """Whether *schema* has no channel that could accept *argument_name*.

    Matches ``patternProperties`` with ``re.search``, since JSON Schema patterns match
    anywhere in the string. Composition/reference keywords bail out conservatively, same as
    ``_schema_accepts_no_arguments``.
    """
    if argument_name in schema.get("properties", {}):
        return False
    if schema.get("additionalProperties") is not False:
        return False
    if any(re.search(pattern, argument_name) for pattern in schema.get("patternProperties", {})):
        return False
    return not any(keyword in schema for keyword in ("anyOf", "oneOf", "allOf", "$ref"))


def _no_arguments_reason(tool: Tool, arguments: dict) -> str | None:
    """Block reason for a tool that accepts no arguments, or ``None`` when the call is allowed.

    Hosted/server tools (no ``name``) are allowlist-only -- the provider owns the call
    shape -- so arguments are accepted here. A function tool that declares no parameters
    must be called with no arguments, so any supplied argument is rejected.
    """
    if tool.name is None:
        return None
    if arguments:
        return f"tool '{tool.key}' accepts no arguments but the call supplied: {sorted(arguments)}"
    return None


def validate_arguments(tool: Tool, arguments: dict) -> str | None:
    """Validate model-supplied tool-call arguments against the tool's schema.

    Returns ``None`` when the arguments are valid. Returns a human-readable reason when
    the arguments violate the schema, when the declared schema itself is not valid JSON
    Schema (e.g. a non-JSON-Schema dialect reaching this validator before its engine
    adapter normalizes it), or when a function tool that declares no parameters is called
    with arguments.
    """
    if tool.arguments_schema is None:
        return _no_arguments_reason(tool, arguments)
    try:
        jsonschema.validate(instance=arguments, schema=tool.arguments_schema)
    except jsonschema.ValidationError as exc:
        return f"arguments for tool '{tool.key}' do not match its schema: {exc.message}"
    except jsonschema.SchemaError as exc:
        return f"declared schema for tool '{tool.key}' is not valid JSON Schema: {exc.message}"
    if _schema_accepts_no_arguments(tool.arguments_schema):
        return _no_arguments_reason(tool, arguments)
    return None


def tool_output_validation(func: Callable[..., Any]) -> Callable[..., Any]:
    """Validate a TOOL_OUTPUT action's arguments against the tool's schema before it runs.

    Every action bound to a ``TOOL_OUTPUT`` surface must carry this decorator (enforced by
    ``test_every_tool_output_action_validates_arguments``). Blocks before the action body
    runs if the call's tool isn't declared, or its arguments don't match the schema. Raises
    if the action was bound a ``$argument=`` name (see ``scope_arguments``) the schema
    doesn't accept, since that is a config defect rather than a per-call decision.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> RailOutcome:
        tool_call: ToolCall = kwargs["tool_call"]
        tool_definition: Tool | None = kwargs["tool_definition"]
        name = tool_call.function.name or tool_call.type
        if tool_definition is None:
            return RailOutcome.block(reason=f"tool call '{name}' is not an allowed tool")
        reason = validate_arguments(tool_definition, tool_call.function.arguments)
        if reason is not None:
            return RailOutcome.block(reason=reason)

        argument_name = kwargs.get("argument_name")
        if argument_name is not None:
            schema = tool_definition.arguments_schema
            if schema is None:
                raise ValueError(f"tool '{name}' declares no schema to validate argument '{argument_name}' against")
            if _schema_rejects_argument(schema, argument_name):
                raise ValueError(f"argument '{argument_name}' is not declared in tool '{name}' schema")

        return await func(*args, **kwargs)

    setattr(wrapper, "_has_tool_output_validation", True)
    return wrapper


def scope_arguments(arguments: dict, argument_name: str | None) -> dict:
    """Narrow *arguments* to one named argument, or return it unchanged.

    ``argument_name`` comes from a flow's ``$argument=<name>`` parameter, frozen at
    compile time. Narrowing lets a check inspect one user-supplied field without seeing
    unrelated call metadata that could false-positive.

    A call that omits the named argument (schema-valid, e.g. an optional field the model
    didn't set this time) falls back to the full, unscoped arguments rather than
    ``{argument_name: None}``, so nothing goes unchecked just because one call happened
    to leave a field out. This leans toward more scrutiny, not less: a check that scoped
    to that name specifically to *avoid* an unrelated field's content may see it anyway on
    a call where the named field is absent. A config typo or a name the schema could never
    produce is caught earlier, at compile time, by ``tool_output_validation``.

    TODO: only a single argument name is supported today; add delimiter-separated
    multi-argument support (e.g. `$argument=a,b`) as a follow-up.
    """
    if argument_name is None or argument_name not in arguments:
        return arguments
    return {argument_name: arguments[argument_name]}
