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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import jsonschema

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
