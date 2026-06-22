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

import jsonschema


@dataclass(frozen=True, slots=True)
class Tool:
    """A declared tool definition the caller offered to the model.

    ``name`` is set for function tools and ``None`` for hosted/server tools that
    are identified only by ``type`` (e.g. web_search). ``arguments_schema`` is the
    JSON Schema for the call arguments, or ``None`` for hosted tools (and any
    function tool that declares no parameters), in which case argument validation
    is skipped and only the allowlist applies.
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


def validate_arguments(tool: Tool, arguments: dict) -> str | None:
    """Validate model-supplied tool-call arguments against the tool's schema.

    Returns ``None`` when the arguments are valid or the tool declares no schema.
    Returns a human-readable reason when the arguments violate the schema, or when
    the declared schema itself is not valid JSON Schema (e.g. a non-JSON-Schema
    dialect reaching this validator before its engine adapter normalizes it).
    """
    if tool.arguments_schema is None:
        return None
    try:
        jsonschema.validate(instance=arguments, schema=tool.arguments_schema)
    except jsonschema.ValidationError as exc:
        return f"arguments for tool '{tool.key}' do not match its schema: {exc.message}"
    except jsonschema.SchemaError as exc:
        return f"declared schema for tool '{tool.key}' is not valid JSON Schema: {exc.message}"
    return None
