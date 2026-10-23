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

"""Request bodies, headers and routes for the HiddenLayer NeMo-Guardrails evaluation API.

A body carries one check in the shape NeMo hands a rail, named by its ``check`` field and not
reshaped: HiddenLayer maps it onto its own interaction model. The route carries the direction,
so the checks made before the model is called go to ``request-evaluations`` and those made after
it responds go to ``response-evaluations``. Each guarded model call is evaluated as a single
round-trip under one ``HL-Roundtrip-Id``, and conversation history reaches HiddenLayer through
``HL-Runtime-Session-Id``.
"""

import logging
import uuid
from typing import Any, Dict, Optional
from urllib.parse import quote

from nemoguardrails.context import get_roundtrip_id
from nemoguardrails.guardrails.guardrails_types import get_request_id
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.library.hiddenlayer._verdict import Surface
from nemoguardrails.types import ToolCall

log = logging.getLogger(__name__)

_REQUEST_PATH = "/detection/v2/nemo-guardrails/request-evaluations"
_RESPONSE_PATH = "/detection/v2/nemo-guardrails/response-evaluations"

_SURFACE_PATH: Dict[Surface, str] = {
    "input": _REQUEST_PATH,
    "tool_result": _REQUEST_PATH,
    "output": _RESPONSE_PATH,
    "tool_call": _RESPONSE_PATH,
}

_NO_REQUEST_ID = "no-req-id"

_warned_without_roundtrip = False


def _reset_warnings() -> None:
    global _warned_without_roundtrip
    _warned_without_roundtrip = False


def _named(**fields: Any) -> Dict[str, Any]:
    """The fields NeMo supplied, without the ones it did not.

    A field left out is a field HiddenLayer fills in or records as absent, so an unset optional
    field is dropped rather than sent as null.
    """
    return {field: value for field, value in fields.items() if value is not None}


def _check(name: Surface, model: Optional[str], **fields: Any) -> Dict[str, Any]:
    return {"check": name, **_named(model=model or None, **fields)}


def _call(tool_call: ToolCall) -> Dict[str, Any]:
    return {"id": tool_call.id, "name": tool_call.function.name, "arguments": tool_call.function.arguments}


def _result(tool_result: ToolResult) -> Dict[str, Any]:
    return {
        "content": tool_result.content,
        "is_error": tool_result.is_error,
        **_named(call_id=tool_result.call_id, name=tool_result.name),
    }


def _definition(tool: Optional[Tool]) -> Optional[Dict[str, Any]]:
    if tool is None:
        return None
    return {"name": tool.key, **_named(description=tool.description, parameters=tool.arguments_schema)}


def _input_check(model: Optional[str], text: str) -> Dict[str, Any]:
    return _check("input", model, user_message=text)


def _output_check(model: Optional[str], text: str) -> Dict[str, Any]:
    return _check("output", model, bot_message=text)


def _tool_call_check(model: Optional[str], tool_call: ToolCall, tool_definition: Optional[Tool]) -> Dict[str, Any]:
    return _check("tool_call", model, tool_call=_call(tool_call), tool_definition=_definition(tool_definition))


def _tool_result_check(
    model: Optional[str],
    tool_call: ToolCall,
    tool_result: ToolResult,
    tool_definition: Optional[Tool],
) -> Dict[str, Any]:
    return _check(
        "tool_result",
        model,
        tool_call=_call(tool_call),
        tool_result=_result(tool_result),
        tool_definition=_definition(tool_definition),
    )


def _roundtrip_id() -> str:
    """The ID of the model call this evaluation belongs to, sent as the framework holds it.

    With no round trip and no request ID the evaluations of one model call cannot be paired, which
    costs the response evaluations their request context.
    """
    global _warned_without_roundtrip
    roundtrip_id = get_roundtrip_id() or get_request_id()
    if roundtrip_id != _NO_REQUEST_ID:
        return roundtrip_id
    if not _warned_without_roundtrip:
        _warned_without_roundtrip = True
        log.warning(
            "No HiddenLayer round-trip ID is in scope; each check is evaluated on its own. "
            "Open one with nemoguardrails.context.roundtrip_id() to pair a model call's checks."
        )
    return str(uuid.uuid4())


def _headers(
    token: str,
    roundtrip_id: str,
    requester_id: str,
    provider_id: Optional[str],
    *,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HL-Roundtrip-Id": roundtrip_id,
        "HL-Requester-Id": requester_id,
    }
    if provider_id:
        headers["HL-Provider-Id"] = provider_id
    if session_id:
        # Percent-encoded because a conversation ID is any string and a header value is ASCII.
        headers["HL-Runtime-Session-Id"] = quote(session_id, safe="")
    if project_id:
        headers["HL-Project-Id"] = project_id
    return headers
