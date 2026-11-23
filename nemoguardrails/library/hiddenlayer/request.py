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
it responds go to ``response-evaluations``. Every check is evaluated on its own.
"""

import uuid
from typing import Any, Dict, Optional

from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.library.hiddenlayer._verdict import Surface
from nemoguardrails.types import ToolCall

_REQUEST_PATH = "/detection/v2/nemo-guardrails/request-evaluations"
_RESPONSE_PATH = "/detection/v2/nemo-guardrails/response-evaluations"

_SURFACE_PATH: Dict[Surface, str] = {
    "input": _REQUEST_PATH,
    "tool_result": _REQUEST_PATH,
    "output": _RESPONSE_PATH,
    "tool_call": _RESPONSE_PATH,
}


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


def _evaluation_id() -> str:
    """A fresh ID for one evaluation.

    The routes require ``HL-Roundtrip-Id`` on every call. Each check gets its own, so
    HiddenLayer evaluates it without reference to any other check.
    """
    return str(uuid.uuid4())


def _headers(
    token: str,
    evaluation_id: str,
    requester_id: str,
    provider_id: Optional[str],
    *,
    project_id: Optional[str] = None,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HL-Roundtrip-Id": evaluation_id,
        "HL-Requester-Id": requester_id,
    }
    if provider_id:
        headers["HL-Provider-Id"] = provider_id
    if project_id:
        headers["HL-Project-Id"] = project_id
    return headers
