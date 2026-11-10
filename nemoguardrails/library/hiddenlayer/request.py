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
import re
import uuid
from typing import Any, Dict, Optional, Set
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

MAX_ROUNDTRIP_ID_BYTES = 256
MAX_SESSION_ID_BYTES = 2048

# Everything a header value cannot carry: non-ASCII, control characters (NUL, CR, LF) and DEL.
_UNSENDABLE_IN_HEADER = re.compile(r"[^\t\x20-\x7e]")

# Generated once; changing it changes every roundtrip ID derived from an unsendable one.
_ROUNDTRIP_NAMESPACE = uuid.UUID("19c5a9b4-eaa2-4f26-9ca7-ad76963cb8be")
_NO_REQUEST_ID = "no-req-id"

_warned: Set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log *message* the first time *key* is reported, and never again in this process."""
    if key in _warned:
        return
    _warned.add(key)
    log.warning(message, *args)


def _reset_warnings() -> None:
    _warned.clear()


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


def _is_nil_uuid(value: str) -> bool:
    """Whether *value* is the nil UUID in any of the forms a UUID parser accepts."""
    try:
        return uuid.UUID(value).int == 0
    except ValueError:
        return False


def _encodable(value: str) -> bool:
    """Whether *value* survives being encoded, which a lone surrogate does not."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _unsendable_as_held(roundtrip_id: str) -> Optional[str]:
    """Why *roundtrip_id* cannot be sent as the framework holds it, or None when it can."""
    if not _encodable(roundtrip_id):
        return "not valid UTF-8"
    if roundtrip_id != roundtrip_id.strip():
        # A hop that strips optional whitespace would leave the server an empty ID.
        return "surrounded by whitespace"
    if len(roundtrip_id.encode("utf-8")) > MAX_ROUNDTRIP_ID_BYTES:
        return f"over {MAX_ROUNDTRIP_ID_BYTES} bytes"
    if _UNSENDABLE_IN_HEADER.search(roundtrip_id):
        return "not sendable in a header"
    if _is_nil_uuid(roundtrip_id):
        return "the nil UUID, which names no round trip"
    return None


def _roundtrip_id() -> str:
    """The ID of the model call this evaluation belongs to, sent as the framework holds it.

    An ID the header cannot carry or the server rejects is mapped onto a UUID instead: the
    mapping is deterministic, so every check of one model call still pairs, and the evaluations
    still happen. With no round trip and no request ID they cannot be paired at all, which costs
    the response evaluations their request context.
    """
    roundtrip_id = get_roundtrip_id() or get_request_id()
    if roundtrip_id == _NO_REQUEST_ID:
        _warn_once(
            "roundtrip",
            "No HiddenLayer round-trip ID is in scope; each check is evaluated on its own. "
            "Open one with nemoguardrails.context.roundtrip_id() to pair a model call's checks.",
        )
        return str(uuid.uuid4())
    unsendable = _unsendable_as_held(roundtrip_id)
    if unsendable is not None:
        _warn_once(
            f"roundtrip-{unsendable}",
            "The round-trip ID is %s, so these checks are paired under a UUID derived from it.",
            unsendable,
        )
        return str(uuid.uuid5(_ROUNDTRIP_NAMESPACE, roundtrip_id))
    return roundtrip_id


def _session_header_value(session_id: Optional[str]) -> Optional[str]:
    """The session ID to send, or None when there is none to send or it cannot be sent.

    A conversation ID is any string and a header value is ASCII, so the ID is percent-encoded.
    One over ``MAX_SESSION_ID_BYTES`` is rejected, and is dropped rather than truncated:
    truncating can map two conversations onto one session and merge their histories, while
    dropping degrades to the "no history" the contract already allows and keeps the checks running.
    """
    if not session_id:
        return None
    if not _encodable(session_id):
        _warn_once(
            "session-encoding",
            "The conversation ID is not valid UTF-8, so HiddenLayer evaluates these checks without session history.",
        )
        return None
    if len(session_id.encode("utf-8")) > MAX_SESSION_ID_BYTES:
        _warn_once(
            "session",
            "The conversation ID is over %s bytes, so HiddenLayer evaluates these checks without session history.",
            MAX_SESSION_ID_BYTES,
        )
        return None
    return quote(session_id, safe="")


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
    session_value = _session_header_value(session_id)
    if session_value:
        headers["HL-Runtime-Session-Id"] = session_value
    if project_id:
        headers["HL-Project-Id"] = project_id
    return headers
