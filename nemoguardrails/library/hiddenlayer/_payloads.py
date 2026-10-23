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

Each guarded model call is evaluated as a single round-trip under one ``HL-Roundtrip-Id``: its
request evaluations (the input check and tool-result checks) go to ``request-evaluations`` and
carry an OpenAI chat-completion request, and its response evaluations (tool-call checks and the
output check) go to ``response-evaluations`` and carry a chat-completion response. Both routes
read the body as OpenAI Chat Completions whatever the model, and each body holds the current turn
only: conversation history reaches HiddenLayer through ``HL-Runtime-Session-Id``.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, Optional, Set, Union

from nemoguardrails.context import get_roundtrip_id
from nemoguardrails.guardrails.guardrails_types import get_request_id
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.library.hiddenlayer._verdict import Surface
from nemoguardrails.types import ToolCall

log = logging.getLogger(__name__)

_REQUEST_PATH = "/detection/v2/nemo-guardrails/request-evaluations"
_RESPONSE_PATH = "/detection/v2/nemo-guardrails/response-evaluations"

# The route carries the kind of evaluation, not the body.
_SURFACE_PATH: Dict[Surface, str] = {
    "input": _REQUEST_PATH,
    "tool_result": _REQUEST_PATH,
    "output": _RESPONSE_PATH,
    "tool_call": _RESPONSE_PATH,
}

MAX_SESSION_ID_BYTES = 2048

# Everything a header value cannot carry: non-ASCII, control characters (NUL, CR, LF) and DEL.
_UNSENDABLE_IN_HEADER = re.compile(r"[^\t\x20-\x7e]")

# Generated once; changing it changes every roundtrip ID derived from a request ID.
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


def _call_id(tool_call: ToolCall, tool_result: Optional[ToolResult] = None) -> str:
    """The call ID to send, synthesized when neither message carries one.

    A ``tool`` message whose ``tool_call_id`` answers no call in the body is rejected, so the same
    value must be used on both messages of one body.
    """
    return tool_call.id or (tool_result.call_id if tool_result is not None else None) or f"call_{uuid.uuid4().hex}"


def _tool_name(tool_call: ToolCall, tool_result: Optional[ToolResult] = None) -> str:
    return tool_call.function.name or (tool_result.name if tool_result is not None else None) or tool_call.type


def _arguments(arguments: Union[Dict[str, Any], str]) -> str:
    """OpenAI sends arguments as a JSON string; encode a dict once and never re-encode a string."""
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


def _call(tool_call: ToolCall, call_id: str, tool_name: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": _arguments(tool_call.function.arguments)},
    }


def _tool_definition(tool: Tool, tool_name: str) -> Dict[str, Any]:
    function: Dict[str, Any] = {"name": tool.name or tool_name}
    if tool.description:
        function["description"] = tool.description
    if tool.arguments_schema is not None:
        function["parameters"] = tool.arguments_schema
    return {"type": "function", "function": function}


def _joined_text(content: list) -> Optional[str]:
    """The text parts of *content* joined with newlines, or None when it yields no text."""
    texts = []
    for part in content:
        if not isinstance(part, dict) or not isinstance(part.get("type"), str):
            return None
        if part["type"] == "text":
            if not isinstance(part.get("text"), str):
                return None
            texts.append(part["text"])
    return "\n".join(texts) or None


def _result_content(content: Union[str, list, None]) -> str:
    """The tool result content to evaluate, always a string.

    A ``tool`` message's ``content`` is typed as a string, so a list of parts is sent as its text
    parts joined with newlines. A list that yields no text at all, such as one whose parts are of
    another vocabulary or are not typed objects, is sent as its JSON encoding: the empty string it
    would otherwise reduce to would leave the result evaluated as nothing.
    """
    if not content:
        return ""
    if isinstance(content, list):
        joined = _joined_text(content)
        return json.dumps(content) if joined is None else joined
    return content


def _completion(model: str, message: Dict[str, Any], finish_reason: str) -> Dict[str, Any]:
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def _input_interaction(model: str, text: str) -> Dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def _output_interaction(model: str, text: str) -> Dict[str, Any]:
    return _completion(model, {"role": "assistant", "content": text}, "stop")


def _tool_call_interaction(model: str, tool_call: ToolCall, call_id: str) -> Dict[str, Any]:
    message = {"role": "assistant", "content": None, "tool_calls": [_call(tool_call, call_id, _tool_name(tool_call))]}
    return _completion(model, message, "tool_calls")


def _tool_result_interaction(
    model: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
    call_id: str,
    tool_definition: Optional[Tool] = None,
) -> Dict[str, Any]:
    """A request evaluation holding the answered call and its result."""
    tool_name = _tool_name(tool_call, tool_result)
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [_call(tool_call, call_id, tool_name)]},
            {"role": "tool", "tool_call_id": call_id, "content": _result_content(tool_result.content)},
        ],
    }
    if tool_definition is not None:
        body["tools"] = [_tool_definition(tool_definition, tool_name)]
    return body


def _roundtrip_id() -> str:
    """The UUID identifying the model call this evaluation belongs to.

    ``HL-Roundtrip-Id`` must be a UUID, while the round-trip ID a caller sets is any string, so a
    value that is not already a UUID is mapped onto one. With no round trip and no request ID the
    evaluations of one model call cannot be paired, which costs the response evaluations their
    request context.
    """
    value = get_roundtrip_id()
    if value is None:
        request_id = get_request_id()
        if request_id == _NO_REQUEST_ID:
            _warn_once(
                "roundtrip",
                "No HiddenLayer round-trip ID is in scope; each check is evaluated on its own. "
                "Open one with nemoguardrails.context.roundtrip_id() to pair a model call's checks.",
            )
            return str(uuid.uuid4())
        value = request_id
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(_ROUNDTRIP_NAMESPACE, value))


def _session_header_value(session_id: Optional[str]) -> Optional[str]:
    """The session ID to send, or None when there is none to send or it cannot be sent.

    ``HL-Runtime-Session-Id`` is at most 2048 bytes, and a header value carries printable ASCII
    and tabs only, with no leading or trailing whitespace. An ID that breaks either rule is
    dropped rather than repaired: stripping or truncating can map two conversations onto one
    session and merge their histories, while dropping degrades to the "no history" the contract
    already allows.
    """
    if not session_id:
        return None
    if _UNSENDABLE_IN_HEADER.search(session_id):
        problem = "holds a character a header value cannot carry"
    elif session_id.strip(" \t") != session_id:
        problem = "starts or ends with whitespace"
    elif len(session_id.encode("utf-8")) > MAX_SESSION_ID_BYTES:
        problem = f"is over {MAX_SESSION_ID_BYTES} bytes"
    else:
        return session_id
    _warn_once(
        "session",
        "The conversation ID %s, so HiddenLayer evaluates these checks without session history.",
        problem,
    )
    return None


def _headers(
    token: str,
    roundtrip_id: str,
    requester_id: str,
    provider_id: str,
    *,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HL-Roundtrip-Id": roundtrip_id,
        "HL-Requester-Id": requester_id,
        "HL-Provider-Id": provider_id,
    }
    if session_id:
        headers["HL-Runtime-Session-Id"] = session_id
    if project_id:
        headers["HL-Project-Id"] = project_id
    return headers
