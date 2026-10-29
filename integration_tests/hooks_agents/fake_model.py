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

"""A fake model for the agents: it calls one scripted tool, then answers with text, over the Anthropic and OpenAI APIs."""

# Claude Code talks to the Anthropic Messages API and Codex to the OpenAI Responses API, so one server answers both.
# The event sequences follow NeMo Relay's coding-agent test provider (scripts/test-support/codex_mock_provider.py) and
# Codex's own hook tests (codex-rs/core/tests/common/responses.rs).

import json
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional, Union, cast

TEXT_REPLY = "done"


@dataclass(frozen=True)
class ScriptedToolCall:
    """The tool call the fake model makes first, named and shaped as the agent's model API offers the tool."""

    name: str
    # A function tool's arguments, or a custom tool's free-form input, such as the code in a Codex code-mode cell.
    arguments: Union[dict[str, Any], str]


class FakeModel:
    """Serves the scripted tool call on loopback, and records every model request the agent sends."""

    def __init__(self, tool_call: ScriptedToolCall):
        self.tool_call = tool_call
        self._requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server = _Server(self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """The server's base URL, without a path."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeModel":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def tool_results(self) -> list[str]:
        """The text of each tool result the agent sent back, once each, in the order the agent sent them."""
        # Each request repeats the conversation so far, so a result appears again in every later request.
        results: dict[str, str] = {}
        with self._lock:
            for body in self._requests:
                for item in _tool_result_items(body):
                    results.setdefault(item.get("tool_use_id") or item.get("call_id") or "", _text(item))
        return list(results.values())

    def answer(self, body: dict[str, Any]) -> Optional[ScriptedToolCall]:
        """Record *body*, and return the tool call to answer it with, or None for the text reply."""
        with self._lock:
            self._requests.append(body)
        # A request without the tool, such as a title for the session, gets text, and so does every turn after the call.
        if self.tool_call.name in _offered_tools(body) and next(_tool_result_items(body), None) is None:
            return self.tool_call
        return None


def _offered_tools(body: dict[str, Any]) -> set[str]:
    """The names of the tools a request offers, in its `tools` or, in Codex's code mode, an `additional_tools` item."""
    tools = list(body.get("tools") or [])
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            tools.extend(item.get("tools") or [])
    names: set[str] = set()
    while tools:
        tool = tools.pop()
        if isinstance(tool, dict):
            names.add(tool.get("name", ""))
            # A namespace holds its tools.
            tools.extend(tool.get("tools") or [])
    return names


def _tool_result_items(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The tool results in a Messages request's `messages`, or in a Responses request's `input`."""
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            yield from (block for block in content if block.get("type") == "tool_result")
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") in ("function_call_output", "custom_tool_call_output"):
            yield item


def _text(item: dict[str, Any]) -> str:
    """A tool result's text: a Messages `tool_result`'s `content`, or a Responses tool call output's `output`."""
    value = item.get("content", item.get("output"))
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    return str(value)


class _Server(ThreadingHTTPServer):
    """The HTTP server on an ephemeral loopback port, holding the fake model its handler answers for."""

    def __init__(self, fake_model: FakeModel):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.fake_model = fake_model


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the server's access log out of the test output."""

    @property
    def fake_model(self) -> FakeModel:
        """The fake model this request is for."""
        return cast(_Server, self.server).fake_model

    def do_GET(self) -> None:  # noqa: N802
        """Answer Codex's model list with no models, as Codex's own test server does."""
        if self.path.split("?", 1)[0].rstrip("/").endswith("/models"):
            self._send_json({"models": []})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        """Answer a Messages or Responses request with the scripted tool call, or with the text reply."""
        path = self.path.split("?", 1)[0].rstrip("/")
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
        if path.endswith("/messages/count_tokens"):
            self._send_json({"input_tokens": 1})
        elif path.endswith("/messages"):
            tool_call = self.fake_model.answer(body)
            if body.get("stream"):
                self._send_events(_messages_events(body, tool_call))
            else:
                self._send_json(_message(body, tool_call))
        elif path.endswith("/responses"):
            self._send_events(_responses_events(self.fake_model.answer(body)))
        else:
            self.send_error(404)

    def _send_json(self, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_events(self, events: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


def _message(body: dict[str, Any], tool_call: Optional[ScriptedToolCall]) -> dict[str, Any]:
    """A whole Messages response."""
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": [_content_block(tool_call, partial=False)],
        "model": body.get("model", "claude-sonnet-4-5"),
        "stop_reason": "tool_use" if tool_call else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _content_block(tool_call: Optional[ScriptedToolCall], partial: bool) -> dict[str, Any]:
    """The response's one content block, the tool call or the text reply; *partial* leaves its body to the deltas."""
    if tool_call is None:
        return {"type": "text", "text": "" if partial else TEXT_REPLY}
    arguments = {} if partial else tool_call.arguments
    return {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex}", "name": tool_call.name, "input": arguments}


def _messages_events(body: dict[str, Any], tool_call: Optional[ScriptedToolCall]) -> list[dict[str, Any]]:
    """A streamed Messages response."""
    if tool_call is None:
        delta = {"type": "text_delta", "text": TEXT_REPLY}
    else:
        delta = {"type": "input_json_delta", "partial_json": json.dumps(tool_call.arguments)}
    return [
        {"type": "message_start", "message": {**_message(body, tool_call), "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": _content_block(tool_call, partial=True)},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if tool_call else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]


def _responses_events(tool_call: Optional[ScriptedToolCall]) -> list[dict[str, Any]]:
    """A streamed Responses response, with only the events Codex reads."""
    response_id = f"resp_{uuid.uuid4().hex}"
    if tool_call is None:
        item = {
            "type": "message",
            "role": "assistant",
            "id": f"msg_{uuid.uuid4().hex}",
            "content": [{"type": "output_text", "text": TEXT_REPLY}],
        }
    elif isinstance(tool_call.arguments, str):
        item = {
            "type": "custom_tool_call",
            "call_id": f"call_{uuid.uuid4().hex}",
            "name": tool_call.name,
            "input": tool_call.arguments,
        }
    else:
        item = {
            "type": "function_call",
            "call_id": f"call_{uuid.uuid4().hex}",
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments),
        }
    usage = {
        "input_tokens": 0,
        "input_tokens_details": None,
        "output_tokens": 0,
        "output_tokens_details": None,
        "total_tokens": 0,
    }
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.output_item.done", "item": item},
        {"type": "response.completed", "response": {"id": response_id, "usage": usage}},
    ]
