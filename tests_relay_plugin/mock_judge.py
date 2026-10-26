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

"""A stand-in judge model for the hooks server's LLM rail: an OpenAI-compatible chat completions endpoint."""

import enum
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, cast

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
# A prompt that contains this gets an unsafe verdict, so a test picks the verdict with the tool call's arguments.
UNSAFE_MARKER = "attacker.example"
# The error message of the judge's HTTP 500.
SERVER_ERROR_MESSAGE = "the mock judge failed"


class JudgeFailure(enum.Enum):
    """A way the judge can fail to give a verdict."""

    MALFORMED_VERDICT = "malformed_verdict"
    SERVER_ERROR = "server_error"
    # The judge holds the request until the test ends, so the check outlasts any timeout.
    HANG = "hang"


class MockJudge:
    """Answers `safe`, or `unsafe: …` for a prompt with UNSAFE_MARKER, unless a test sets a failure."""

    def __init__(self) -> None:
        self.failure: Optional[JudgeFailure] = None
        self.prompts: list[str] = []
        # Set when a request starts to hang, so a test can tell the judge was reached.
        self.hanging = threading.Event()
        self._release = threading.Event()
        self._server = _JudgeServer(self)
        self._thread = threading.Thread(target=self._server.serve_forever, name="mock-judge", daemon=True)

    @property
    def base_url(self) -> str:
        """The URL a model config's `parameters.base_url` takes."""
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        """Serves requests on a thread of the test process."""
        self._thread.start()

    def stop(self) -> None:
        """Releases any hanging request and stops serving."""
        self._release.set()
        self._server.shutdown()
        self._server.server_close()

    def reset(self) -> None:
        """Releases any hanging request, then forgets the failure and the prompts, for the next test."""
        self._release.set()
        self._release = threading.Event()
        self.hanging = threading.Event()
        self.failure = None
        self.prompts = []

    def answer(self, prompt: str) -> tuple[int, dict]:
        """The status and body that answer *prompt*, after any hang ends."""
        self.prompts.append(prompt)
        failure = self.failure
        if failure is JudgeFailure.HANG:
            release = self._release
            self.hanging.set()
            release.wait()
        if failure is JudgeFailure.SERVER_ERROR:
            return 500, {"error": {"message": SERVER_ERROR_MESSAGE, "type": "server_error"}}
        if failure is JudgeFailure.MALFORMED_VERDICT:
            return 200, _completion("I can't tell.")
        if UNSAFE_MARKER in prompt:
            return 200, _completion(f"unsafe: the call reaches {UNSAFE_MARKER}")
        return 200, _completion("safe")


def _completion(content: str) -> dict:
    """A chat completion whose one choice is *content*."""
    return {
        "id": "chatcmpl-mock-judge",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-judge",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


class _JudgeServer(ThreadingHTTPServer):
    """A local HTTP server whose handlers answer with *judge*."""

    def __init__(self, judge: MockJudge) -> None:
        super().__init__(("127.0.0.1", 0), _JudgeHandler)
        self.judge = judge


class _JudgeHandler(BaseHTTPRequestHandler):
    """Answers POST /v1/chat/completions with the MockJudge's answer to the request's messages."""

    def do_POST(self) -> None:
        """Answers a chat completion request, or 404 for any other path."""
        if self.path != CHAT_COMPLETIONS_PATH:
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "\n".join(str(message.get("content", "")) for message in request["messages"])
        self._send(*cast(_JudgeServer, self.server).judge.answer(prompt))

    def _send(self, status: int, body: dict) -> None:
        """Writes *body* as JSON with *status*."""
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keeps request lines out of the test output; MockJudge.prompts records the requests."""
