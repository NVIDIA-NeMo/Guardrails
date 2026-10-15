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

"""Keep Responses completion and stream identity checks outside field policy."""

import json

from nemoguardrails.server.experimental._json_payload import parse_json_object
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import GuardedStreamEvent, UnsupportedProviderStream


class ResponsesStreamHooks:
    """Require native completion and one consistent assistant identity.

    The kernel verifies every text snapshot against observed deltas. These hooks
    additionally reject post-terminal data and changes to supplied text item IDs,
    output positions, or sequence order. State belongs to one upstream stream.
    Missing unrelated provider metadata is not synthesized or required.
    """

    def __init__(self) -> None:
        self._terminal = False
        self._done = False
        self._sequence: int | None = None
        self._item_id: str | None = None
        self._output_index: int | None = None

    def _observe_text_identity(self, item_id: str | None, output_index: int | None) -> None:
        """Remember supplied identity and reject switching the guarded target."""
        if item_id is not None:
            if self._item_id is not None and item_id != self._item_id:
                raise UnsupportedProviderStream("Responses changed the guarded output item ID.")
            self._item_id = item_id
        if output_index is not None:
            if self._output_index is not None and output_index != self._output_index:
                raise UnsupportedProviderStream("Responses changed the guarded output index.")
            self._output_index = output_index

    def observe_event(self, event: ServerSentEvent, classified: GuardedStreamEvent) -> None:
        """Validate protocol state after the classifier has validated the event."""
        if event.data is None:
            return
        if event.data.strip() == b"[DONE]":
            if not self._terminal or self._done:
                raise UnsupportedProviderStream("Responses DONE requires a preceding native terminal event.")
            self._done = True
            return
        if self._terminal:
            raise UnsupportedProviderStream("Responses emitted data after native completion.")
        payload = parse_json_object(event.data)
        sequence = payload.get("sequence_number")
        if isinstance(sequence, int):
            if self._sequence is not None and sequence <= self._sequence:
                raise UnsupportedProviderStream("Responses sequence numbers must increase.")
            self._sequence = sequence
        shape = classified.shape
        if shape.startswith(("response.output_text.", "response.content_part.")):
            self._observe_text_identity(payload.get("item_id"), payload.get("output_index"))
        elif shape.endswith(":message"):
            self._observe_text_identity(payload["item"].get("id"), payload.get("output_index"))
        elif "response" in payload:
            for index, item in enumerate(payload["response"]["output"]):
                if item["type"] == "message":
                    self._observe_text_identity(item.get("id"), index)
        if shape in {"response.completed", "response.failed", "response.incomplete", "error"}:
            self._terminal = True

    def validate_end_of_stream(self) -> None:
        """Reject truncated foreground streams even if they contain DONE."""
        if not self._terminal:
            raise UnsupportedProviderStream("Responses ended without a native terminal event.")

    def encode_error(self, rendered_body: bytes) -> tuple[bytes]:
        """Render a terminal Responses error event, not a Chat DONE sentinel."""
        details = json.loads(rendered_body)["error"]
        payload = {
            "type": "error",
            "code": details["code"],
            "message": details["message"],
            "param": details.get("param"),
            "sequence_number": (self._sequence + 1 if self._sequence is not None else 0),
        }
        return (b"event: error\ndata: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n",)
