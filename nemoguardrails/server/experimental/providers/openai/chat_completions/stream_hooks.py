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

"""Bind OpenAI Chat Completions protocol hooks to the staged classifier."""

from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import (
    GuardedStreamEvent,
    StreamEventRole,
    UnsupportedProviderStream,
)


class ChatCompletionsStreamHooks:
    """Require successful streams to end with ``[DONE]``.

    A provider error may end the stream without a sentinel. Data-less keepalives
    are permitted throughout; payload events cannot follow an error or sentinel,
    except for a sentinel after an error. State belongs to one upstream stream.
    """

    def __init__(self) -> None:
        self._done = False
        self._provider_error = False

    def observe_event(self, event: ServerSentEvent, classified: GuardedStreamEvent) -> None:
        """Track the actual sentinel and reject payloads after termination."""

        data = event.data
        if data is None:
            return
        if self._done:
            raise UnsupportedProviderStream("A Chat Completions data event follows after [DONE].")
        if data.strip() == b"[DONE]":
            self._done = True
            return
        if self._provider_error:
            raise UnsupportedProviderStream("A Chat Completions payload follows after a provider error.")
        if classified.role is StreamEventRole.PROVIDER_ERROR:
            self._provider_error = True

    def validate_end_of_stream(self) -> None:
        """Reject successful streams that end before their sentinel."""

        if not self._done and not self._provider_error:
            raise UnsupportedProviderStream("The Chat Completions stream ended before [DONE].")

    def encode_error(self, rendered_body: bytes) -> tuple[bytes, bytes]:
        """Encode an OpenAI-compatible error and terminal chat event."""

        return b"data: " + rendered_body + b"\n\n", b"data: [DONE]\n\n"
