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

"""Middleware that hands a guardrail rejection back to the model as a tool error, so the agent can correct itself."""

from collections.abc import Awaitable, Callable
from typing import Any, Optional, Union

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Relay raises a plain RuntimeError for a rejection, distinguishable only by this prefix.
RELAY_REJECTION_PREFIX = "guardrail rejected: "


def rejection_reason(exc: BaseException) -> Optional[str]:
    """The reason a Relay guardrail gave for rejecting a call, or None when *exc* is any other error."""
    # Exactly RuntimeError: a subclass, or a guardrail that raised (fail closed), is not a rejection.
    if type(exc) is not RuntimeError:
        return None
    message = str(exc)
    if not message.startswith(RELAY_REJECTION_PREFIX):
        return None
    return message[len(RELAY_REJECTION_PREFIX) :]


class DenyFeedbackMiddleware(AgentMiddleware):
    """Turn a Relay guardrail rejection into a `ToolMessage(status="error")` the model reads on its next turn.

    Place it before `NemoRelayDeepAgentsMiddleware` in the middleware list, so it wraps Relay's tool call.
    A failed check is not a rejection: it propagates, and the run stops.
    """

    def __init__(self, *, on_rejection: Optional[Callable[[str, str], None]] = None) -> None:
        super().__init__()
        self._on_rejection = on_rejection

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        try:
            return handler(request)
        except RuntimeError as exc:
            return self._feedback(request, exc)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Union[ToolMessage, Command]]],
    ) -> Union[ToolMessage, Command]:
        try:
            return await handler(request)
        except RuntimeError as exc:
            return self._feedback(request, exc)

    def _feedback(self, request: Any, exc: RuntimeError) -> ToolMessage:
        reason = rejection_reason(exc)
        if reason is None:
            raise exc
        tool_call = request.tool_call
        if self._on_rejection is not None:
            self._on_rejection(tool_call["name"], reason)
        return ToolMessage(
            content=reason,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            status="error",
        )
