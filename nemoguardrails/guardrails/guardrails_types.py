# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import secrets
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeAlias

from nemoguardrails.types import LLMResponse, UsageInfo

# LLMMessage can contain role/content, plus optional tool_calls / tool_call_id / name; content may be None
LLMMessage: TypeAlias = dict[str, Any]
LLMMessages: TypeAlias = list[LLMMessage]


class RailDirection(Enum):
    """Direction of a rail check, used for logging."""

    INPUT = "Input"
    OUTPUT = "Output"


@dataclass(frozen=True, slots=True)
class RailCallRecord:
    """One rail's execution record, carried on RailResult for GenerationLog synthesis.

    Captures what a single rail did — its verdict and the (at most one) model call it
    made — as engine-neutral data. IORails maps a ``RailCallRecord`` to an
    ``ActivatedRail`` (with a single synthetic ``ExecutedAction`` and ``LLMCallInfo``);
    the raw ``usage``/timing is kept here so this module stays free of the pydantic
    ``GenerationLog`` types. Tool rails that make no model call leave ``usage`` None.
    """

    flow: str
    rail_type: str
    is_safe: bool
    made_call: bool = False
    action_name: Optional[str] = None
    return_value: Any = None
    task: Optional[str] = None
    request_id: Optional[str] = None
    usage: Optional[UsageInfo] = None
    llm_model_name: Optional[str] = None
    llm_provider_name: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration: Optional[float] = None


@dataclass(frozen=True, slots=True)
class RailResult:
    """Result of a rail safety check.

    ``records`` carries the per-rail execution records for every rail that ran in this
    check (not just the blocking one), so IORails can synthesize a ``GenerationLog``.
    It is empty unless log collection is active. ``return_value`` is the rail's
    structured verdict (e.g. ``{"allowed": ..., "policy_violations": [...]}``) when the
    action supplies one, used as the log's ``ExecutedAction.return_value``.

    ``records`` and ``return_value`` are log-capture metadata, not part of the safety
    verdict, so they are excluded from equality and hashing (``compare=False``) — two
    results with the same ``is_safe``/``reason``/``triggered_rail`` compare equal
    regardless of captured log data.
    """

    is_safe: bool
    reason: str | None = None
    triggered_rail: str | None = None
    records: tuple[RailCallRecord, ...] = field(default=(), compare=False)
    return_value: Any = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class TimedLLMResponse:
    """An LLM response paired with wall-clock start/finish timestamps and a monotonic duration.

    Returned by IORails' main-model call helper so the sequential and speculative paths both
    carry real timing into the generation ``RailCallRecord``.
    """

    response: LLMResponse
    started_at: float
    finished_at: float
    duration: float


# Default max character length for truncate(). Used to keep DEBUG log lines short.
LOG_CONTENT_TRUNCATE_LENGTH = 200

# Request ID sizing: 8 bytes → 16 hex characters (64 bits of entropy).
REQUEST_ID_BYTES = 8
REQUEST_ID_HEX_CHARS = REQUEST_ID_BYTES * 2

_request_id_var: ContextVar[str] = ContextVar("request_id", default="no-req-id")


def set_new_request_id() -> Token[str]:
    """Generate a random request ID, set it in the current context, and return the reset token."""
    rid = secrets.token_hex(REQUEST_ID_BYTES)
    return _request_id_var.set(rid)


def _set_request_id(request_id: str) -> Token[str]:
    """Set an explicit request ID (e.g., derived from an OTEL trace ID).

    Unlike ``set_new_request_id`` which generates a random ID, this accepts
    a caller-provided string.  Returns the reset token for use with
    ``reset_request_id``.
    """
    return _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the current per-request correlation ID."""
    return _request_id_var.get()


def reset_request_id(token: Token[str]) -> None:
    """Restore the request ID ContextVar to its previous value."""
    _request_id_var.reset(token)


def truncate(text: object, max_len: int | None = None) -> str:
    """Return ``str(text)`` truncated to *max_len* characters (default: LOG_CONTENT_TRUNCATE_LENGTH)."""
    s = str(text)
    limit = max_len if max_len is not None else LOG_CONTENT_TRUNCATE_LENGTH
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def serialize_prompt(messages: list[dict]) -> str:
    """Render a chat message list to a role-labeled string for GenerationLog's ``prompt``.

    Content parity with LLMRails' logged prompt, not byte-for-byte format parity: each
    message becomes ``"<role>: <content>"``. Non-content fields present on the message
    (``name``, ``tool_call_id``, ``tool_calls``, ``reasoning``) are appended as a compact
    ``[key=value, ...]`` suffix so tool-call and reasoning-only turns are preserved rather
    than dropped. Messages are blank-line separated.
    """
    lines = []
    for m in messages:
        line = f"{m.get('role', '')}: {m.get('content') or ''}"
        extras = [f"{key}={m[key]}" for key in ("name", "tool_call_id", "tool_calls", "reasoning") if m.get(key)]
        if extras:
            line += f" [{', '.join(extras)}]"
        lines.append(line)
    return "\n\n".join(lines)
