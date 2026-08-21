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
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeAlias

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.types import LLMResponse, UsageInfo

# LLMMessage can contain role/content, plus optional tool_calls / tool_call_id / name; content may be None
LLMMessage: TypeAlias = dict[str, Any]
LLMMessages: TypeAlias = list[LLMMessage]


def current_user_turn_index(messages: LLMMessages) -> Optional[int]:
    """Position of the turn being checked: the last user message that carries content."""
    for index, message in reversed(list(enumerate(messages))):
        if message.get("role") == "user" and message.get("content"):
            return index
    return None


def last_user_content(messages: LLMMessages) -> str:
    """Return the content of the turn being checked, or "" as the library actions expect."""
    index = current_user_turn_index(messages)
    return "" if index is None else messages[index]["content"]


def rewrite_user_message(messages: LLMMessages, text: str) -> LLMMessages:
    """Return *messages* with the turn ``last_user_content`` reads rewritten to *text*.

    Copied at both levels, because the caller's own list reaches the engine by identity.
    """
    index = current_user_turn_index(messages)
    if index is None:
        raise ValueError("no user turn carries content, so there is nothing to rewrite")
    rewritten = list(messages)
    rewritten[index] = {**messages[index], "content": text}
    return rewritten


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
    """Wrapper-class around `RailOutcome` object with IORails-specific metadata

    The verdict itself lives entirely in ``outcome``, which is the single source of
    truth: ``is_safe``, ``reason`` and ``return_value`` are derived views of it rather
    than a second copy that could drift. What this type adds is the aggregation
    ``RailOutcome`` has no concept of, because it belongs to running *many* rails:
    which one blocked (``triggered_rail``) and what every rail did (``records``).

    ``records`` carries the per-rail execution records for every rail that ran in this
    check (not just the blocking one), so IORails can synthesize a ``GenerationLog``.
    It is empty unless log collection is active, and it is log-capture data rather than
    part of the verdict, so it is excluded from equality (``compare=False``).

    ``__hash__`` is spelled out as None because ``RailOutcome`` is deliberately
    unhashable: without this a frozen dataclass would generate a ``__hash__`` that
    raises from *inside* ``hash()`` instead of reporting this type as unhashable.
    """

    outcome: RailOutcome
    triggered_rail: str | None = None
    records: tuple[RailCallRecord, ...] = field(default=(), compare=False)
    __hash__ = None

    @property
    def is_safe(self) -> bool:
        """Whether the checked content may proceed."""
        return not self.outcome.is_blocked

    @property
    def failed(self) -> bool:
        """Whether this block came from a rail that raised rather than one that decided."""
        return self.outcome.failed

    @property
    def reason(self) -> str | None:
        """The rail's own explanation, when it authored one."""
        return self.outcome.reason

    @property
    def return_value(self) -> dict[str, Any]:
        """The rail's structured verdict, as the log's ``ExecutedAction.return_value``.

        The verdict keys are applied last so they win: ``metadata`` is free-form evidence and a
        custom action may put an ``allowed`` or ``failed`` key in it, which must not be able to
        record a blocked rail as having allowed the content, nor forge a rail failure.

        ``failed`` is always present, as ``allowed`` is, so a log consumer reads a verdict
        rather than inferring one from a missing key. Without it a rail that broke and a rail
        that decided to block are the same record, which is the distinction the client-facing
        message already draws.
        """
        return {
            **self.outcome.metadata,
            _VERDICT_DECISION_KEY: self.is_safe,
            _VERDICT_FAILED_KEY: self.failed,
        }

    @classmethod
    def allow(
        cls,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        records: tuple[RailCallRecord, ...] = (),
    ) -> "RailResult":
        """A result that lets the content through."""
        return cls(RailOutcome.allow(reason=reason, metadata=metadata), records=records)

    @classmethod
    def block(
        cls,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        triggered_rail: str | None = None,
        records: tuple[RailCallRecord, ...] = (),
    ) -> "RailResult":
        """A result that stops the content. Only a block names a triggering rail."""
        return cls(
            RailOutcome.block(reason=reason, metadata=metadata),
            triggered_rail=triggered_rail,
            records=records,
        )


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


_VERDICT_DECISION_KEY = "allowed"
_VERDICT_FAILED_KEY = "failed"
_UNSPECIFIED_REASON = "unspecified"


def _rendered_evidence(value: Any) -> str:
    """Render one verdict value, flattening a sequence into a comma-separated list."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _has_evidence(value: Any) -> bool:
    """Whether a verdict value carries anything worth showing."""
    # Emptiness rather than falsiness, so a legitimate ``0`` or ``False`` still renders.
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) > 0
    return True


# Verdict fields that repeat the request rather than describe it: a block writes a log line
# unconditionally, and content belongs in a span only through configured content capture.
_CONTENT_EVIDENCE_KEYS = frozenset({"text", "user_message", "bot_message"})


def _metadata_evidence(metadata: Any) -> Optional[str]:
    """Render a rail's metadata as text, or None when it carries no evidence."""
    if not isinstance(metadata, Mapping):
        return None
    parts = [
        f"{key}: {_rendered_evidence(value)}"
        for key, value in metadata.items()
        if key not in _CONTENT_EVIDENCE_KEYS and _has_evidence(value)
    ]
    return "; ".join(parts) or None


def display_reason(result: RailResult) -> str:
    """Render a blocked rail's full explanation for a log line or a span."""
    if result.reason:
        return result.reason
    return _metadata_evidence(result.outcome.metadata) or result.triggered_rail or _UNSPECIFIED_REASON


def client_reason(result: RailResult) -> str:
    """Render a blocked rail's explanation for the error payload sent to the caller."""
    return result.reason or result.triggered_rail or _UNSPECIFIED_REASON
