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

"""Typed streaming error chunks for guardrails streaming responses.

Before this module existed, IORails and LLMRails each constructed error
payloads inline with ``json.dumps({"error": {...}})`` and slightly different
key sets (LLMRails included ``param``, IORails did not; type and code
strings were free-form). Consumers downstream had no contract to parse
against, and the same error JSON was indistinguishable from regular LLM
output so it leaked into the ``RollingBuffer`` and tripped output rails.

The ``StreamError`` model defined here is the single source of truth:

- Both rails engines construct ``StreamError`` instances and serialize via
  :func:`stream_error_to_chunk` when pushing into a ``StreamingHandler``.
- :func:`parse_stream_error_chunk` recovers a ``StreamError`` from a chunk
  string; downstream code (the buffer-skip path and consumer parsers)
  uses this instead of substring or shape heuristics.
- :class:`StreamErrorType` and :class:`StreamErrorCode` enums lock down the
  vocabulary used historically by both engines, so type/code values stop
  drifting between code sites.

Backwards compatibility: the serialized JSON shape matches what consumers
already receive (``{"error": {"message": ..., "type": ..., "code": ...,
"param": ...}}``), so existing OpenAI-style error handling keeps working
without changes.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


__all__ = [
    "StreamError",
    "StreamErrorCode",
    "StreamErrorDetails",
    "StreamErrorType",
    "parse_stream_error_chunk",
    "stream_error_to_chunk",
]


class StreamErrorType(str, Enum):
    """High-level category of a streaming error chunk.

    Values are stable across releases because they appear on the wire to
    consumers. They mirror the categories the rails engines were already
    emitting inline.
    """

    GENERATION_ERROR = "generation_error"
    GUARDRAILS_VIOLATION = "guardrails_violation"
    INTERNAL_ERROR = "internal_error"


class StreamErrorCode(str, Enum):
    """Specific machine-readable reason for a streaming error chunk.

    Pairs with :class:`StreamErrorType` to give consumers enough resolution
    to branch on (e.g. retry on ``generation_failed`` versus surfacing a
    refusal on ``content_blocked``).
    """

    GENERATION_FAILED = "generation_failed"
    CONTENT_BLOCKED = "content_blocked"
    RAIL_EXECUTION_FAILURE = "rail_execution_failure"


class StreamErrorDetails(BaseModel):
    """Inner payload describing a single streaming error.

    ``type`` and ``code`` are typed as strings rather than the enums above
    so the model also accepts upstream error shapes (e.g. OpenAI 400
    payloads passed through :func:`nemoguardrails.utils.extract_error_json`)
    without raising during construction. Callers within this repo should
    always use :class:`StreamErrorType` / :class:`StreamErrorCode` values
    when building a ``StreamError`` so vocabulary stays consistent.
    """

    model_config = ConfigDict(extra="allow")

    message: str
    type: str
    code: str
    # `param` carries the flow id when applicable; absent for engine-level
    # errors like generation failures.
    param: Optional[str] = None


class StreamError(BaseModel):
    """A typed streaming error chunk.

    The on-the-wire shape is ``{"error": {...}}`` matching the inline
    ``json.dumps({"error": ...})`` payloads the rails engines used to emit.
    Construct via the convenience classmethods (:meth:`generation_failed`,
    :meth:`content_blocked`, :meth:`rail_execution_failure`) when possible
    so type/code stay aligned with the enums.
    """

    model_config = ConfigDict(extra="allow")

    error: StreamErrorDetails = Field(..., description="Structured error payload.")

    def to_chunk(self) -> str:
        """Serialize to the on-the-wire JSON string used by streaming handlers."""
        return self.model_dump_json()

    def __str__(self) -> str:
        # Stringifying a StreamError gives the same JSON string consumers
        # used to receive from the inline ``json.dumps`` call sites.
        return self.to_chunk()

    @classmethod
    def from_error_dict(cls, error_dict: dict) -> "StreamError":
        """Build a ``StreamError`` from a loose ``{"error": {...}}`` dict.

        Tolerates partial dicts (e.g. ``extract_error_json`` may produce
        just ``{"error": {"message": ...}}``) by defaulting missing
        ``type`` and ``code`` to ``internal_error`` / ``generation_failed``
        respectively; the resulting chunk still carries the original
        message verbatim.
        """
        inner = error_dict.get("error", {}) if isinstance(error_dict, dict) else {}
        if not isinstance(inner, dict):
            inner = {}

        message = inner.get("message")
        if not isinstance(message, str):
            message = str(message) if message is not None else "Unknown error"

        type_value = inner.get("type") or StreamErrorType.INTERNAL_ERROR.value
        code_value = inner.get("code") or StreamErrorCode.GENERATION_FAILED.value
        param_value = inner.get("param") if isinstance(inner.get("param"), str) else None

        details_kwargs = {
            "message": message,
            "type": str(type_value),
            "code": str(code_value),
            "param": param_value,
        }
        # Preserve any unknown keys the upstream provider sent (`extra="allow"`).
        for key, value in inner.items():
            if key not in details_kwargs:
                details_kwargs[key] = value

        return cls(error=StreamErrorDetails(**details_kwargs))

    @classmethod
    def generation_failed(
        cls,
        message: str,
        *,
        type_: StreamErrorType = StreamErrorType.GENERATION_ERROR,
        code: StreamErrorCode = StreamErrorCode.GENERATION_FAILED,
        param: Optional[str] = None,
    ) -> "StreamError":
        """Construct a ``generation_error`` / ``generation_failed`` chunk."""
        return cls(
            error=StreamErrorDetails(
                message=message,
                type=type_.value,
                code=code.value,
                param=param,
            )
        )

    @classmethod
    def content_blocked(cls, message: str, *, param: Optional[str] = None) -> "StreamError":
        """Construct a ``guardrails_violation`` / ``content_blocked`` chunk."""
        return cls(
            error=StreamErrorDetails(
                message=message,
                type=StreamErrorType.GUARDRAILS_VIOLATION.value,
                code=StreamErrorCode.CONTENT_BLOCKED.value,
                param=param,
            )
        )

    @classmethod
    def rail_execution_failure(cls, message: str, *, param: Optional[str] = None) -> "StreamError":
        """Construct an ``internal_error`` / ``rail_execution_failure`` chunk."""
        return cls(
            error=StreamErrorDetails(
                message=message,
                type=StreamErrorType.INTERNAL_ERROR.value,
                code=StreamErrorCode.RAIL_EXECUTION_FAILURE.value,
                param=param,
            )
        )


def stream_error_to_chunk(error: StreamError) -> str:
    """Serialize a ``StreamError`` to the chunk string that goes on the wire.

    This is the inverse of :func:`parse_stream_error_chunk` and exists so
    rails engines have a single, named call site to use when bridging
    typed errors into the existing string-based ``StreamingHandler`` API.
    """
    return error.to_chunk()


def parse_stream_error_chunk(chunk: object) -> Optional[StreamError]:
    """Try to recover a ``StreamError`` from a streamed chunk.

    Returns ``None`` when the chunk is not a JSON-encoded ``StreamError``
    payload, so callers can use this as a typed test (``if parse_... is not
    None``) to distinguish error chunks from ordinary LLM output. Accepts
    dicts directly in case a future ``StreamingHandler`` mode forwards
    structured chunks without serialization.
    """
    if isinstance(chunk, StreamError):
        return chunk

    if isinstance(chunk, dict):
        payload = chunk
    elif isinstance(chunk, str):
        chunk_str = chunk.lstrip()
        if not chunk_str.startswith("{"):
            return None
        try:
            payload = json.loads(chunk_str)
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        return None

    if not isinstance(payload, dict) or "error" not in payload:
        return None

    try:
        return StreamError.model_validate(payload)
    except Exception:  # pragma: no cover - defensive
        return None
