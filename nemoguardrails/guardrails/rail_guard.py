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

"""The engine's fail-closed error envelope, shared by every IORails rail.

A rail that fails unexpectedly blocks: the failure is recorded on the action span and
returned as ``RailResult(is_safe=False)`` with a redacted reason, so a rail bug or a
malformed payload cannot silently let content through.

The one exception is a failure carrying an upstream HTTP status. That propagates, so a
provider outage or rate limit reaches the client as the status the provider sent rather
than as a guardrail block — the two mean very different things to a caller, and only one
of them is worth retrying.

This policy belongs to the engine rather than to any individual rail, which is why it
lives here instead of in a rail base class. Both the manifest-driven rails and the
hand-written tool rails call it from their own ``except`` handler.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.guardrails.api_engine import APIEngineError
from nemoguardrails.guardrails.guardrails_types import RailResult, get_request_id
from nemoguardrails.guardrails.model_engine import ModelEngineError
from nemoguardrails.guardrails.telemetry import record_span_error
from nemoguardrails.llm.clients._errors import _redact_secrets

if TYPE_CHECKING:
    from opentelemetry.trace import Span

log = logging.getLogger(__name__)

# The exception types that can carry an upstream HTTP status on ``.status``.
# A rail that reaches its model through ``llm_call`` only ever sees ``LLMCallException``,
# which wraps everything ``generate_async`` raises; the two engine errors are what a rail
# calling ``EngineRegistry`` directly sees. ``APIEngineError`` leaves this tuple in the
# commit that deletes ``APIEngine``.
_STATUS_BEARING_ERRORS = (ModelEngineError, APIEngineError, LLMCallException)


def _upstream_http_status(exc: Exception) -> Optional[int]:
    """Return the upstream HTTP status *exc* carries, or None when it carries none."""
    if isinstance(exc, _STATUS_BEARING_ERRORS):
        return exc.status
    return None


def _blocked_reason_or_reraise(span: Optional["Span"], action_name: str, exc: Exception) -> str:
    """Record *exc* on *span* and return a redacted block reason, or re-raise on an HTTP status.

    The exception text is redacted once, up front, and that redacted form is what reaches the
    log on both paths. A provider error can carry a credential in its message — an API key in
    a request URL, an echoed bearer token — and logging it raw would defeat the redaction
    applied to the reason a few lines below (CWE-532).

    *exc* itself is propagated unmodified. The server maps the original exception to a
    response and sanitises the client-facing payload separately, so rewriting its message here
    would only cost an operator the real text in a traceback.

    Raises:
        Exception: *exc* itself, when it carries an upstream HTTP status.
    """
    record_span_error(span, exc)
    request_id = get_request_id()
    detail = _redact_secrets(str(exc))

    status = _upstream_http_status(exc)
    if status is not None:
        log.error("[%s] %s failed (HTTP %d): %s", request_id, action_name, status, detail)
        raise exc

    log.error("[%s] %s failed: %s", request_id, action_name, detail)
    return f"{action_name} error: {detail}"


def rail_error_result(span: Optional["Span"], action_name: str, exc: Exception) -> RailResult:
    """Map a failed tool rail to a blocking ``RailResult``, or re-raise on an HTTP status.

    Call this from the ``except`` handler of a tool rail's own ``try``. *exc* is recorded
    on *span* on both paths, so a propagated failure is still visible in the trace.

    Raises:
        Exception: *exc* itself, when it carries an upstream HTTP status.
    """
    return RailResult(is_safe=False, reason=_blocked_reason_or_reraise(span, action_name, exc))


def rail_error_outcome(span: Optional["Span"], action_name: str, exc: Exception) -> RailOutcome:
    """Map a failed compiled rail to a blocking ``RailOutcome``, or re-raise on an HTTP status.

    Call this from the ``except`` handler of a ``CompiledRail``'s own ``try``. *exc* is
    recorded on *span* on both paths, so a propagated failure is still visible in the trace.

    Raises:
        Exception: *exc* itself, when it carries an upstream HTTP status.
    """
    return RailOutcome.block(reason=_blocked_reason_or_reraise(span, action_name, exc))
