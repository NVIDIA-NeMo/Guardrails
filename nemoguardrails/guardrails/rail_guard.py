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

A rail that fails blocks, with a redacted reason, so a rail bug cannot let content through.
A failure carrying an upstream HTTP status propagates instead, so a provider outage reaches
the client as the provider's status rather than as a guardrail decision — the two mean very
different things to a caller, and only one is worth retrying.

The policy is the engine's, not any individual rail's, which is why it lives here rather
than in a base class. Every rail calls it from its own ``except`` handler.
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

# Exception types carrying an upstream HTTP status on ``.status``. A rail reaching its model
# through ``llm_call`` sees only ``LLMCallException``, which wraps everything
# ``generate_async`` raises; the engine errors are what a direct ``EngineRegistry`` caller sees.
_STATUS_BEARING_ERRORS = (ModelEngineError, APIEngineError, LLMCallException)


def _upstream_http_status(exc: Exception) -> Optional[int]:
    """Return the upstream HTTP status *exc* carries, or None when it carries none."""
    if isinstance(exc, _STATUS_BEARING_ERRORS):
        return exc.status
    return None


def _blocked_reason_or_reraise(span: Optional["Span"], action_name: str, exc: Exception) -> str:
    """Record *exc* on *span* and return a redacted block reason, or re-raise on an HTTP status.

    The text is redacted once and that form is what reaches the log on both paths: a provider
    error can carry a credential in its message (CWE-532). *exc* itself propagates unmodified,
    so an operator still sees the real text in a traceback.

    The span error is recorded only on the blocking path. Callers run inside ``action_span``,
    which records anything escaping it, so recording here too would double up.

    Raises:
        Exception: *exc* itself, when it carries an upstream HTTP status.
    """
    request_id = get_request_id()
    detail = _redact_secrets(str(exc))

    status = _upstream_http_status(exc)
    if status is not None:
        log.error("[%s] %s failed (HTTP %d): %s", request_id, action_name, status, detail)
        raise exc

    record_span_error(span, exc)
    log.error("[%s] %s failed: %s", request_id, action_name, detail)
    return f"{action_name} error: {detail}"


def rail_error_result(span: Optional["Span"], action_name: str, exc: Exception) -> RailResult:
    """Map a failed tool rail to a blocking ``RailResult``, or re-raise on an HTTP status.

    Call this from the ``except`` handler of a tool rail's own ``try``.

    Raises:
        Exception: *exc* itself, when it carries an upstream HTTP status.
    """
    return RailResult(is_safe=False, reason=_blocked_reason_or_reraise(span, action_name, exc))


def rail_error_outcome(span: Optional["Span"], action_name: str, exc: Exception) -> RailOutcome:
    """Map a failed compiled rail to a blocking ``RailOutcome``, or re-raise on an HTTP status.

    Call this from the ``except`` handler of a ``CompiledRail``'s own ``try``.

    Raises:
        Exception: *exc* itself, when it carries an upstream HTTP status.
    """
    return RailOutcome.block(reason=_blocked_reason_or_reraise(span, action_name, exc))
