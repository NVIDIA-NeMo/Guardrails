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

"""Base class for IORails tool-calling rails.

Tool rails are local structural/schema validators. Unlike ``RailAction`` they make
no LLM or API call, render no prompt, and need no model: they take already-normalized
tool data (a ``Toolset``, the model's ``ToolCall`` list, or incoming ``ToolResult``
objects — all produced by the engine adapter) and return a ``RailResult``.
``requires_model`` is therefore ``False`` and the only collaborator is an optional
tracer for spans.

Subclasses set :attr:`ToolRailAction.action_name` and implement an async ``run`` with
their own typed inputs (the inputs differ per rail), performing the check through
:meth:`ToolRailAction._guarded` so every rail gets a consistent action span and turns
an unexpected error into a blocking result rather than letting it propagate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from nemoguardrails.guardrails.guardrails_types import RailResult, get_request_id
from nemoguardrails.guardrails.telemetry import action_span, record_span_error

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

log = logging.getLogger(__name__)


class ToolRailAction:
    """Base for the local, model-free tool-calling rails (tool-call and tool-result)."""

    action_name: str
    requires_model: bool = False

    def __init__(self, tracer: Optional["Tracer"] = None) -> None:
        """Store the optional tracer used to emit the action span."""
        self._tracer = tracer

    def _guarded(self, check: Callable[[], RailResult]) -> RailResult:
        """Run *check* inside an action span, converting any error into a block.

        Mirrors ``RailAction``'s error contract: an unexpected exception is recorded
        on the span and returned as ``RailResult(is_safe=False, ...)`` so a malformed
        input or a rail bug fails closed rather than crashing the request.
        """
        with action_span(self._tracer, self.action_name) as span:
            try:
                return check()
            except Exception as e:
                record_span_error(span, e)
                log.error("[%s] %s failed: %s", get_request_id(), self.action_name, e)
                return RailResult(is_safe=False, reason=f"{self.action_name} error: {e}")
