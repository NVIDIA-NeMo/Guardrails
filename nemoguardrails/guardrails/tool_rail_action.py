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

"""Base class for IORails tool-calling rails: local structural validators, no model call."""

# Subclasses set ``action_name`` and implement an async ``run`` over their own typed inputs,
# checking through ``_guarded`` so each gets an action span and turns an unexpected error into
# a blocking result rather than letting it propagate.

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Optional

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.rail_guard import rail_error_outcome
from nemoguardrails.guardrails.telemetry import action_span
from nemoguardrails.rails.llm.options import ToolViolation

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

log = logging.getLogger(__name__)

# Metadata is the only channel out of an action, so a tool rail's violations travel in it.
TOOL_VIOLATIONS_KEY = "tool_violations"


def violations_outcome(violations: Sequence[ToolViolation]) -> RailOutcome:
    """Allow when nothing was violated, else block once, with the first violation's reason."""
    if not violations:
        return RailOutcome.allow()
    return RailOutcome.block(
        reason=violations[0].reason,
        metadata={TOOL_VIOLATIONS_KEY: [violation.model_dump(mode="json") for violation in violations]},
    )


def reported_violations(outcome: RailOutcome) -> tuple[ToolViolation, ...]:
    """The violations a tool rail put in *outcome*, read back from its metadata."""
    return tuple(ToolViolation.model_validate(violation) for violation in outcome.metadata.get(TOOL_VIOLATIONS_KEY, ()))


class ToolRailAction:
    """Base for the local, model-free tool-calling rails (tool-call and tool-result)."""

    action_name: str
    requires_model: bool = False

    def __init__(self, tracer: Optional["Tracer"] = None) -> None:
        """Store the optional tracer used to emit the action span."""
        self._tracer = tracer

    def _guarded(self, check: Callable[[], RailOutcome]) -> RailOutcome:
        """Run *check* inside an action span, converting a failure into a block.

        Shares the engine's fail-closed contract with every other rail through
        :func:`~nemoguardrails.guardrails.rail_guard.rail_error_outcome`, so a malformed
        input or a rail bug fails closed rather than crashing the request.

        HTTP Errors from downstream calls propagate to the client.

        Raises:
            Exception: whatever *check* raised, when it carries an upstream HTTP status.
        """
        with action_span(self._tracer, self.action_name) as span:
            try:
                return check()
            except Exception as e:
                return rail_error_outcome(span, self.action_name, e)
