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

"""Map a HiddenLayer evaluation response to a RailOutcome.

The verdict is in the response headers; the body is the check to forward.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.library.hiddenlayer.errs import (
    HiddenLayerActionError,
    HiddenLayerUnknownActionError,
    HiddenLayerVerdictError,
)

Surface = Literal["input", "output", "tool_call", "tool_result"]

# Where a REDACT is applied on each surface, and the field the redacted text comes back in. The
# tool surfaces are absent because NeMo has no rewrite channel for tool payloads, and a tool rail
# returning a transform is refused by the engine, so a tool REDACT blocks instead.
_REDACTABLE: Dict[Surface, Tuple[TransformTarget, str]] = {
    "input": (TransformTarget.USER_MESSAGE, "user_message"),
    "output": (TransformTarget.BOT_MESSAGE, "bot_message"),
}

BLOCK_REASON = "Blocked by HiddenLayer AI Runtime Security"
REDACT_REASON = "Redacted by HiddenLayer AI Runtime Security"
FAIL_OPEN_REASON = "HiddenLayer AI Runtime Security unavailable; fail_open allowed the request"

log = logging.getLogger(__name__)

_ACTION_HEADER = "HL-Runtime-Action"
_PROTECTION_MODE_HEADER = "HL-Runtime-Protection-Mode"
_ALLOW_ACTIONS = frozenset({"DETECT", "NONE"})


@dataclass(frozen=True)
class _Evaluation:
    """A 200 from a NeMo-Guardrails evaluation route."""

    headers: Mapping[str, str]
    body: Any


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    """The header's value, matched without regard to case, or None when it is absent."""
    lowered = name.lower()
    return next((value for key, value in headers.items() if key.lower() == lowered), None)


def _action(headers: Mapping[str, str]) -> str:
    action = _header(headers, _ACTION_HEADER)
    if action is None:
        raise HiddenLayerActionError(f"HiddenLayer response has no {_ACTION_HEADER} header")
    return action


def _protection_mode(headers: Mapping[str, str]) -> Optional[str]:
    """The protection mode as sent.

    It is never checked against an enum: the contract may add modes, and refusing an unknown one
    would turn an additive server change into a rail outage.
    """
    return _header(headers, _PROTECTION_MODE_HEADER)


def _metadata(action: str, headers: Mapping[str, str]) -> Dict[str, Any]:
    """The verdict, and nothing the evaluated or returned payload carries."""
    return {"action": action, "protection_mode": _protection_mode(headers)}


def _outcome(surface: Surface, evaluation: _Evaluation, checked_text: Optional[str]) -> RailOutcome:
    """Map an evaluation to the rail's outcome on *surface*.

    Raises:
        HiddenLayerActionError: The action header is missing or not one the rail knows, so there
            is no verdict to apply.
        HiddenLayerVerdictError: A REDACT on a text surface carries no usable redacted text.
            HiddenLayer can answer REDACT with the text unchanged, which counts as unusable.
    """
    action = _action(evaluation.headers)
    metadata = _metadata(action, evaluation.headers)

    if action in _ALLOW_ACTIONS:
        return RailOutcome.allow(metadata=metadata)
    if action == "BLOCK":
        return RailOutcome.block(reason=BLOCK_REASON, metadata=metadata)
    if action != "REDACT":
        # The server decided something this rail cannot carry out. Allowing it would let a
        # restrictive action the contract adds later pass unapplied, so it blocks like any
        # other action the rail cannot apply.
        raise HiddenLayerUnknownActionError(f"HiddenLayer response has an unknown {_ACTION_HEADER}")

    redactable = _REDACTABLE.get(surface)
    if redactable is None:
        # HiddenLayer promotes a REDACT on a tool evaluation to BLOCK, so this is the defensive path.
        return RailOutcome.block(reason=BLOCK_REASON, metadata=metadata)
    target, field = redactable
    if not isinstance(checked_text, str):
        # The evaluated turn is content parts, and a transform replaces the whole content with
        # the redacted string, dropping every non-text part with it. HiddenLayer flattens the
        # text to evaluate it, so there is no part-wise rewrite to apply.
        log.warning("HiddenLayer %s REDACT cannot be applied to content parts; blocking.", surface)
        return RailOutcome.block(reason=BLOCK_REASON, metadata=metadata)
    text = evaluation.body.get(field) if isinstance(evaluation.body, dict) else None
    if not isinstance(text, str):
        raise HiddenLayerVerdictError("HiddenLayer REDACT response has no redacted text")
    if text == checked_text:
        raise HiddenLayerVerdictError("HiddenLayer REDACT response left the text unchanged")
    return RailOutcome.transform([(target, text)], reason=REDACT_REASON, metadata=metadata)
