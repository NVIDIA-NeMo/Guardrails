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

The verdict is in the response headers; the body is the payload to forward.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.http.retry import _header

Surface = Literal["input", "output", "tool_call", "tool_result"]

# Where a REDACT is applied on each surface. NeMo has no rewrite channel for tool payloads, and
# a tool rail returning a transform is refused by the engine, so a tool REDACT blocks instead.
_SURFACE_TARGET: Dict[Surface, Optional[TransformTarget]] = {
    "input": TransformTarget.USER_MESSAGE,
    "output": TransformTarget.BOT_MESSAGE,
    "tool_call": None,
    "tool_result": None,
}

BLOCK_REASON = "Blocked by HiddenLayer AI Runtime Security"
REDACT_REASON = "Redacted by HiddenLayer AI Runtime Security"
FAIL_OPEN_REASON = "HiddenLayer AI Runtime Security unavailable; fail_open allowed the request"

_ACTION_HEADER = "HL-Runtime-Action"
_PROTECTION_MODE_HEADER = "HL-Runtime-Protection-Mode"
_ALLOW_ACTIONS = frozenset({"DETECT", "NONE"})


@dataclass(frozen=True)
class _Sent:
    """What the rail sent for evaluation."""

    checked_text: Optional[str] = None
    call_id: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Union[dict, str, None] = None
    content: Union[str, list, None] = None


@dataclass(frozen=True)
class _Evaluation:
    """A 200 from a NeMo-Guardrails evaluation route."""

    headers: Mapping[str, str]
    body: Any


class HiddenLayerActionError(Exception):
    """The response carries no action to apply. The message never quotes the response."""


class HiddenLayerVerdictError(Exception):
    """The action cannot be applied to what was sent. The message never quotes the response."""


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


def _redacted_text(surface: Surface, body: Any) -> Any:
    """The text the redacted payload carries, or None when the body does not hold one."""
    try:
        if surface == "input":
            return body["messages"][-1]["content"]
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _outcome(surface: Surface, evaluation: _Evaluation, sent: _Sent) -> RailOutcome:
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
    # A BLOCK body carries HiddenLayer's own refusal; NeMo renders its own through the flows.
    if action == "BLOCK":
        return RailOutcome.block(reason=BLOCK_REASON, metadata=metadata)
    if action != "REDACT":
        raise HiddenLayerActionError(f"HiddenLayer response has an unknown {_ACTION_HEADER}")

    target = _SURFACE_TARGET[surface]
    if target is None:
        # HiddenLayer promotes a REDACT on a tool evaluation to BLOCK, so this is the defensive path.
        return RailOutcome.block(reason=BLOCK_REASON, metadata=metadata)
    text = _redacted_text(surface, evaluation.body)
    if not isinstance(text, str):
        raise HiddenLayerVerdictError("HiddenLayer REDACT response has no redacted text")
    if text == sent.checked_text:
        raise HiddenLayerVerdictError("HiddenLayer REDACT response left the text unchanged")
    return RailOutcome.transform([(target, text)], reason=REDACT_REASON, metadata=metadata)
