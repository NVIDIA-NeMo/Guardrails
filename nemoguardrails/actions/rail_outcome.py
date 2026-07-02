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

"""The engine-neutral outcome of a single rail check.

``RailOutcome`` is the canonical, backend-agnostic verdict a rail produces. It
carries only the DECISION and neutral evidence, never how a consequence is
rendered: which exception to raise, which bot intent or refusal message to
emit, and how to localize it are presentation concerns that each engine (the
Colang flow, IORails) decides for itself from this outcome plus its config.

Fields:
- ``decision``: ALLOW / BLOCK / TRANSFORM.
- ``reason``: optional neutral, human-readable explanation.
- ``metadata``: neutral evidence the decision is based on (policy violations,
  categories, scores). Never load-bearing for the decision itself.
- ``transforms``: for TRANSFORM, which conversation variables are rewritten
  and to what. Transform outcomes are non-streaming; streaming output bypasses
  observe them as non-blocking and do not apply rewrites.

Deliberately NOT here (they are rendering, not decision): exception type,
refusal intent/message, language, and Colang event/context-update channels.
A BLOCK always means "stop"; there is no soft-block flag.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RailDecision(Enum):
    """The three mutually exclusive things a rail can decide."""

    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


class TransformTarget(str, Enum):
    """The conversation variable a transform rewrites."""

    USER_MESSAGE = "user_message"
    BOT_MESSAGE = "bot_message"
    RELEVANT_CHUNKS = "relevant_chunks"


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """A rewrite of a single conversation variable."""

    target: TransformTarget
    text: str


@dataclass(frozen=True, slots=True)
class RailOutcome:
    """The engine-neutral verdict of one rail check.

    ``transforms`` is non-empty if and only if ``decision`` is TRANSFORM. Each
    engine renders the consequence of a BLOCK its own way; this object does not
    encode it.
    """

    decision: RailDecision
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    transforms: tuple[TransformSpec, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.transforms) != (self.decision is RailDecision.TRANSFORM):
            raise ValueError("transforms must be non-empty if and only if decision is TRANSFORM")

    @property
    def is_blocked(self) -> bool:
        """The single field both engines read to gate; rendering is theirs."""
        return self.decision is RailDecision.BLOCK

    @property
    def is_transform(self) -> bool:
        return self.decision is RailDecision.TRANSFORM

    @property
    def transform_text(self) -> dict[str, str]:
        return {spec.target.value: spec.text for spec in self.transforms}

    @classmethod
    def allow(cls, *, reason: str | None = None, **metadata: Any) -> "RailOutcome":
        return cls(decision=RailDecision.ALLOW, reason=reason, metadata=dict(metadata))

    @classmethod
    def block(cls, *, reason: str | None = None, **metadata: Any) -> "RailOutcome":
        return cls(decision=RailDecision.BLOCK, reason=reason, metadata=dict(metadata))

    @classmethod
    def transform(
        cls,
        rewrites: Sequence[tuple[TransformTarget, str]],
        *,
        reason: str | None = None,
        **metadata: Any,
    ) -> "RailOutcome":
        return cls(
            decision=RailDecision.TRANSFORM,
            reason=reason,
            transforms=tuple(TransformSpec(target=target, text=text) for target, text in rewrites),
            metadata=dict(metadata),
        )


def require_rail_outcome(result: object) -> RailOutcome:
    if not isinstance(result, RailOutcome):
        raise TypeError(f"Output rail action must return RailOutcome, got {type(result).__name__}")
    return result
