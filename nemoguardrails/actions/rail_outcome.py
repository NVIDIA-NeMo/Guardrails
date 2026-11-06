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

``RailOutcome`` is the canonical, backend-agnostic verdict a rail produces.
Each engine or Colang flow renders the consequence from the decision, its
configuration, and any explicitly supported presentation metadata. For example,
a consumer may use ``metadata["refusal_message"]`` as a localized refusal
override; unsupported metadata must not be assumed to affect rendering.

Fields:
- ``decision``: ALLOW / BLOCK / TRANSFORM.
- ``reason``: optional neutral, human-readable explanation.
- ``metadata``: evidence the decision is based on (policy violations,
  categories, scores, or backend-specific details) and supported presentation
  hints. The top-level mapping is copied at construction and is never
  load-bearing for the decision itself. Consumers own serialization, disclosure,
  and the interpretation of presentation hints.
- ``transforms``: for TRANSFORM, which conversation variables are rewritten
  and to what. Transform outcomes are non-streaming; streaming output paths
  treat the check as non-blocking and do not apply the rewrite.
- ``failed``: the rail raised instead of returning a verdict, so this BLOCK is
  the engine's fail-closed envelope rather than the rail's own decision.

Exception types and Colang event/context-update channels remain the consumer's
responsibility. Presentation metadata does not change the verdict: a BLOCK
always means "stop"; there is no soft-block flag.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["RailDecision", "TransformTarget", "TransformSpec", "RailOutcome", "require_rail_outcome"]


class RailDecision(str, Enum):
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", TransformTarget(self.target))
        if not isinstance(self.text, str):
            raise TypeError("transform text must be a string")


@dataclass(frozen=True, slots=True)
class RailOutcome:
    """The engine-neutral verdict of one rail check.

    ``transforms`` is non-empty if and only if ``decision`` is TRANSFORM. Each
    engine renders the consequence of a BLOCK its own way; this object does not
    determine how presentation metadata is rendered.
    """

    decision: RailDecision
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    transforms: tuple[TransformSpec, ...] = ()
    failed: bool = False
    __hash__ = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", RailDecision(self.decision))
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata = dict(self.metadata)
        if not all(isinstance(key, str) for key in metadata):
            raise TypeError("metadata keys must be strings")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if bool(self.transforms) != (self.decision is RailDecision.TRANSFORM):
            raise ValueError("transforms must be non-empty if and only if decision is TRANSFORM")
        if not all(isinstance(spec, TransformSpec) for spec in self.transforms):
            raise TypeError("transforms must contain only TransformSpec instances")
        targets = [spec.target for spec in self.transforms]
        if len(targets) != len(set(targets)):
            raise ValueError("transforms must not repeat the same TransformTarget")
        if self.failed and self.decision is not RailDecision.BLOCK:
            raise ValueError("failed must be set only on a BLOCK decision")

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
    def allow(
        cls,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RailOutcome":
        return cls(decision=RailDecision.ALLOW, reason=reason, metadata={} if metadata is None else metadata)

    @classmethod
    def block(
        cls,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RailOutcome":
        return cls(decision=RailDecision.BLOCK, reason=reason, metadata={} if metadata is None else metadata)

    @classmethod
    def failure(
        cls,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RailOutcome":
        """A block the engine synthesized because the rail raised instead of deciding."""
        return cls(
            decision=RailDecision.BLOCK,
            reason=reason,
            metadata={} if metadata is None else metadata,
            failed=True,
        )

    @classmethod
    def transform(
        cls,
        rewrites: Sequence[tuple[TransformTarget, str]],
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RailOutcome":
        return cls(
            decision=RailDecision.TRANSFORM,
            reason=reason,
            transforms=tuple(TransformSpec(target=target, text=text) for target, text in rewrites),
            metadata={} if metadata is None else metadata,
        )


def require_rail_outcome(result: object) -> RailOutcome:
    if not isinstance(result, RailOutcome):
        raise TypeError(f"Rail action must return RailOutcome, got {type(result).__name__}")
    return result
