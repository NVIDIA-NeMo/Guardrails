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

"""Define the stream boundary implemented by provider adapters."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from nemoguardrails.server.experimental.provider.payload import ProjectionFieldCoverage
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent


class UnsupportedProviderStream(ValueError):
    """Report a provider event that the guarded stream cannot interpret."""


class StreamEventRole(str, Enum):
    """Describe how one provider event relates to guarded content."""

    GUARDED_TEXT = "guarded_text"
    TEXT_SNAPSHOT = "text_snapshot"
    OPAQUE_METADATA = "opaque_metadata"
    PROVIDER_ERROR = "provider_error"


class StreamCapabilityProfile(str, Enum):
    """Identify a closed streaming capability understood by the framework."""

    SINGLE_TEXT_DELTA_V1 = "single_text_delta.v1"


@dataclass(frozen=True, slots=True)
class StreamShapeCoverage:
    """Classify every reviewed native event shape in a provider stream."""

    guarded_shapes: frozenset[str] = frozenset()
    snapshot_shapes: frozenset[str] = frozenset()
    opaque_shapes: frozenset[str] = frozenset()
    provider_error_shapes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        classifications = (
            self.guarded_shapes,
            self.snapshot_shapes,
            self.opaque_shapes,
            self.provider_error_shapes,
        )
        repeated = set()
        for index, shapes in enumerate(classifications):
            for other_shapes in classifications[index + 1 :]:
                repeated.update(shapes & other_shapes)
        if repeated:
            raise ValueError(f"Stream shapes have multiple classifications: {sorted(repeated)}")

    @property
    def reviewed_shapes(self) -> frozenset[str]:
        """Return every shape deliberately covered by the stream contract."""

        return self.guarded_shapes | self.snapshot_shapes | self.opaque_shapes | self.provider_error_shapes


@dataclass(frozen=True, slots=True)
class StreamProjectionContract:
    """Describe the runtime semantics of one guarded stream projection."""

    projection_id: str
    profile: StreamCapabilityProfile
    shapes: StreamShapeCoverage
    fields: ProjectionFieldCoverage | None = None

    def __post_init__(self) -> None:
        if not self.projection_id.strip():
            raise ValueError("A stream projection contract must have an identifier.")
        if not self.shapes.reviewed_shapes:
            raise ValueError("A stream projection contract must classify at least one shape.")


@dataclass(frozen=True, slots=True)
class GuardedStreamEvent:
    """Describe one provider event at the Guardrails boundary."""

    shape: str
    role: StreamEventRole
    text: str | None = None

    def __post_init__(self) -> None:
        """Require text exactly for roles that carry generated content."""

        if not self.shape:
            raise ValueError("A guarded stream event must identify its provider shape.")
        carries_text = self.role in {StreamEventRole.GUARDED_TEXT, StreamEventRole.TEXT_SNAPSHOT}
        if carries_text != (self.text is not None):
            raise ValueError("Guarded text and snapshot events must carry text; other event roles must not.")


def validate_stream_event(contract: StreamProjectionContract, event: GuardedStreamEvent) -> None:
    """Require a classified event to match its declared shape semantics."""

    roles = {
        StreamEventRole.GUARDED_TEXT: contract.shapes.guarded_shapes,
        StreamEventRole.TEXT_SNAPSHOT: contract.shapes.snapshot_shapes,
        StreamEventRole.OPAQUE_METADATA: contract.shapes.opaque_shapes,
        StreamEventRole.PROVIDER_ERROR: contract.shapes.provider_error_shapes,
    }
    declared_role = next((role for role, shapes in roles.items() if event.shape in shapes), None)
    if declared_role is None:
        raise UnsupportedProviderStream(
            f"Provider stream shape {event.shape!r} is not declared by {contract.projection_id!r}."
        )
    if event.role is not declared_role:
        raise UnsupportedProviderStream(
            f"Provider stream shape {event.shape!r} is classified as {event.role.value!r}; "
            f"the contract declares {declared_role.value!r}."
        )


class ProviderStreamClassifier(Protocol):
    """Classify native events using declarative projection semantics."""

    contract: StreamProjectionContract

    def classify_event(self, event: ServerSentEvent) -> GuardedStreamEvent:
        """Validate and classify one provider event."""

        ...


class ProviderStreamHooks(Protocol):
    """Implement stateful protocol checks and provider-native error framing."""

    def observe_event(self, event: ServerSentEvent, classified: GuardedStreamEvent) -> None:
        """Observe one classified event and update protocol state."""

        ...

    def validate_end_of_stream(self) -> None:
        """Validate provider protocol state when the byte stream ends."""

        ...

    def encode_error(self, rendered_body: bytes) -> Sequence[bytes]:
        """Frame a proxy error and its provider-native body for the stream."""

        ...


class ProviderStreamAdapter(Protocol):
    """Expose the complete stream behavior consumed by the shared runtime."""

    contract: StreamProjectionContract

    def classify_event(self, event: ServerSentEvent) -> GuardedStreamEvent:
        """Validate, classify, and observe one provider event."""

        ...

    def validate_end_of_stream(self) -> None:
        """Validate provider protocol state when the byte stream ends."""

        ...

    def encode_error(self, rendered_body: bytes) -> Sequence[bytes]:
        """Frame a proxy error and its provider-native body for the stream."""

        ...


@dataclass(slots=True)
class ClassifiedStreamAdapter:
    """Compose a generated classifier with handwritten protocol hooks."""

    classifier: ProviderStreamClassifier
    hooks: ProviderStreamHooks

    @property
    def contract(self) -> StreamProjectionContract:
        """Return the classifier's stream projection contract."""

        return self.classifier.contract

    def classify_event(self, event: ServerSentEvent) -> GuardedStreamEvent:
        """Classify an event and expose it to stateful protocol hooks."""

        classified = self.classifier.classify_event(event)
        self.hooks.observe_event(event, classified)
        return classified

    def validate_end_of_stream(self) -> None:
        """Delegate end-of-stream validation to provider hooks."""

        self.hooks.validate_end_of_stream()

    def encode_error(self, rendered_body: bytes) -> Sequence[bytes]:
        """Delegate provider-native error framing to provider hooks."""

        return self.hooks.encode_error(rendered_body)


def create_classified_stream_adapter_factory(
    classifier: ProviderStreamClassifier,
    hooks_factory: Callable[[], ProviderStreamHooks],
) -> Callable[[], ProviderStreamAdapter]:
    """Create a zero-argument adapter factory for an endpoint declaration."""

    return lambda: ClassifiedStreamAdapter(classifier, hooks_factory())
