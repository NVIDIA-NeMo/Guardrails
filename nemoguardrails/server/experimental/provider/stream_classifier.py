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

"""Classify provider stream events from generated semantic rules."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from nemoguardrails.server.experimental._json_payload import parse_json_object
from nemoguardrails.server.experimental.provider.payload import guarded_schema_error
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import (
    GuardedStreamEvent,
    StreamEventRole,
    StreamProjectionContract,
    UnsupportedProviderStream,
    validate_stream_event,
)

StreamMatchValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class StreamArraySelection:
    """Select stream content by provider discriminator fields."""

    match: tuple[tuple[str, StreamMatchValue], ...]
    cardinality: Literal["at_most_one", "exactly_one"]

    def select(self, items: Sequence[object]) -> object | None:
        matches = [
            item
            for item in items
            if all(
                type(value := (item.get(name) if isinstance(item, dict) else getattr(item, name, object())))
                is type(expected)
                and value == expected
                for name, expected in self.match
            )
        ]
        if len(matches) > 1 or (self.cardinality == "exactly_one" and not matches):
            raise ValueError(f"Stream text selection requires {self.cardinality.replace('_', ' ')} matching item.")
        return matches[0] if matches else None


StreamTextPath = tuple[str | int | StreamArraySelection, ...]


@dataclass(frozen=True, slots=True)
class StreamEventRule:
    """Declare validation and classification for one provider event shape."""

    shape: str
    model: type[BaseModel]
    role: StreamEventRole
    match: tuple[tuple[str, StreamMatchValue], ...] = ()
    required_fields: frozenset[str] = frozenset()
    text_path: StreamTextPath | None = None
    missing_text_role: StreamEventRole | None = None
    missing_text_shape: str | None = None
    missing_text_value: str | None = None

    def matches(self, payload: dict[str, Any]) -> bool:
        """Return whether a provider payload selects this event rule."""

        missing = object()

        def resolve(path: str) -> object:
            value: object = payload
            for segment in path.split("."):
                if not isinstance(value, dict):
                    return missing
                value = value.get(segment, missing)
            return value

        return self.required_fields <= payload.keys() and all(
            type(value := resolve(name)) is type(expected) and value == expected for name, expected in self.match
        )


@dataclass(frozen=True, slots=True)
class StreamClassifierDefinition:
    """Describe the event-shape semantics of one guarded provider stream."""

    subject: str
    contract: StreamProjectionContract
    rules: tuple[StreamEventRule, ...]
    sentinels: tuple[tuple[bytes, str], ...] = ()
    non_data_shape: str | None = None
    event_type_field: str | None = None
    require_event_type: bool = False


@dataclass(frozen=True, slots=True)
class StreamClassifier:
    """Classify provider events using validated stream rules."""

    subject: str
    contract: StreamProjectionContract
    rules: tuple[StreamEventRule, ...]
    sentinels: tuple[tuple[bytes, str], ...]
    non_data_shape: str | None
    event_type_field: str | None
    require_event_type: bool

    def classify_event(self, event: ServerSentEvent) -> GuardedStreamEvent:
        """Validate and classify one native provider event."""

        data = event.data
        if data is None:
            if self.non_data_shape is None:
                raise UnsupportedProviderStream(f"{self.subject} events must contain data.")
            return GuardedStreamEvent(self.non_data_shape, StreamEventRole.OPAQUE_METADATA)
        stripped = data.strip()
        for sentinel, shape in self.sentinels:
            if stripped == sentinel:
                return GuardedStreamEvent(shape, StreamEventRole.OPAQUE_METADATA)
        try:
            payload = parse_json_object(data)
            if self.event_type_field is not None:
                event_type = payload.get(self.event_type_field)
                if not isinstance(event_type, str):
                    raise UnsupportedProviderStream(
                        f"{self.subject} events must declare string field {self.event_type_field!r}."
                    )
                if self.require_event_type and event.event_type is None:
                    raise UnsupportedProviderStream(f"{self.subject} SSE event field is required.")
                if event.event_type is not None and event.event_type != event_type.encode("ascii"):
                    raise UnsupportedProviderStream(
                        f"{self.subject} SSE event field must match data field {self.event_type_field!r}."
                    )
            matching_rules = [candidate for candidate in self.rules if candidate.matches(payload)]
            if not matching_rules:
                raise UnsupportedProviderStream(f"{self.subject} event is not covered by the guarded profile.")
            if len(matching_rules) > 1:
                shapes = ", ".join(repr(rule.shape) for rule in matching_rules)
                raise UnsupportedProviderStream(f"{self.subject} event ambiguously matches {shapes}.")
            rule = matching_rules[0]
            projection = rule.model.model_validate(payload)
            text = _resolve_text(projection, rule.text_path)
            if text is None and rule.missing_text_role is not None:
                return GuardedStreamEvent(
                    rule.missing_text_shape or rule.shape,
                    rule.missing_text_role,
                    rule.missing_text_value,
                )
            if text is None and rule.role in {StreamEventRole.GUARDED_TEXT, StreamEventRole.TEXT_SNAPSHOT}:
                raise UnsupportedProviderStream(f"{self.subject} event {rule.shape!r} has no guarded text value.")
            return GuardedStreamEvent(rule.shape, rule.role, text)
        except UnsupportedProviderStream:
            raise
        except ValidationError as error:
            raise UnsupportedProviderStream(guarded_schema_error(error, self.subject)) from error
        except (IndexError, TypeError, UnicodeError, ValueError) as error:
            raise UnsupportedProviderStream(str(error)) from error


def _resolve_text(projection: BaseModel, path: StreamTextPath | None) -> str | None:
    if path is None:
        return None
    value: object = projection
    for segment in path:
        if isinstance(segment, str) and isinstance(value, BaseModel):
            value = getattr(value, segment)
        elif isinstance(segment, int) and isinstance(value, list):
            if segment < 0:
                raise ValueError("Stream event text paths cannot use negative indexes.")
            if segment >= len(value):
                return None
            value = value[segment]
        elif isinstance(segment, StreamArraySelection) and isinstance(value, list):
            value = segment.select(value)
        else:
            raise TypeError("Stream event text path does not match its projection.")
        if value is None:
            return None
    if not isinstance(value, str):
        raise TypeError("Stream event text must resolve to a string.")
    return value


def build_stream_classifier(definition: StreamClassifierDefinition) -> StreamClassifier:
    """Validate generated event rules and return their runtime classifier."""

    if not definition.subject:
        raise ValueError("A stream classifier definition must identify its provider event subject.")
    if not definition.rules:
        raise ValueError("A stream classifier definition must declare at least one event rule.")
    if definition.subject != definition.contract.projection_id:
        raise ValueError("A stream classifier subject must match its projection contract identifier.")
    if definition.require_event_type and definition.event_type_field is None:
        raise ValueError("A required SSE event field needs a data discriminator.")
    signatures: set[tuple[tuple[tuple[str, Any], ...], frozenset[str]]] = set()

    def validate_declared_event(shape: str, role: StreamEventRole, text: str | None = None) -> None:
        try:
            validate_stream_event(definition.contract, GuardedStreamEvent(shape, role, text))
        except UnsupportedProviderStream as error:
            raise ValueError(str(error)) from error

    if definition.non_data_shape is not None:
        validate_declared_event(definition.non_data_shape, StreamEventRole.OPAQUE_METADATA)
    for _, shape in definition.sentinels:
        validate_declared_event(shape, StreamEventRole.OPAQUE_METADATA)
    for rule in definition.rules:
        if not rule.shape:
            raise ValueError("A stream event rule must identify its provider shape.")
        if not rule.match and not rule.required_fields:
            raise ValueError(f"Stream event rule {rule.shape!r} must declare how it is selected.")
        signature = (rule.match, rule.required_fields)
        if signature in signatures:
            raise ValueError(f"Stream event rule {rule.shape!r} duplicates another selection.")
        signatures.add(signature)
        carries_text = rule.role in {StreamEventRole.GUARDED_TEXT, StreamEventRole.TEXT_SNAPSHOT}
        if carries_text != (rule.text_path is not None):
            raise ValueError(f"Stream event rule {rule.shape!r} has an inconsistent text role and path.")
        if rule.missing_text_role is not None and rule.text_path is None:
            raise ValueError(f"Stream event rule {rule.shape!r} cannot classify missing text without a text path.")
        if rule.missing_text_shape is not None and rule.missing_text_role is None:
            raise ValueError(f"Stream event rule {rule.shape!r} cannot rename missing text without classifying it.")
        if rule.missing_text_value is not None and rule.missing_text_role is None:
            raise ValueError(f"Stream event rule {rule.shape!r} cannot supply missing text without classifying it.")
        if (
            rule.missing_text_role in {StreamEventRole.GUARDED_TEXT, StreamEventRole.TEXT_SNAPSHOT}
            and rule.missing_text_value is None
        ):
            raise ValueError(f"Stream event rule {rule.shape!r} cannot classify missing text as content.")
        validate_declared_event(rule.shape, rule.role, "" if carries_text else None)
        if rule.missing_text_role is not None:
            validate_declared_event(
                rule.missing_text_shape or rule.shape,
                rule.missing_text_role,
                rule.missing_text_value,
            )
    return StreamClassifier(
        subject=definition.subject,
        contract=definition.contract,
        rules=definition.rules,
        sentinels=definition.sentinels,
        non_data_shape=definition.non_data_shape,
        event_type_field=definition.event_type_field,
        require_event_type=definition.require_event_type,
    )
