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

"""Define provider payload projections and exact guarded message targets."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, ClassVar, Literal, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError, ValidationInfo, model_validator
from typing_extensions import Self

from nemoguardrails.server.experimental.provider.types import GuardedMessage, JsonObject, UnknownContentFieldPolicy

_UNKNOWN_CONTENT_FIELDS_CONTEXT = "unknown_content_fields"
_PROJECTION_COVERAGE_CONTEXT = "projection_coverage"


def _require_boolean(value: object) -> object:
    """Reject integer values where a strict boolean is required."""
    if not isinstance(value, bool):
        raise ValueError("value must be a boolean")
    return value


StrictFalse = Annotated[Literal[False], BeforeValidator(_require_boolean)]


class PayloadCapabilityProfile(str, Enum):
    """Identify a closed payload capability understood by the framework."""

    SINGLE_TEXT_V1 = "single_text.v1"


@dataclass(frozen=True, slots=True)
class ProjectionFieldCoverage:
    """Classify every reviewed field of one provider object."""

    guarded_fields: frozenset[str] = frozenset()
    constrained_fields: frozenset[str] = frozenset()
    opaque_fields: frozenset[str] = frozenset()
    local_extension_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Require each reviewed field to have exactly one classification."""
        classifications = (
            self.guarded_fields,
            self.constrained_fields,
            self.opaque_fields,
            self.local_extension_fields,
        )
        repeated = set()
        for index, fields in enumerate(classifications):
            for other_fields in classifications[index + 1 :]:
                repeated.update(fields & other_fields)
        if repeated:
            raise ValueError(f"Projection fields have multiple classifications: {sorted(repeated)}")

    @property
    def reviewed_fields(self) -> frozenset[str]:
        """Return every field accounted for by this coverage declaration."""
        return self.guarded_fields | self.constrained_fields | self.opaque_fields | self.local_extension_fields


@dataclass(frozen=True, slots=True)
class ProjectionModelContract:
    """Bind field coverage to one nested projection model."""

    model: type[BaseModel]
    coverage: ProjectionFieldCoverage
    source_schema: str | None = None

    def __post_init__(self) -> None:
        """Validate an explicitly declared provider source schema."""
        if self.source_schema is not None and not self.source_schema.strip():
            raise ValueError("A projection source schema must be non-empty when declared.")


@dataclass(frozen=True, slots=True)
class PayloadProjectionContract:
    """Describe the runtime semantics of one guarded payload projection."""

    projection_id: str
    direction: Literal["request", "response"]
    profile: PayloadCapabilityProfile
    root: ProjectionFieldCoverage
    content_models: tuple[ProjectionModelContract, ...] = ()

    def __post_init__(self) -> None:
        """Validate the projection identity and nested model declarations."""
        if not self.projection_id.strip():
            raise ValueError("A payload projection contract must have an identifier.")
        models = [entry.model for entry in self.content_models]
        if len(models) != len(set(models)):
            raise ValueError("A payload projection contract must describe each content model once.")


def validate_payload_projection_contract(
    model: type[BaseModel],
    expected_direction: Literal["request", "response"],
) -> PayloadProjectionContract:
    """Validate and return the semantic contract attached to a payload model."""

    contract = getattr(model, "projection_contract", None)
    if not isinstance(contract, PayloadProjectionContract):
        raise ValueError(f"Payload model {model.__name__!r} must declare a projection contract.")
    if contract.direction != expected_direction:
        raise ValueError(
            f"Payload model {model.__name__!r} declares {contract.direction!r}, expected {expected_direction!r}."
        )

    def field_names(projection_model: type[BaseModel]) -> frozenset[str]:
        """Return serialized field names for one projection model."""
        return frozenset(
            field.alias if field.alias is not None else name for name, field in projection_model.model_fields.items()
        )

    missing_root_fields = field_names(model) - contract.root.reviewed_fields
    if missing_root_fields:
        raise ValueError(
            f"Payload contract {contract.projection_id!r} omits root fields: {sorted(missing_root_fields)}"
        )
    for entry in contract.content_models:
        missing_fields = field_names(entry.model) - entry.coverage.reviewed_fields
        if missing_fields:
            raise ValueError(
                f"Payload contract {contract.projection_id!r} omits {entry.model.__name__} fields: "
                f"{sorted(missing_fields)}"
            )
    return contract


def guarded_schema_error(error: ValidationError, subject: str) -> str:
    """Describe one projection failure without echoing provider content."""

    issue = error.errors(include_url=False, include_context=False, include_input=False)[0]
    location = ".".join(str(part) for part in issue["loc"])
    location_text = f" at {location}" if location else ""
    return f"The {subject} does not match the guarded content schema{location_text}: {issue['msg']}."


class GuardedProjectionModel(BaseModel):
    """Provide common validation behavior for guarded payload projections."""

    model_config = ConfigDict(extra="allow", frozen=True)
    projection_contract: ClassVar[PayloadProjectionContract | None] = None

    @classmethod
    def validate_payload(
        cls,
        document: object,
        *,
        unknown_content_fields: UnknownContentFieldPolicy = UnknownContentFieldPolicy.ALLOW,
    ) -> Self:
        """Validate a provider payload with the requested unknown-field policy."""
        coverage = (
            {entry.model: entry.coverage for entry in cls.projection_contract.content_models}
            if cls.projection_contract is not None
            else {}
        )
        return cls.model_validate(
            document,
            context={
                _UNKNOWN_CONTENT_FIELDS_CONTEXT: unknown_content_fields,
                _PROJECTION_COVERAGE_CONTEXT: coverage,
            },
        )


class GuardedContentModel(GuardedProjectionModel):
    """Validate the guarded subset of a content-bearing provider object."""

    @model_validator(mode="after")
    def reject_unreviewed_fields(self, info: ValidationInfo) -> Self:
        """Reject unknown content fields when closed validation is requested."""
        context = info.context or {}
        if context.get(_UNKNOWN_CONTENT_FIELDS_CONTEXT) != UnknownContentFieldPolicy.FORBID:
            return self
        coverage_by_model = cast(
            dict[type[BaseModel], ProjectionFieldCoverage],
            context.get(_PROJECTION_COVERAGE_CONTEXT, {}),
        )
        coverage = coverage_by_model.get(type(self))
        opaque_fields = coverage.opaque_fields if coverage is not None else frozenset()
        unreviewed_fields = set(self.model_extra or {}) - opaque_fields
        if unreviewed_fields:
            fields = ", ".join(sorted(unreviewed_fields))
            raise ValueError(f"unreviewed content fields are forbidden: {fields}")
        return self


@dataclass(slots=True)
class GuardedMessageTarget:
    """Identify the exact provider message field guarded by the pipeline."""

    role: Literal["user", "assistant"]
    _object: JsonObject
    _member: str
    allows_replacement: bool

    @property
    def message(self) -> GuardedMessage:
        """Return the message currently stored at the target."""
        content = self._object[self._member]
        if not isinstance(content, str):
            raise TypeError("Guarded message content must be a string.")
        return GuardedMessage(role=self.role, content=content)

    def replace_content(self, content: str) -> None:
        """Replace guarded content when the contract permits modification."""
        if not self.allows_replacement:
            raise ValueError("The guarded provider content does not support replacement.")
        self._object[self._member] = content


@dataclass(frozen=True, slots=True)
class GuardedArraySelection:
    """Select one array member by provider field values."""

    match: tuple[tuple[str, str | int | bool], ...]
    cardinality: Literal["exactly_one"]

    def select(self, items: Sequence[object]) -> object:
        """Return the single array item matching the declared discriminator."""
        missing = object()
        matches = []
        for item in items:
            values = {
                field: item.get(field, missing) if isinstance(item, dict) else getattr(item, field, missing)
                for field, _ in self.match
            }
            if all(values[field] == expected for field, expected in self.match):
                matches.append(item)
        if len(matches) != 1:
            raise ValueError("Guarded provider content requires exactly one matching array member.")
        return matches[0]


@dataclass(frozen=True, slots=True)
class GuardedTextLocation:
    """Locate one guarded text member in a validated provider payload."""

    role: Literal["user", "assistant"]
    object_path: tuple[str | int | GuardedArraySelection, ...]
    member: str
    allows_replacement: bool
    replacement_blocked_by: str | None = None

    def locate(self, payload: JsonObject) -> GuardedMessageTarget:
        """Locate guarded text in the original provider payload."""
        value: object = payload
        for segment in self.object_path:
            if isinstance(segment, str) and isinstance(value, dict):
                value = cast(JsonObject, value)[segment]
            elif isinstance(segment, int) and isinstance(value, list):
                value = cast(list[object], value)[segment]
            elif isinstance(segment, GuardedArraySelection) and isinstance(value, list):
                value = segment.select(cast(list[object], value))
            else:
                raise TypeError("Guarded text location does not match the provider payload.")
        if not isinstance(value, dict):
            raise TypeError("Guarded text location must resolve to a provider object.")
        allows_replacement = self.allows_replacement and not (
            self.replacement_blocked_by is not None and bool(value.get(self.replacement_blocked_by))
        )
        return GuardedMessageTarget(
            self.role,
            cast(JsonObject, value),
            self.member,
            allows_replacement=allows_replacement,
        )

    def validate_projection(self, projection: BaseModel) -> None:
        """Require the projected location to resolve to text."""
        value: object = projection
        for segment in self.object_path:
            if isinstance(segment, str) and isinstance(value, BaseModel):
                value = getattr(value, segment)
            elif isinstance(segment, int) and isinstance(value, list):
                value = cast(list[object], value)[segment]
            elif isinstance(segment, GuardedArraySelection) and isinstance(value, list):
                value = segment.select(cast(list[object], value))
            else:
                raise ValueError("Guarded text location does not match the provider projection.")
        if not isinstance(value, BaseModel) or not isinstance(getattr(value, self.member, None), str):
            raise ValueError("Guarded text location does not resolve to projected text.")

    def matches_projection(self, projection: BaseModel) -> bool:
        """Return whether this location resolves to text in the projection."""

        def unwrap_root(value: object) -> object:
            """Unwrap nested Pydantic root models."""
            while isinstance(value, BaseModel) and type(value).__pydantic_root_model__:
                value = getattr(value, "root")
            return value

        try:
            value: object = projection
            for segment in self.object_path:
                value = unwrap_root(value)
                if isinstance(segment, str) and isinstance(value, BaseModel):
                    value = getattr(value, segment)
                elif isinstance(segment, int) and isinstance(value, list):
                    value = cast(list[object], value)[segment]
                elif isinstance(segment, GuardedArraySelection) and isinstance(value, list):
                    value = segment.select(cast(list[object], value))
                else:
                    return False
            value = unwrap_root(value)
            return isinstance(value, BaseModel) and isinstance(unwrap_root(getattr(value, self.member)), str)
        except (AttributeError, IndexError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class GuardedTextAlternatives:
    """Select one of several provider encodings for the same guarded text."""

    locations: tuple[GuardedTextLocation, ...]

    def _projection_location(self, projection: BaseModel) -> GuardedTextLocation:
        """Return the single location matching the projected representation."""
        matches = [location for location in self.locations if location.matches_projection(projection)]
        if len(matches) != 1:
            raise ValueError("Guarded text alternatives require exactly one matching representation.")
        return matches[0]

    def locate(self, projection: BaseModel, payload: JsonObject) -> GuardedMessageTarget:
        """Locate guarded text using the projected provider representation."""
        return self._projection_location(projection).locate(payload)

    def validate_projection(self, projection: BaseModel) -> None:
        """Require exactly one alternative to match the projection."""
        self._projection_location(projection)


class GuardedPayloadModel(GuardedProjectionModel):
    """Validate and locate guarded text in a provider payload projection."""

    guarded_text_location: ClassVar[GuardedTextLocation | GuardedTextAlternatives | None] = None

    @classmethod
    def has_guarded_text_binding(cls) -> bool:
        """Return whether the model declares how to locate guarded text."""
        return (
            cls.guarded_text_location is not None
            or cls.locate_guarded_message is not GuardedPayloadModel.locate_guarded_message
        )

    @model_validator(mode="after")
    def validate_guarded_text_target(self) -> Self:
        """Validate the declared guarded text location after model parsing."""
        if self.guarded_text_location is not None:
            self.guarded_text_location.validate_projection(self)
        return self

    def locate_guarded_message(self, payload: JsonObject) -> GuardedMessageTarget:
        """Locate the guarded message in the original provider payload."""
        if self.guarded_text_location is None:
            raise TypeError("The guarded payload model does not declare a guarded text location.")
        if isinstance(self.guarded_text_location, GuardedTextAlternatives):
            return self.guarded_text_location.locate(self, payload)
        return self.guarded_text_location.locate(payload)


class GuardedRequestModel(GuardedPayloadModel):
    """Validate a provider request and determine its response mode."""

    stream_selector_field: ClassVar[str | None] = None

    @classmethod
    def has_response_mode_binding(cls) -> bool:
        """Return whether the request declares how to detect streaming."""
        return cls.stream_selector_field is not None or cls.streams_response is not GuardedRequestModel.streams_response

    @property
    def streams_response(self) -> bool:
        """Return the provider response mode selected by this request."""
        if self.stream_selector_field is None:
            raise TypeError("The guarded request model does not declare a stream selector field.")
        return bool(getattr(self, self.stream_selector_field))
