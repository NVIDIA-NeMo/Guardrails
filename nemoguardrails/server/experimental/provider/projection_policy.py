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

"""Field and object policies for provider payload projections.

Model declarations provide field coverage, guarded text locations, and
exported contract schemas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, ClassVar, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, ValidationInfo, model_validator
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema
from typing_extensions import Self

from nemoguardrails.server.experimental.provider.payload import (
    GuardedArraySelection,
    GuardedTextLocation,
    PayloadCapabilityProfile,
    PayloadProjectionContract,
    ProjectionFieldCoverage,
    ProjectionModelContract,
    unknown_content_field_policy,
)
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy

EXTENSION = "x-nemo-guardrails"
CONTRACT_VERSION = "1.0.0-alpha.1"

# Formats from guard-contract.schema.json, checked when policy is declared.
_REASON = re.compile(r"(core_capability|provider_integrity|projection_policy)\.[a-z][a-z0-9_]*")
_SOURCE_COMPONENT = re.compile(r"[A-Za-z_][A-Za-z0-9._-]*")


def _check_reason(reason: str) -> None:
    """Require a structured reason such as core_capability.tool_content."""
    if not _REASON.fullmatch(reason):
        raise ValueError(f"Reason {reason!r} must be a structured reason such as 'core_capability.tool_content'")


@dataclass(frozen=True)
class ArrayVariant:
    """Identify one reviewed object branch in a heterogeneous array.

    Match keys must be required singleton Literal fields of the owning model.
    Cardinality applies when a text binding selects this branch from an array,
    not when the object is validated in isolation. Non-text branches use 'any'.
    The same declaration describes selection in exported guard contracts.
    """

    match: tuple[tuple[str, str | int | bool], ...]
    cardinality: Literal["any", "at_most_one", "exactly_one"]

    def __post_init__(self) -> None:
        """Reject ambiguous or unrepresentable selector declarations."""
        if not self.match or len(dict(self.match)) != len(self.match):
            raise ValueError("Array variant match must have unique named discriminators")
        if any(not isinstance(key, str) or not key or type(value) not in (str, int, bool) for key, value in self.match):
            raise ValueError("Array variant match requires names and string, integer, or boolean values")
        if self.cardinality not in {"any", "at_most_one", "exactly_one"}:
            raise ValueError("Unsupported array variant cardinality")


@dataclass(frozen=True)
class ObjectPolicy:
    """Describe the reviewed boundary of one JSON object.

    Attributes:
        source: Provider component schema name, without a JSON Pointer prefix.
            This is provenance metadata, not a schema lookup or validation step.
        opaque: Reviewed passthrough fields not declared as Pydantic fields.
            Names must be unique and must not overlap declared fields.
        unknown_fields: How unreviewed members are handled. "forbid" always
            rejects them. "configurable" rejects them unless validation
            explicitly allows unknown content fields, which is reserved for
            trusted configuration. Members listed in opaque are reviewed.
        variant: Optional discriminator and selection cardinality when this
            model is a branch of a guarded heterogeneous array.

    This metadata does not select guardrails, a deployment configuration, or a
    capability profile. Field-specific policy belongs on the field annotations.
    """

    source: str | None = None
    opaque: tuple[str, ...] = ()
    unknown_fields: Literal["forbid", "configurable"] = "forbid"
    variant: ArrayVariant | None = None

    def __post_init__(self) -> None:
        """Reject values the exported contract format cannot represent."""
        if self.source is not None and not _SOURCE_COMPONENT.fullmatch(self.source):
            raise ValueError(f"Source {self.source!r} must be a bare component schema name")
        if any(not name or name == "*" for name in self.opaque):
            raise ValueError("Opaque fields must be named; wildcards are not supported")


class PolicyModel(BaseModel):
    """Require declared fields to carry guard policy metadata.

    Combine this base with the appropriate runtime projection base. At subclass
    creation it checks field classifications, opaque inventory overlap, disabled
    field defaults, and unsupported aliases. At validation it enforces the
    object's unknown-field policy. It does not prove upstream field coverage or
    serialize arbitrary custom validators into the exported schema.
    """

    # Unknown members are kept so the object policy, not Pydantic, decides them.
    model_config = ConfigDict(extra="allow")
    policy: ClassVar[ObjectPolicy] = ObjectPolicy()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Check local policy consistency after Pydantic has assembled fields."""
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_config.get("extra") != "allow":
            raise ValueError(f"{cls.__name__}: ObjectPolicy requires extra='allow' to validate unreviewed members")
        if len(set(cls.policy.opaque)) != len(cls.policy.opaque):
            raise ValueError(f"{cls.__name__}: duplicate opaque fields")
        overlap = set(cls.policy.opaque) & cls.model_fields.keys()
        if overlap:
            raise ValueError(f"{cls.__name__}: opaque overlaps declared fields: {sorted(overlap)}")
        for name, field in cls.model_fields.items():
            metadata = field_policy(field)
            if metadata.get("classification") not in {"guarded", "constrained", "opaque"}:
                raise ValueError(f"{cls.__name__}.{name}: missing field policy")
            if field.alias or field.validation_alias or field.serialization_alias:
                raise ValueError(f"{cls.__name__}.{name}: aliases are not supported")
            if metadata.get("gate") == "disabled" and (
                field.annotation is not type(None) or field.is_required() or field.default is not None
            ):
                raise ValueError(f"{cls.__name__}.{name}: disabled fields must be optional and null-only")
        if cls.policy.variant is not None:
            for name, expected in cls.policy.variant.match:
                field = cls.model_fields.get(name)
                if (
                    field is None
                    or not field.is_required()
                    or get_origin(field.annotation) is not Literal
                    or len(get_args(field.annotation)) != 1
                    or type(get_args(field.annotation)[0]) is not type(expected)
                    or get_args(field.annotation)[0] != expected
                    or field_policy(field).get("classification") != "constrained"
                ):
                    raise ValueError(
                        f"{cls.__name__}.{name}: variant match requires a required matching constrained Literal"
                    )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Attach policy using model identity, independent of schema display titles.

        Keep Pydantic's references and validation assertions. Object closure
        describes default validation; trusted overrides and case-alias checks
        remain explicit guard annotations.
        """
        node = dict(handler.resolve_ref_schema(handler(core_schema)))
        if node.get("type") != "object" or "properties" not in node:
            raise ValueError(f"{cls.__name__}: ObjectPolicy requires an object schema")
        metadata = dict(node.get(EXTENSION, {}))
        if cls.policy.source:
            metadata["source"] = "#/components/schemas/" + cls.policy.source
        node["additionalProperties"] = False
        metadata["unknown_fields"] = cls.policy.unknown_fields
        metadata["reject_case_aliases"] = True
        if cls.policy.variant is not None:
            metadata["variant"] = {
                "match": dict(cls.policy.variant.match),
                "cardinality": cls.policy.variant.cardinality,
            }
        properties = {
            **node["properties"],
            **{name: {EXTENSION: {"classification": "opaque"}} for name in cls.policy.opaque},
        }
        node["properties"] = dict(sorted(properties.items()))
        node[EXTENSION] = metadata
        return node

    @model_validator(mode="after")
    def reject_unreviewed_fields(self, info: ValidationInfo) -> Self:
        """Reject members outside the reviewed fields unless policy allows them.

        A case variant of a reviewed name, such as ``Tools``, is always
        rejected: some providers match member names case-insensitively, so it
        could reach a field this policy disables or inspects.
        """
        reviewed = {name.casefold(): name for name in (*type(self).model_fields, *self.policy.opaque)}
        unreviewed = sorted(set(self.model_extra or {}) - set(self.policy.opaque))
        for name in unreviewed:
            reviewed_name = reviewed.get(name.casefold())
            if reviewed_name is not None:
                raise ValueError(f"field {name!r} differs from reviewed field {reviewed_name!r} only by case")
        allowed = (
            self.policy.unknown_fields == "configurable"
            and unknown_content_field_policy(info) == UnknownContentFieldPolicy.ALLOW
        )
        if unreviewed and not allowed:
            raise ValueError(f"unreviewed fields are forbidden: {', '.join(unreviewed)}")
        return self


def field_policy(field: FieldInfo) -> dict[str, Any]:
    """Read a field's guard metadata, returning an empty mapping if absent.

    The returned mapping may belong to the field; callers must not mutate it.
    """
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    metadata = extra.get(EXTENSION)
    return cast(dict[str, Any], metadata) if isinstance(metadata, dict) else {}


# Runtime constraints that declarations may use. Export separately rejects any
# schema keyword the guard contract cannot carry. JSON Schema has no keyword for
# Python strictness, so `strict` is enforced at runtime only.
_FIELD_CONSTRAINTS = frozenset({"min_length", "max_length", "pattern", "ge", "le", "strict"})


def _field(metadata: dict[str, Any], constraints: dict[str, Any]) -> Any:
    """Build Annotated field metadata while keeping defaults in assignments."""
    if {"default", "default_factory", "alias"} & constraints.keys():
        raise ValueError("Declare defaults explicitly on the field; aliases are not supported")
    unsupported = constraints.keys() - _FIELD_CONSTRAINTS
    if unsupported:
        raise ValueError(f"Unsupported Pydantic constraints: {sorted(unsupported)}")
    return Field(json_schema_extra={EXTENSION: metadata}, **constraints)


def guarded(
    role: Literal["user", "assistant"] | None = None,
    *,
    replaceable: bool | None = None,
    blocked_by: str | None = None,
    replacement_reason: str | None = None,
    **constraints: Any,
) -> Any:
    """Mark guarded content or a container leading to guarded content.

    Args:
        role: Text subject role. Omit for structural container fields.
        replaceable: Whether the subject permits replacement. The buffered
            binding defaults to read-only when this is not declared.
        blocked_by: Sibling field whose non-empty value prevents replacement.
        replacement_reason: Structured reason for the replacement restriction.
        **constraints: min_length, max_length, pattern, ge, le, or strict.
            Format support is checked separately when exporting a complete
            guard contract; strictness has no JSON Schema keyword.

    Use inside Annotated and put defaults on the field assignment. Declaring
    replacement eligibility does not enable runtime replacement support.
    """
    metadata: dict[str, Any] = {"classification": "guarded"}
    if role is None and any(value is not None for value in (replaceable, blocked_by, replacement_reason)):
        raise ValueError("Text replacement policy requires a subject role")
    if role is not None:
        subject: dict[str, Any] = {"kind": "text", "role": role}
        if replaceable is not None:
            subject["replaceable"] = replaceable
        if blocked_by is not None:
            subject["replacement_blocked_by"] = blocked_by
        if replacement_reason is not None:
            _check_reason(replacement_reason)
            subject["replacement_reason"] = replacement_reason
        metadata["subject"] = subject
    return _field(metadata, constraints)


def constrained(*, reason: str | None = None, **constraints: Any) -> Any:
    """Mark a field restricted by its type and optional Pydantic constraints.

    The optional reason explains the restriction in the exported contract.
    This helper adds no restriction by itself; the annotation and constraints
    must express it. Defaults belong on the field assignment.
    """
    metadata: dict[str, Any] = {"classification": "constrained"}
    if reason is not None:
        _check_reason(reason)
        metadata["reason"] = reason
    return _field(metadata, constraints)


def disabled(reason: str, *, extension: bool = False) -> Any:
    """Declare an unsupported feature as an omittable, null-only field.

    Use field: Annotated[None, disabled(reason)] = None. Non-null values are
    rejected by the field type, and PolicyModel checks this declaration.
    A nonblank structured reason explains the restriction. Set extension for a
    locally recognized field outside the pinned provider schema.
    """
    _check_reason(reason)
    metadata: dict[str, Any] = {"classification": "constrained", "gate": "disabled", "reason": reason}
    if extension:
        metadata["extension"] = True
    return _field(metadata, {})


def opaque() -> Any:
    """Mark a declared field as provider-owned data outside guarded content.

    Use field: Annotated[Any, opaque()] = None for an unconstrained optional
    value. ObjectPolicy.opaque is the alternative for reviewed fields that do
    not need a model attribute. Do not declare the same field in both places.
    """
    return _field({"classification": "opaque"}, {})


def model_graph(root: type[PolicyModel]) -> dict[str, type[PolicyModel]]:
    """Collect the root and nested policy models in traversal order.

    Models are keyed by class name. Reject nested models without policy and
    distinct models sharing a name. Repeated references are visited once;
    recursive schema export is rejected separately.
    """
    models: dict[str, type[PolicyModel]] = {}

    def visit(annotation: Any) -> None:
        """Follow model fields and container or union type arguments once."""
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if not issubclass(annotation, PolicyModel):
                raise ValueError("Nested models must declare field policies")
            name = annotation.__name__
            if name in models:
                if models[name] is not annotation:
                    raise ValueError("Model names must be unique within a payload")
                return
            models[name] = annotation
            for field in annotation.model_fields.values():
                visit(field.annotation)
        for argument in get_args(annotation):
            visit(argument)

    visit(root)
    return models


def field_coverage(model: type[PolicyModel]) -> ProjectionFieldCoverage:
    """Derive disjoint runtime field inventories from one object's declarations.

    Local extensions occupy a separate inventory even when their annotation
    is constrained. The opaque inventory includes declared opaque attributes
    and ObjectPolicy.opaque names. This does not check an upstream schema.
    """
    groups: dict[str, set[str]] = {
        "guarded": set(),
        "constrained": set(),
        "opaque": set(model.policy.opaque),
        "extension": set(),
    }
    for name, field in model.model_fields.items():
        metadata = field_policy(field)
        classification = "extension" if metadata.get("extension") else metadata["classification"]
        groups[classification].add(name)
    return ProjectionFieldCoverage(
        guarded_fields=frozenset(groups["guarded"]),
        constrained_fields=frozenset(groups["constrained"]),
        opaque_fields=frozenset(groups["opaque"]),
        local_extension_fields=frozenset(groups["extension"]),
    )


def guarded_array_variant(annotation: Any) -> tuple[type[PolicyModel], ArrayVariant] | None:
    """Resolve the one text-bearing branch of a reviewed object union.

    Every branch must declare a mutually exclusive discriminator. The selected
    branch has bounded cardinality; other branches must not hide guarded text.
    Return None for a non-union item, whose ordinary array bounds apply instead.
    No JSON Schema generation or provider lookup participates in binding.
    """
    if get_origin(annotation) not in (Union, UnionType):
        return None
    branches: list[tuple[type[PolicyModel], ArrayVariant]] = []
    for model in get_args(annotation):
        if not isinstance(model, type) or not issubclass(model, PolicyModel) or model.policy.variant is None:
            raise ValueError("Guarded object-union branches require explicit array variant policy")
        branches.append((model, model.policy.variant))
    for index, (_, left) in enumerate(branches):
        for _, right in branches[index + 1 :]:
            right_match = dict(right.match)
            if not any(key in right_match and value != right_match[key] for key, value in left.match):
                raise ValueError("Array variant discriminators must be disjoint")
    selected = [(model, variant) for model, variant in branches if variant.cardinality != "any"]
    if len(selected) != 1:
        raise ValueError("Guarded arrays require exactly one bounded text variant")
    for model, variant in branches:
        if variant.cardinality == "any" and any(
            field_policy(field).get("subject")
            for nested in model_graph(model).values()
            for field in nested.model_fields.values()
        ):
            raise ValueError("Unselected array variants must not hide guarded subjects")
    return selected[0]


def payload_contract(
    model: type[PolicyModel],
    *,
    projection_id: str,
    direction: Literal["request", "response"],
    profile: PayloadCapabilityProfile = PayloadCapabilityProfile.SINGLE_TEXT_V1,
) -> PayloadProjectionContract:
    """Build runtime coverage metadata for a request or response projection.

    Args:
        model: Policy-annotated root, with policy-annotated nested models.
        projection_id: Stable identifier for this payload projection.
        direction: Whether the projection describes a request or response.
        profile: Framework-defined capability implemented by the runtime.

    Returns:
        Root and nested-model coverage, ready to attach in a binding module.
        Text extraction is derived separately by text_location.
    """
    return PayloadProjectionContract(
        projection_id=projection_id,
        direction=direction,
        profile=profile,
        root=field_coverage(model),
        content_models=tuple(
            ProjectionModelContract(model=nested, source_schema=nested.policy.source, coverage=field_coverage(nested))
            for nested in model_graph(model).values()
            if nested is not model
        ),
    )


def policy_json_schema(model: type[PolicyModel]) -> dict[str, Any]:
    """Describe default object validation without the contract's union restrictions.

    Add reviewed opaque properties, default closure, and explicit object policy
    to the root and nested model definitions. Inline references so the schema can
    be embedded in an OpenAPI document without dangling definition paths; retain
    union semantics. Recursive models cannot be inlined and are rejected.
    The result is fresh; arbitrary Python validators are not serialized.
    """
    model_graph(model)
    document = model.model_json_schema()
    definitions = document.pop("$defs", {})

    def inline(node: dict[str, Any], active: tuple[str, ...] = ()) -> dict[str, Any]:
        """Resolve schema references without rewriting unions or opaque values."""
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            if name in active:
                raise ValueError("Recursive models are not supported by the inline policy schema")
            resolved = {**definitions[name], **{key: value for key, value in node.items() if key != "$ref"}}
            if EXTENSION in resolved:
                resolved[EXTENSION] = {**node.get(EXTENSION, {}), **definitions[name].get(EXTENSION, {})}
            return inline(resolved, (*active, name))
        result = dict(node)
        if "properties" in result:
            metadata = dict(result.get(EXTENSION, {}))
            for keyword in ("source", "unknown_fields", "reject_case_aliases"):
                if keyword in metadata:
                    metadata[keyword] = metadata.pop(keyword)
            result[EXTENSION] = metadata
            result["properties"] = {name: inline(child, active) for name, child in result["properties"].items()}
        if "items" in result:
            result["items"] = inline(result["items"], active)
        if isinstance(result.get("additionalProperties"), dict):
            result["additionalProperties"] = inline(result["additionalProperties"], active)
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            if keyword in result:
                result[keyword] = [inline(child, active) for child in result[keyword]]
        return result

    return inline(document)


def export_payload_schema(model: type[PolicyModel], *, projection_id: str) -> dict[str, Any]:
    """Export a fresh, inline schema with object and field guard annotations.

    Args:
        model: Trusted policy-annotated model to describe.
        projection_id: Exported root title; the class name remains model metadata.

    Returns:
        A payload schema, not a complete operation contract.
        Defaults and nullable types are retained. Disjoint nullable anyOf
        branches are rendered as oneOf for the current contract vocabulary.
        Reviewed opaque names are exported as opaque-classified properties,
        and additionalProperties reflects closed-by-default validation. Object
        metadata explicitly describes trusted unknown-field overrides and Unicode
        case-alias rejection.

    Raises:
        ValueError: Nested models are not policy-annotated, names collide,
            references recurse, or an anyOf union cannot be safely converted.

    This function does not validate the complete contract format, inspect an
    upstream provider schema, or capture arbitrary Python validator behavior.
    The operation exporter is responsible for format validation.
    """
    document = policy_json_schema(model)

    def expand(node: dict[str, Any]) -> dict[str, Any]:
        """Normalize the inline schema to the narrower contract vocabulary."""
        supported = {
            "title",
            "description",
            "$comment",
            "examples",
            "deprecated",
            "readOnly",
            "writeOnly",
            "type",
            "const",
            "default",
            "enum",
            "minItems",
            "maxItems",
            "minLength",
            "maxLength",
            "pattern",
            "minimum",
            "maximum",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "anyOf",
            "oneOf",
            EXTENSION,
        }
        unsupported = {key for key in node if key not in supported and not key.startswith("x-")}
        if unsupported:
            raise ValueError(f"Schema keywords not expressible in the guard contract: {sorted(unsupported)}")
        result = dict(node)
        metadata = dict(result.get(EXTENSION, {}))
        if "properties" in result:
            result["properties"] = {name: expand(child) for name, child in result["properties"].items()}
        else:
            result.pop("title", None)
        if "items" in result:
            result["items"] = expand(result["items"])
        if "anyOf" in result:
            alternatives = result.pop("anyOf")
            non_null = [branch for branch in alternatives if branch != {"type": "null"}]
            nullable = (
                len(alternatives) == 2
                and len(non_null) == 1
                and non_null[0].get("type") in {"array", "object", "string", "boolean", "integer", "number"}
            )
            objects = all(branch.get("type") == "object" for branch in alternatives)
            disjoint = objects and all(
                any(
                    key in right.get("required", [])
                    and "const" in left.get("properties", {}).get(key, {})
                    and "const" in right.get("properties", {}).get(key, {})
                    and left["properties"][key]["const"] != right["properties"][key]["const"]
                    for key in left.get("required", [])
                )
                for index, left in enumerate(alternatives)
                for right in alternatives[index + 1 :]
            )
            if not nullable and not disjoint:
                raise ValueError("Contract export supports only disjoint nullable or required-literal object unions")
            result["oneOf"] = alternatives
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in result:
                result[keyword] = [expand(child) for child in result[keyword]]
        if metadata:
            result[EXTENSION] = metadata
        return result

    exported = expand(document)
    exported["title"] = projection_id
    exported.setdefault(EXTENSION, {})["model"] = model.__name__
    return exported


def replacement_blocker(model: type[PolicyModel], subject: dict[str, Any]) -> str | None:
    """Return a subject's replacement blocker after checking it names a reviewed sibling.

    A misspelled blocker would never match a provider member and would silently
    allow replacement, so buffered and stream subjects both validate it.

    Raises:
        ValueError: The blocker is not a declared or reviewed opaque field of the model.
    """
    blocker = subject.get("replacement_blocked_by")
    if blocker is not None and blocker not in set(model.model_fields) | set(model.policy.opaque):
        raise ValueError(f"Replacement blocker {blocker!r} is not a field of the subject's object")
    return blocker


def text_location(model: type[PolicyModel]) -> GuardedTextLocation:
    """Derive one buffered text target directly from typed field declarations.

    Follow guarded models and exactly-one-item lists without schema generation
    or contract-format conversion. Heterogeneous arrays use ObjectPolicy.variant
    to select exactly one text-bearing member. A nullable subject, or one without
    a minimum length, may hold no text.

    Raises:
        ValueError: A subject is missing or ambiguous, a guarded container is
            unsupported, or a replacement blocker is not a reviewed sibling.
    """
    locations: list[GuardedTextLocation] = []

    def unwrap(annotation: Any, metadata: list[Any]) -> tuple[Any, list[Any]]:
        """Read Annotated container arguments without serializing their types."""
        if get_origin(annotation) is Annotated:
            annotation, *inner = get_args(annotation)
            metadata = [*inner, *metadata]
        return annotation, metadata

    def bound(metadata: list[Any], name: str, default: int | None = None) -> int | None:
        """Read the last declared length constraint, matching field precedence."""
        return next((getattr(item, name) for item in reversed(metadata) if hasattr(item, name)), default)

    def visit_container(
        annotation: Any,
        metadata: list[Any],
        path: tuple[str | int | GuardedArraySelection, ...],
        active: tuple[type[PolicyModel], ...],
    ) -> None:
        """Follow only explicitly guarded model and singleton-list containers."""
        annotation, metadata = unwrap(annotation, metadata)
        if get_origin(annotation) is list:
            selected = guarded_array_variant(get_args(annotation)[0])
            if selected is not None:
                item, variant = selected
                if variant.cardinality != "exactly_one":
                    raise ValueError("Buffered array variants require exactly_one cardinality")
                visit_model(item, (*path, GuardedArraySelection(variant.match, "exactly_one")), active)
                return
            if bound(metadata, "min_length") != 1 or bound(metadata, "max_length") != 1:
                raise ValueError("Buffered text extraction requires exactly one array item")
            visit_container(get_args(annotation)[0], [], (*path, 0), active)
        elif isinstance(annotation, type) and issubclass(annotation, PolicyModel):
            visit_model(annotation, path, active)
        else:
            raise ValueError(
                "Guarded containers must lead to a named string field; alternatives need explicit bindings"
            )

    def visit_model(
        current: type[PolicyModel],
        path: tuple[str | int | GuardedArraySelection, ...],
        active: tuple[type[PolicyModel], ...],
    ) -> None:
        """Collect subjects from the model objects that own their policies."""
        if current in active:
            raise ValueError("Recursive guarded text paths are not supported")
        for name, field in current.model_fields.items():
            policy = field_policy(field)
            if policy.get("classification") != "guarded":
                continue
            subject = policy.get("subject")
            if not subject:
                visit_container(field.annotation, field.metadata, (*path, name), (*active, current))
                continue
            annotation, metadata = unwrap(field.annotation, field.metadata)
            nullable = get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation)
            if nullable:
                alternatives = [item for item in get_args(annotation) if item is not type(None)]
                annotation = alternatives[0] if len(alternatives) == 1 else None
                annotation, metadata = unwrap(annotation, metadata)
            literal_text = get_origin(annotation) is Literal and all(
                isinstance(value, str) for value in get_args(annotation)
            )
            if annotation is not str and not literal_text:
                raise ValueError("A guarded subject must be a named string field")
            blocker = replacement_blocker(current, subject)
            locations.append(
                GuardedTextLocation(
                    role=subject["role"],
                    object_path=path,
                    member=name,
                    allows_replacement=subject.get("replaceable", False),
                    replacement_blocked_by=blocker,
                    allows_empty=nullable or (bound(metadata, "min_length", 0) or 0) < 1,
                )
            )

    visit_model(model, (), ())
    if len(locations) != 1:
        raise ValueError("Buffered text extraction requires exactly one guarded subject")
    return locations[0]
