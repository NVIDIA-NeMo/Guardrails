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

"""Declare projection policy and derive buffered runtime metadata and schemas.

Model field types define accepted values; Annotated field helpers attach guard
policy, and assignments declare defaults. ObjectPolicy records object-level
source and coverage information. PolicyModel checks local declaration consistency.

Bindings use these declarations to derive coverage and a single guarded text
location. The exporter produces a read-side schema without loading YAML, checking
an upstream provider document, or compiling a contract into Python. These helpers
operate on trusted framework-authored models, not externally supplied Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from pydantic.fields import FieldInfo
from typing_extensions import Self

from nemoguardrails.server.experimental.provider.payload import (
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

    This metadata does not select guardrails, a deployment configuration, or a
    capability profile. Field-specific policy belongs on the field annotations.
    """

    source: str | None = None
    opaque: tuple[str, ...] = ()
    unknown_fields: Literal["forbid", "configurable"] = "forbid"

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


# Pydantic constraints whose exported keywords (minLength/minItems,
# maxLength/maxItems, pattern) exist in the guard contract field vocabulary.
_EXPORTABLE_CONSTRAINTS = frozenset({"min_length", "max_length", "pattern"})


def _field(metadata: dict[str, Any], constraints: dict[str, Any]) -> Any:
    """Build Annotated field metadata while keeping defaults in assignments."""
    if {"default", "default_factory", "alias"} & constraints.keys():
        raise ValueError("Declare defaults explicitly on the field; aliases are not supported")
    unsupported = constraints.keys() - _EXPORTABLE_CONSTRAINTS
    if unsupported:
        raise ValueError(f"Constraints not expressible in the guard contract: {sorted(unsupported)}")
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
        **constraints: min_length, max_length, or pattern; other Pydantic
            constraints have no guard contract keyword and are rejected.

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


def export_payload_schema(model: type[PolicyModel], *, projection_id: str) -> dict[str, Any]:
    """Export a fresh, inline schema with object and field guard annotations.

    Args:
        model: Trusted policy-annotated model to describe.
        projection_id: Exported root title; the class name remains model metadata.

    Returns:
        A payload schema, not a complete operation contract.
        Defaults and nullable types are retained. Disjoint nullable anyOf
        branches are rendered as oneOf for the current contract vocabulary.

    Raises:
        ValueError: Nested models are not policy-annotated, names collide,
            references recurse, or an anyOf union cannot be safely converted.

    This function does not validate the complete contract format, inspect an
    upstream provider schema, or capture arbitrary Python validator behavior.
    The operation exporter is responsible for format validation.
    """
    document = model.model_json_schema()
    definitions = document.pop("$defs", {})
    models = model_graph(model)

    def expand(node: dict[str, Any], active: tuple[str, ...] = ()) -> dict[str, Any]:
        """Inline references and attach object policy without mutating the input."""
        if "$ref" in node:
            name = node["$ref"].removeprefix("#/$defs/")
            if name in active:
                raise ValueError("Recursive models are not supported by the contract export")
            return expand({**definitions[name], **{k: v for k, v in node.items() if k != "$ref"}}, (*active, name))
        result = dict(node)
        metadata = dict(result.get(EXTENSION, {}))
        if "properties" in result:
            policy = models[result["title"]].policy
            if policy.source:
                metadata["source"] = "#/components/schemas/" + policy.source
            if policy.opaque:
                metadata["opaque_fields"] = sorted(policy.opaque)
            result["additionalProperties"] = policy.unknown_fields == "configurable"
            if policy.unknown_fields == "configurable":
                metadata["unknown_fields"] = policy.unknown_fields
            result["properties"] = {name: expand(child, active) for name, child in sorted(result["properties"].items())}
        else:
            result.pop("title", None)
        if "items" in result:
            result["items"] = expand(result["items"], active)
        if "anyOf" in result:
            alternatives = result.pop("anyOf")
            non_null = [branch for branch in alternatives if branch != {"type": "null"}]
            if (
                len(alternatives) != 2
                or len(non_null) != 1
                or non_null[0].get("type") not in {"array", "object", "string", "boolean", "integer", "number"}
            ):
                raise ValueError("Contract export supports only disjoint nullable anyOf unions")
            result["oneOf"] = alternatives
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in result:
                result[keyword] = [expand(child, active) for child in result[keyword]]
        if metadata:
            result[EXTENSION] = metadata
        return result

    exported = expand(document)
    exported["title"] = projection_id
    exported.setdefault(EXTENSION, {})["model"] = model.__name__
    return exported


def text_location(model: type[PolicyModel]) -> GuardedTextLocation:
    """Derive one buffered text target by following guarded object fields.

    Traversed arrays must constrain their size to exactly one item. The subject
    must be a named string field; its role and replacement restrictions become
    runtime location metadata. This helper does not provide union selectors or
    streaming classification and does not itself attach the result to a model.

    Raises:
        ValueError: No unique text subject can be derived, a traversed array is
            not constrained to one item, a replacement blocker is not a field of
            the subject's object, or the schema cannot be exported.
    """
    locations: list[GuardedTextLocation] = []

    def visit(node: dict[str, Any], path: tuple[str | int, ...]) -> None:
        """Accumulate subject locations along guarded object and array paths."""
        subject = node.get(EXTENSION, {}).get("subject")
        if subject:
            if node.get("type") != "string" or not path or not isinstance(path[-1], str):
                raise ValueError("A guarded subject must be a named string field")
            locations.append(
                GuardedTextLocation(
                    role=subject["role"],
                    object_path=path[:-1],
                    member=path[-1],
                    allows_replacement=subject.get("replaceable", False),
                    replacement_blocked_by=subject.get("replacement_blocked_by"),
                )
            )
        if node.get("type") == "array":
            if node.get("minItems") != 1 or node.get("maxItems") != 1:
                raise ValueError("Buffered text extraction requires exactly one array item")
            visit(node["items"], (*path, 0))
        for name, child in node.get("properties", {}).items():
            blocker = child.get(EXTENSION, {}).get("subject", {}).get("replacement_blocked_by")
            # A misspelled blocker would never match and silently allow replacement.
            if blocker is not None and blocker not in node["properties"]:
                if blocker not in node.get(EXTENSION, {}).get("opaque_fields", ()):
                    raise ValueError(f"Replacement blocker {blocker!r} is not a field of the subject's object")
            if child.get(EXTENSION, {}).get("classification") == "guarded":
                visit(child, (*path, name))

    visit(export_payload_schema(model, projection_id=model.__name__), ())
    if len(locations) != 1:
        raise ValueError("Buffered text extraction requires exactly one guarded subject")
    return locations[0]
