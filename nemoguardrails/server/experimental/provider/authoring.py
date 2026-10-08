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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast, get_args

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from nemoguardrails.server.experimental.provider.payload import (
    GuardedTextLocation,
    PayloadCapabilityProfile,
    PayloadProjectionContract,
    ProjectionFieldCoverage,
    ProjectionModelContract,
)

EXTENSION = "x-nemo-guardrails"
CONTRACT_VERSION = "1.0.0-alpha.1"


@dataclass(frozen=True)
class Policy:
    source: str | None = None
    opaque: tuple[str, ...] = ()
    unknown_fields: Literal["configurable"] | None = None


class PolicyModel(BaseModel):
    policy: ClassVar[Policy] = Policy()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
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


def field_policy(field: FieldInfo) -> dict[str, Any]:
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    metadata = extra.get(EXTENSION)
    return cast(dict[str, Any], metadata) if isinstance(metadata, dict) else {}


def _field(metadata: dict[str, Any], constraints: dict[str, Any]) -> Any:
    if {"default", "default_factory", "alias"} & constraints.keys():
        raise ValueError("Declare defaults explicitly on the field; aliases are not supported")
    return Field(json_schema_extra={EXTENSION: metadata}, **constraints)


def guarded(
    role: Literal["user", "assistant"] | None = None,
    *,
    replaceable: bool | None = None,
    blocked_by: str | None = None,
    replacement_reason: str | None = None,
    **constraints: Any,
) -> Any:
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
            subject["replacement_reason"] = replacement_reason
        metadata["subject"] = subject
    return _field(metadata, constraints)


def constrained(*, reason: str | None = None, **constraints: Any) -> Any:
    metadata: dict[str, Any] = {"classification": "constrained"}
    if reason is not None:
        metadata["reason"] = reason
    return _field(metadata, constraints)


def disabled(reason: str, *, extension: bool = False) -> Any:
    if not reason.strip():
        raise ValueError("Disabled fields require a reason")
    metadata: dict[str, Any] = {"classification": "constrained", "gate": "disabled", "reason": reason}
    if extension:
        metadata["extension"] = True
    return _field(metadata, {})


def opaque() -> Any:
    return _field({"classification": "opaque"}, {})


def model_graph(root: type[PolicyModel]) -> dict[str, type[PolicyModel]]:
    models: dict[str, type[PolicyModel]] = {}

    def visit(annotation: Any) -> None:
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
    document = model.model_json_schema()
    definitions = document.pop("$defs", {})
    models = model_graph(model)

    def expand(node: dict[str, Any], active: tuple[str, ...] = ()) -> dict[str, Any]:
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
            if policy.unknown_fields:
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
    locations: list[GuardedTextLocation] = []

    def visit(node: dict[str, Any], path: tuple[str | int, ...]) -> None:
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
            if child.get(EXTENSION, {}).get("classification") == "guarded":
                visit(child, (*path, name))

    visit(export_payload_schema(model, projection_id=model.__name__), ())
    if len(locations) != 1:
        raise ValueError("Buffered text extraction requires exactly one guarded subject")
    return locations[0]
