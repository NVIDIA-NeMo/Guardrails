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

"""Derive streaming metadata from typed projection fields and event rules.

Field policy declares the guarded text location; StreamEventRule declares event
selection, roles, and missing-text handling. These helpers remove duplicate
inventories without inferring provider lifecycle behavior or upstream coverage.
They operate on trusted framework declarations, not user-supplied schemas.
"""

from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from nemoguardrails.server.experimental.provider.projection_policy import PolicyModel, field_policy
from nemoguardrails.server.experimental.provider.stream import StreamEventRole, StreamShapeCoverage
from nemoguardrails.server.experimental.provider.stream_classifier import StreamEventRule, StreamTextPath


def stream_text_path(model: type[PolicyModel]) -> StreamTextPath:
    """Derive one named string or nullable-string subject from typed guarded fields.

    Follow guarded models, nullable model containers, and lists that allow at
    most one item, without schema generation, so unrelated field types cannot
    affect the binding. Unlike buffered extraction, empty arrays, null
    containers, and null text are valid: the event rule decides whether absent
    text is rejected or classified as metadata. This function only derives the
    path; it does not validate an event or choose its role.

    Raises:
        ValueError: The declaration has no unique text subject, an array that
            may hold more than one item, an unsupported subject type, a union
            container, or a recursive path. Object-union selectors and
            multi-item arrays need an explicit StreamEventRule.text_path.
    """
    paths: list[StreamTextPath] = []

    def unwrap(annotation: Any, metadata: list[Any]) -> tuple[Any, list[Any]]:
        """Read Annotated arguments and drop one null alternative."""
        if get_origin(annotation) is Annotated:
            annotation, *inner = get_args(annotation)
            metadata = [*inner, *metadata]
        if get_origin(annotation) in (Union, UnionType):
            alternatives = [item for item in get_args(annotation) if item is not type(None)]
            if len(alternatives) != 1 or len(alternatives) == len(get_args(annotation)):
                raise ValueError("Stream text extraction does not infer object-union selectors")
            return unwrap(alternatives[0], metadata)
        return annotation, metadata

    def max_length(metadata: list[Any]) -> int | None:
        """Read the last declared maximum length, matching field precedence."""
        return next((item.max_length for item in reversed(metadata) if hasattr(item, "max_length")), None)

    def visit_container(
        annotation: Any, metadata: list[Any], path: StreamTextPath, active: tuple[type[PolicyModel], ...]
    ) -> None:
        """Follow guarded models and lists bounded to at most one item."""
        annotation, metadata = unwrap(annotation, metadata)
        if get_origin(annotation) is list:
            if max_length(metadata) != 1:
                raise ValueError("Stream text extraction requires at most one array item")
            visit_container(get_args(annotation)[0], [], (*path, 0), active)
        elif isinstance(annotation, type) and issubclass(annotation, PolicyModel):
            visit_model(annotation, path, active)
        else:
            raise ValueError("Guarded stream containers must lead to a named string field")

    def visit_model(current: type[PolicyModel], path: StreamTextPath, active: tuple[type[PolicyModel], ...]) -> None:
        """Collect subjects from the models that own their policies."""
        if current in active:
            raise ValueError("Recursive guarded stream text paths are not supported")
        for name, field in current.model_fields.items():
            policy = field_policy(field)
            if policy.get("classification") != "guarded":
                continue
            if not policy.get("subject"):
                visit_container(field.annotation, list(field.metadata), (*path, name), (*active, current))
                continue
            annotation, _ = unwrap(field.annotation, list(field.metadata))
            if annotation is not str:
                raise ValueError("A guarded stream subject must be a named string field")
            paths.append((*path, name))

    visit_model(model, (), ())
    if len(paths) != 1:
        raise ValueError("Stream text extraction requires exactly one guarded subject")
    return paths[0]


def stream_shape_coverage(
    rules: tuple[StreamEventRule, ...],
    *,
    sentinels: tuple[tuple[bytes, str], ...] = (),
    non_data_shape: str | None = None,
) -> StreamShapeCoverage:
    """Collect event shapes by role, including fallbacks and opaque transport.

    A shape shared by different roles is rejected by StreamShapeCoverage.
    Selection and fallback consistency are checked by build_stream_classifier;
    deriving coverage does not replace that validation or prove provider-wide
    event coverage.
    """
    groups: dict[StreamEventRole, set[str]] = {role: set() for role in StreamEventRole}
    for rule in rules:
        groups[rule.role].add(rule.shape)
        if rule.missing_text_role is not None:
            groups[rule.missing_text_role].add(rule.missing_text_shape or rule.shape)
    groups[StreamEventRole.OPAQUE_METADATA].update(shape for _, shape in sentinels)
    if non_data_shape is not None:
        groups[StreamEventRole.OPAQUE_METADATA].add(non_data_shape)
    return StreamShapeCoverage(
        guarded_shapes=frozenset(groups[StreamEventRole.GUARDED_TEXT]),
        snapshot_shapes=frozenset(groups[StreamEventRole.TEXT_SNAPSHOT]),
        opaque_shapes=frozenset(groups[StreamEventRole.OPAQUE_METADATA]),
        provider_error_shapes=frozenset(groups[StreamEventRole.PROVIDER_ERROR]),
    )
