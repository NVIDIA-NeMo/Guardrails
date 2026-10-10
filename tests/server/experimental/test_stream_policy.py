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

"""Exercise typed stream metadata derivation without provider services."""

from dataclasses import replace
from typing import Annotated

import pytest
from pydantic import create_model

from nemoguardrails.server.experimental.provider import projection_policy
from nemoguardrails.server.experimental.provider.projection_policy import (
    PolicyModel,
    constrained,
    field_coverage,
    guarded,
    text_location,
)
from nemoguardrails.server.experimental.provider.stream import (
    StreamCapabilityProfile,
    StreamEventRole,
    StreamProjectionContract,
)
from nemoguardrails.server.experimental.provider.stream_classifier import (
    StreamClassifierDefinition,
    StreamEventRule,
    build_stream_classifier,
)
from nemoguardrails.server.experimental.provider.stream_policy import stream_shape_coverage, stream_text_path


class TextDelta(PolicyModel):
    text: Annotated[str | None, guarded("assistant")] = None


class DeltaEnvelope(PolicyModel):
    deltas: Annotated[list[TextDelta], guarded(max_length=1)]


def test_stream_path_allows_empty_arrays_and_nullable_text_without_weakening_buffered_paths():
    assert stream_text_path(DeltaEnvelope) == ("deltas", 0, "text")
    assert DeltaEnvelope(deltas=[]).deltas == []
    assert TextDelta().text is None
    assert TextDelta(text="").text == ""
    with pytest.raises(ValueError, match="exactly one array item"):
        text_location(DeltaEnvelope)
    # Buffered extraction also accepts a nullable subject, reported as allowing empty text.
    assert text_location(TextDelta).allows_empty is True


def test_stream_path_follows_nullable_model_containers():
    """A null container yields missing text; the event rule decides how that is classified."""

    class Envelope(PolicyModel):
        delta: Annotated[TextDelta | None, guarded()] = None

    assert stream_text_path(Envelope) == ("delta", "text")


@pytest.mark.parametrize("maximum", [None, 0, 2])
def test_stream_path_rejects_arrays_without_an_explicit_single_item_boundary(maximum):
    class Envelope(PolicyModel):
        deltas: Annotated[list[TextDelta], guarded(max_length=maximum)]

    with pytest.raises(ValueError, match="at most one array item"):
        stream_text_path(Envelope)


@pytest.mark.parametrize("annotation", [int, list[str], str | int])
def test_stream_path_rejects_non_string_or_ambiguous_subjects(annotation):
    model = create_model(
        "InvalidSubject", __base__=PolicyModel, text=(Annotated[annotation, guarded("assistant")], ...)
    )
    with pytest.raises(ValueError):
        stream_text_path(model)


@pytest.mark.parametrize("fields", [{}, {"a": TextDelta, "b": TextDelta}])
def test_stream_path_requires_one_subject(fields):
    model = create_model(
        "Envelope", __base__=PolicyModel, **{name: (Annotated[child, guarded()], ...) for name, child in fields.items()}
    )
    with pytest.raises(ValueError, match="exactly one guarded subject"):
        stream_text_path(model)


def rule(shape, role, **kwargs):
    return StreamEventRule(shape=shape, model=TextDelta, role=role, match=(("type", shape),), **kwargs)


def test_shape_coverage_includes_rules_snapshots_fallbacks_and_transport():
    rules = (
        rule(
            "text",
            StreamEventRole.GUARDED_TEXT,
            text_path=("text",),
            missing_text_role=StreamEventRole.OPAQUE_METADATA,
            missing_text_shape="metadata",
        ),
        rule(
            "snapshot",
            StreamEventRole.TEXT_SNAPSHOT,
            text_path=("text",),
            missing_text_role=StreamEventRole.TEXT_SNAPSHOT,
            missing_text_value="",
        ),
        rule("error", StreamEventRole.PROVIDER_ERROR),
        rule("usage", StreamEventRole.OPAQUE_METADATA),
    )
    sentinels = ((b"[DONE]", "done"),)
    coverage = stream_shape_coverage(rules, sentinels=sentinels, non_data_shape="keepalive")
    assert coverage.guarded_shapes == frozenset({"text"})
    assert coverage.snapshot_shapes == frozenset({"snapshot"})
    assert coverage.provider_error_shapes == frozenset({"error"})
    assert coverage.opaque_shapes == frozenset({"metadata", "usage", "done", "keepalive"})
    contract = StreamProjectionContract("example", StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1, coverage)
    classifier = build_stream_classifier(
        StreamClassifierDefinition("example", contract, rules, sentinels=sentinels, non_data_shape="keepalive")
    )
    assert classifier.contract.shapes == coverage


def test_shape_coverage_rejects_conflicting_roles():
    rules = (rule("text", StreamEventRole.GUARDED_TEXT, text_path=("text",)),)
    with pytest.raises(ValueError, match="multiple classifications"):
        stream_shape_coverage(rules, non_data_shape="text")


def test_derived_coverage_does_not_bypass_classifier_consistency_checks():
    rules = (rule("text", StreamEventRole.GUARDED_TEXT),)
    contract = StreamProjectionContract(
        "example", StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1, stream_shape_coverage(rules)
    )
    with pytest.raises(ValueError, match="inconsistent text role and path"):
        build_stream_classifier(StreamClassifierDefinition("example", contract, rules))


def test_stream_path_ignores_unrelated_field_types():
    """A constrained union elsewhere in the model cannot break the guarded text binding."""

    class Envelope(PolicyModel):
        deltas: Annotated[list[TextDelta], guarded(max_length=1)]
        seed: Annotated[str | int | None, constrained()] = None

    assert stream_text_path(Envelope) == ("deltas", 0, "text")


def test_stream_path_does_not_generate_schemas(monkeypatch):
    """Runtime bindings come from typed fields, not from contract export."""

    def unexpected(*args, **kwargs):
        raise AssertionError("stream text paths must not generate schemas")

    monkeypatch.setattr(projection_policy, "export_payload_schema", unexpected)
    monkeypatch.setattr(PolicyModel, "model_json_schema", classmethod(unexpected))
    assert stream_text_path(DeltaEnvelope) == ("deltas", 0, "text")


def test_stream_path_rejects_union_containers_and_recursion():
    class Other(PolicyModel):
        text: Annotated[str, guarded("assistant")]

    class Either(PolicyModel):
        delta: Annotated[TextDelta | Other, guarded()]

    class Node(PolicyModel):
        child: Annotated["Node | None", guarded()] = None

    Node.model_rebuild()
    with pytest.raises(ValueError, match="object-union selectors"):
        stream_text_path(Either)
    with pytest.raises(ValueError, match="Recursive"):
        stream_text_path(Node)


@pytest.mark.parametrize("mutation", ["path", "role", "coverage", "inventory"])
def test_typed_classifier_rejects_policy_drift(mutation):
    """Binding mistakes must not downgrade guarded content to metadata."""
    from nemoguardrails.server.experimental.provider import stream_policy

    event = StreamEventRule(
        "text",
        DeltaEnvelope,
        StreamEventRole.GUARDED_TEXT,
        required_fields=frozenset({"deltas"}),
        text_path=("deltas", 0, "text"),
        missing_text_role=StreamEventRole.OPAQUE_METADATA,
        missing_text_shape="metadata",
    )
    coverage = stream_shape_coverage((event,))
    contract = StreamProjectionContract(
        "example", StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1, coverage, field_coverage(DeltaEnvelope)
    )
    if mutation == "path":
        event = replace(event, text_path=("deltas", 1, "text"))
    elif mutation == "role":
        event = replace(
            event, role=StreamEventRole.OPAQUE_METADATA, text_path=None, missing_text_role=None, missing_text_shape=None
        )
        contract = replace(contract, shapes=stream_shape_coverage((event,)))
    elif mutation == "coverage":
        contract = replace(
            contract, fields=replace(contract.fields, guarded_fields=frozenset(), opaque_fields=frozenset({"deltas"}))
        )
    else:
        contract = replace(contract, shapes=replace(coverage, opaque_shapes=frozenset({"metadata", "undeclared"})))
    definition = StreamClassifierDefinition("example", contract, (event,))
    with pytest.raises(ValueError, match="field policy|guarded subject|event rules"):
        stream_policy.build_policy_stream_classifier(definition)


def test_typed_classifier_preserves_nullable_missing_text_and_checks_actual_text():
    """Validated bindings retain explicit missing-text behavior."""
    from nemoguardrails.server.experimental.provider import stream_policy
    from nemoguardrails.server.experimental.provider.sse import ServerSentEvent

    event = StreamEventRule(
        "text",
        DeltaEnvelope,
        StreamEventRole.GUARDED_TEXT,
        required_fields=frozenset({"deltas"}),
        text_path=stream_text_path(DeltaEnvelope),
        missing_text_role=StreamEventRole.OPAQUE_METADATA,
        missing_text_shape="metadata",
    )
    contract = StreamProjectionContract(
        "example", StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1, stream_shape_coverage((event,))
    )
    classifier = stream_policy.build_policy_stream_classifier(StreamClassifierDefinition("example", contract, (event,)))
    assert (
        classifier.classify_event(ServerSentEvent.from_bytes(b'data: {"deltas": []}\n\n')).role
        is StreamEventRole.OPAQUE_METADATA
    )
    classified = classifier.classify_event(ServerSentEvent.from_bytes(b'data: {"deltas": [{"text": "checked"}]}\n\n'))
    assert classified.role is StreamEventRole.GUARDED_TEXT
    assert classified.text == "checked"
