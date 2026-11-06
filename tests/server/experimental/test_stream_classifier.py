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

import json
from dataclasses import replace
from typing import Literal

import pytest
from pydantic import BaseModel, model_validator

from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import (
    GuardedStreamEvent,
    StreamCapabilityProfile,
    StreamEventRole,
    StreamProjectionContract,
    StreamShapeCoverage,
    UnsupportedProviderStream,
    validate_stream_event,
)
from nemoguardrails.server.experimental.provider.stream_classifier import (
    StreamArraySelection,
    StreamClassifierDefinition,
    StreamEventRule,
    build_stream_classifier,
)


class Delta(BaseModel):
    text: str | None = None


class Chunk(BaseModel):
    kind: Literal["chunk"]
    deltas: list[Delta]


class Error(BaseModel):
    error: str


class SnapshotItem(BaseModel):
    kind: str
    text: str


class Snapshot(BaseModel):
    kind: Literal["snapshot"]
    items: list[SnapshotItem]


CONTRACT = StreamProjectionContract(
    projection_id="test.stream",
    profile=StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1,
    shapes=StreamShapeCoverage(
        guarded_shapes=frozenset({"chunk:text"}),
        snapshot_shapes=frozenset({"snapshot"}),
        opaque_shapes=frozenset({"done", "chunk:metadata"}),
        provider_error_shapes=frozenset({"error"}),
    ),
)


def classifier():
    return build_stream_classifier(
        StreamClassifierDefinition(
            subject="test.stream",
            contract=CONTRACT,
            sentinels=((b"[DONE]", "done"),),
            rules=(
                StreamEventRule(
                    shape="error",
                    model=Error,
                    role=StreamEventRole.PROVIDER_ERROR,
                    required_fields=frozenset({"error"}),
                ),
                StreamEventRule(
                    shape="chunk:text",
                    model=Chunk,
                    role=StreamEventRole.GUARDED_TEXT,
                    match=(("kind", "chunk"),),
                    text_path=("deltas", 0, "text"),
                    missing_text_role=StreamEventRole.OPAQUE_METADATA,
                    missing_text_shape="chunk:metadata",
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    ("raw", "shape", "role", "text"),
    [
        (b"data: [DONE]\n\n", "done", StreamEventRole.OPAQUE_METADATA, None),
        (
            b'data: {"kind":"chunk","deltas":[{"text":"hello"}]}\n\n',
            "chunk:text",
            StreamEventRole.GUARDED_TEXT,
            "hello",
        ),
        (
            b'data: {"kind":"chunk","deltas":[]}\n\n',
            "chunk:metadata",
            StreamEventRole.OPAQUE_METADATA,
            None,
        ),
        (b'data: {"error":"failed"}\n\n', "error", StreamEventRole.PROVIDER_ERROR, None),
    ],
)
def test_classifier_validates_and_classifies_reviewed_shapes(raw, shape, role, text):
    event = classifier().classify_event(ServerSentEvent.from_bytes(raw))

    assert (event.shape, event.role, event.text) == (shape, role, text)


def test_classifier_rejects_unknown_and_ambiguous_events():
    with pytest.raises(UnsupportedProviderStream, match="not covered"):
        classifier().classify_event(ServerSentEvent.from_bytes(b'data: {"kind":"future"}\n\n'))

    with pytest.raises(UnsupportedProviderStream, match="ambiguously matches"):
        classifier().classify_event(
            ServerSentEvent.from_bytes(b'data: {"kind":"chunk","error":"failed","deltas":[]}\n\n')
        )


def test_classifier_can_select_one_nested_text_item():
    snapshot_classifier = build_stream_classifier(
        StreamClassifierDefinition(
            subject="test.stream",
            contract=CONTRACT,
            rules=(
                StreamEventRule(
                    shape="snapshot",
                    model=Snapshot,
                    role=StreamEventRole.TEXT_SNAPSHOT,
                    match=(("kind", "snapshot"),),
                    text_path=(
                        "items",
                        StreamArraySelection(match=(("kind", "message"),), cardinality="at_most_one"),
                        "text",
                    ),
                    missing_text_role=StreamEventRole.TEXT_SNAPSHOT,
                    missing_text_value="",
                ),
            ),
        )
    )

    empty = snapshot_classifier.classify_event(ServerSentEvent.from_bytes(b'data: {"kind":"snapshot","items":[]}\n\n'))
    populated = snapshot_classifier.classify_event(
        ServerSentEvent.from_bytes(b'data: {"kind":"snapshot","items":[{"kind":"message","text":"hello"}]}\n\n')
    )

    assert empty.text == ""
    assert populated.text == "hello"


def test_stream_contract_rejects_unknown_or_misclassified_events():
    with pytest.raises(UnsupportedProviderStream, match="contract declares"):
        validate_stream_event(CONTRACT, GuardedStreamEvent("chunk:text", StreamEventRole.OPAQUE_METADATA))


def test_classifier_definition_must_agree_with_the_stream_contract():
    definition = StreamClassifierDefinition(
        subject="test.stream",
        contract=CONTRACT,
        rules=(
            StreamEventRule(
                shape="chunk:text",
                model=Chunk,
                role=StreamEventRole.OPAQUE_METADATA,
                match=(("kind", "chunk"),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="contract declares 'guarded_text'"):
        build_stream_classifier(definition)


def test_stream_shape_coverage_rejects_multiple_classifications():
    with pytest.raises(ValueError, match="multiple classifications"):
        StreamShapeCoverage(
            guarded_shapes=frozenset({"chunk"}),
            opaque_shapes=frozenset({"chunk"}),
        )


@pytest.mark.parametrize("expected,other", [(False, 0), (True, 1), (0, False), (1, True), (1, 1.0)])
def test_stream_discriminators_require_matching_types(expected, other):
    correct = {"kind": expected}
    incorrect = {"kind": other}
    selector = StreamArraySelection(match=(("kind", expected),), cardinality="exactly_one")
    rule = StreamEventRule(
        shape="chunk:text", model=Chunk, role=StreamEventRole.GUARDED_TEXT, match=(("kind", expected),)
    )

    assert selector.select([incorrect, correct]) is correct
    assert rule.matches(correct) is True
    assert rule.matches(incorrect) is False


@pytest.mark.parametrize("mutation", ["text", "drop", "reorder"])
def test_classifier_rejects_validation_that_changes_the_original_subject(mutation):
    class TransformingChunk(Chunk):
        @model_validator(mode="before")
        @classmethod
        def transform(cls, value):
            if mutation == "text":
                value["deltas"][0]["text"] = "safe"
            elif mutation == "drop":
                value["deltas"] = []
            else:
                value["deltas"].reverse()
            return value

    original = classifier()
    modified = replace(original, rules=(replace(original.rules[1], model=TransformingChunk),))
    event = ServerSentEvent.from_bytes(b'data: {"kind":"chunk","deltas":[{"text":"unsafe"},{"text":"safe"}]}\n\n')

    with pytest.raises(UnsupportedProviderStream, match="original guarded text"):
        modified.classify_event(event)


@pytest.mark.parametrize("value", [None, "", "hello"])
def test_classifier_preserves_nullable_and_empty_original_text(value):
    event = ServerSentEvent.from_bytes(
        b"data: " + json.dumps({"kind": "chunk", "deltas": [{"text": value}]}).encode() + b"\n\n"
    )
    assert classifier().classify_event(event).text == value


def test_classifier_distinguishes_boolean_integer_and_float_selections():
    rules = tuple(
        StreamEventRule(
            shape="chunk:metadata", model=BaseModel, role=StreamEventRole.OPAQUE_METADATA, match=(("kind", value),)
        )
        for value in (True, 1, 1.0)
    )
    result = build_stream_classifier(StreamClassifierDefinition(subject="test.stream", contract=CONTRACT, rules=rules))
    for value in (True, 1, 1.0):
        assert sum(rule.matches({"kind": value}) for rule in result.rules) == 1


def test_array_selection_distinguishes_missing_from_null():
    selector = StreamArraySelection(match=(("kind", None),), cardinality="exactly_one")
    explicit_null = {"kind": None}
    assert selector.select([{}, explicit_null]) is explicit_null


def chunk_rule(**changes):
    rule = StreamEventRule(
        shape="chunk:text",
        model=Chunk,
        role=StreamEventRole.GUARDED_TEXT,
        match=(("kind", "chunk"),),
        text_path=("deltas", 0, "text"),
    )
    return replace(rule, **changes)


def definition(**changes):
    return replace(
        StreamClassifierDefinition(subject="test.stream", contract=CONTRACT, rules=(chunk_rule(),)), **changes
    )


def event(raw):
    return ServerSentEvent.from_bytes(raw)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"subject": ""}, "identify its provider event subject"),
        ({"rules": ()}, "at least one event rule"),
        ({"subject": "other.stream"}, "match its projection contract identifier"),
        ({"require_event_type": True}, "needs a data discriminator"),
        ({"non_data_shape": "chunk:text"}, "contract declares"),
        ({"rules": (chunk_rule(shape=""),)}, "identify its provider shape"),
        ({"rules": (chunk_rule(match=()),)}, "must declare how it is selected"),
        ({"rules": (chunk_rule(), chunk_rule())}, "duplicates another selection"),
        ({"rules": (chunk_rule(text_path=None),)}, "inconsistent text role and path"),
        (
            {
                "rules": (
                    chunk_rule(
                        shape="chunk:metadata",
                        role=StreamEventRole.OPAQUE_METADATA,
                        text_path=None,
                        missing_text_role=StreamEventRole.OPAQUE_METADATA,
                    ),
                )
            },
            "without a text path",
        ),
        ({"rules": (chunk_rule(missing_text_shape="chunk:metadata"),)}, "rename missing text without classifying"),
        ({"rules": (chunk_rule(missing_text_value=""),)}, "supply missing text without classifying"),
        ({"rules": (chunk_rule(missing_text_role=StreamEventRole.GUARDED_TEXT),)}, "classify missing text as content"),
    ],
)
def test_classifier_definition_rejects_inconsistent_declarations(changes, message):
    with pytest.raises(ValueError, match=message):
        build_stream_classifier(definition(**changes))


def test_data_less_events_need_a_declared_shape():
    data_less = event(b"event: ping\n\n")
    with pytest.raises(UnsupportedProviderStream, match="must contain data"):
        build_stream_classifier(definition()).classify_event(data_less)

    classified = build_stream_classifier(definition(non_data_shape="done")).classify_event(data_less)
    assert classified == GuardedStreamEvent("done", StreamEventRole.OPAQUE_METADATA)


def test_event_type_field_must_agree_with_the_sse_event_name():
    typed = build_stream_classifier(definition(event_type_field="kind", require_event_type=True))
    chunk = b'data: {"kind":"chunk","deltas":[{"text":"hi"}]}\n\n'

    assert typed.classify_event(event(b"event: chunk\n" + chunk)).text == "hi"
    with pytest.raises(UnsupportedProviderStream, match="SSE event field is required"):
        typed.classify_event(event(chunk))
    with pytest.raises(UnsupportedProviderStream, match="must match data field 'kind'"):
        typed.classify_event(event(b"event: other\n" + chunk))
    with pytest.raises(UnsupportedProviderStream, match="must declare string field 'kind'"):
        typed.classify_event(event(b'event: chunk\ndata: {"kind":1,"deltas":[]}\n\n'))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'data: {"kind":"chunk","deltas":[{"text":null}]}\n\n', "has no guarded text value"),
        (b'data: {"kind":"chunk","deltas":[{"text":"hi"},5]}\n\n', "test.stream"),
        (b"data: {not json}\n\n", "JSON"),
    ],
    ids=["missing-text", "schema", "json"],
)
def test_classifier_rejects_unusable_event_payloads(raw, message):
    with pytest.raises(UnsupportedProviderStream, match=message):
        build_stream_classifier(definition()).classify_event(event(raw))


@pytest.mark.parametrize(
    ("text_path", "message"),
    [
        (("deltas", -1, "text"), "negative indexes"),
        (("deltas", "text"), "does not match its projection"),
        (("deltas",), "must resolve to a string"),
    ],
)
def test_classifier_rejects_text_paths_that_do_not_reach_a_string(text_path, message):
    classifier = build_stream_classifier(definition(rules=(chunk_rule(text_path=text_path),)))
    with pytest.raises(UnsupportedProviderStream, match=message):
        classifier.classify_event(event(b'data: {"kind":"chunk","deltas":[{"text":"hi"}]}\n\n'))


def test_exactly_one_array_selection_rejects_missing_or_repeated_items():
    selection = StreamArraySelection(match=(("kind", "message"),), cardinality="exactly_one")
    with pytest.raises(ValueError, match="exactly one matching item"):
        selection.select([{"kind": "other"}])
    with pytest.raises(ValueError, match="exactly one matching item"):
        selection.select([{"kind": "message"}, {"kind": "message"}])


def test_dotted_match_keys_do_not_match_through_scalars():
    rule = chunk_rule(match=(("kind.name", "chunk"),))
    assert rule.matches({"kind": {"name": "chunk"}}) is True
    assert rule.matches({"kind": "chunk"}) is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"projection_id": " "}, "must have an identifier"),
        ({"shapes": StreamShapeCoverage()}, "at least one shape"),
    ],
)
def test_stream_projection_contract_requires_identity_and_shapes(kwargs, message):
    values = {"projection_id": "p", "profile": StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1, "shapes": CONTRACT.shapes}
    with pytest.raises(ValueError, match=message):
        StreamProjectionContract(**{**values, **kwargs})


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("", StreamEventRole.OPAQUE_METADATA), "identify its provider shape"),
        (("chunk:text", StreamEventRole.GUARDED_TEXT), "must carry text"),
        (("done", StreamEventRole.OPAQUE_METADATA, "text"), "must carry text"),
    ],
)
def test_guarded_stream_event_requires_shape_and_role_consistent_text(args, message):
    with pytest.raises(ValueError, match=message):
        GuardedStreamEvent(*args)


def test_stream_contract_rejects_undeclared_shapes():
    with pytest.raises(UnsupportedProviderStream, match="is not declared"):
        validate_stream_event(CONTRACT, GuardedStreamEvent("future", StreamEventRole.OPAQUE_METADATA))
