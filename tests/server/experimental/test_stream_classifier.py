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

from typing import Literal

import pytest
from pydantic import BaseModel

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
