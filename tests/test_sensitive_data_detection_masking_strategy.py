# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the `masking_strategy` option of the Presidio masking rail.

These drive `_anonymize` with hand-built analyzer results, so they need only
`presidio-anonymizer` -- no analyzer, no spaCy model. End-to-end coverage lives in
`test_sensitive_data_detection.py`, which skips unless the full SDD setup is present.
"""

import subprocess
import sys
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.rails.llm.config import RailsConfig, SensitiveDataDetectionOptions

ANONYMIZER_PRESENT = check_optional_dependency("presidio_anonymizer")

requires_anonymizer = pytest.mark.skipif(not ANONYMIZER_PRESENT, reason="presidio-anonymizer is not installed.")

if ANONYMIZER_PRESENT:
    from presidio_anonymizer.entities import RecognizerResult

    from nemoguardrails.library.sensitive_data_detection.actions import _anonymize


def _person(start: int, end: int) -> "RecognizerResult":
    return RecognizerResult("PERSON", start, end, 0.85)


TWO_PEOPLE = "Alice met Bob"
TWO_PEOPLE_SPANS = [(0, 5), (10, 13)]


@pytest.mark.unit
@requires_anonymizer
def test_per_type_is_the_existing_behavior():
    """The default must keep producing exactly "<ENTITY_TYPE>"."""
    spans = [_person(*s) for s in TWO_PEOPLE_SPANS]
    assert _anonymize(TWO_PEOPLE, spans, ["PERSON"], "per_type") == "<PERSON> met <PERSON>"


@pytest.mark.unit
@requires_anonymizer
def test_per_value_numbers_distinct_values():
    spans = [_person(*s) for s in TWO_PEOPLE_SPANS]
    masked = _anonymize(TWO_PEOPLE, spans, ["PERSON"], "per_value")
    assert masked == "<PERSON_1> met <PERSON_2>"


@pytest.mark.unit
@requires_anonymizer
def test_per_value_numbers_in_text_order():
    """Presidio invokes operators right to left, so this is the regression that matters."""
    # Supplied deliberately out of text order, as a recognizer may well return them.
    spans = [_person(10, 13), _person(0, 5)]
    assert _anonymize(TWO_PEOPLE, spans, ["PERSON"], "per_value") == "<PERSON_1> met <PERSON_2>"


@pytest.mark.unit
@requires_anonymizer
def test_per_value_reuses_the_token_for_a_repeated_value():
    text = "Alice met Bob. Alice left."
    spans = [_person(0, 5), _person(10, 13), _person(15, 20)]
    masked = _anonymize(text, spans, ["PERSON"], "per_value")
    assert masked == "<PERSON_1> met <PERSON_2>. <PERSON_1> left."


@pytest.mark.unit
@requires_anonymizer
def test_per_value_counts_each_entity_type_separately():
    text = "Alice lives in Paris with Bob"
    spans = [
        RecognizerResult("PERSON", 0, 5, 0.9),
        RecognizerResult("LOCATION", 15, 20, 0.9),
        RecognizerResult("PERSON", 26, 29, 0.9),
    ]
    masked = _anonymize(text, spans, ["PERSON", "LOCATION"], "per_value")
    assert masked == "<PERSON_1> lives in <LOCATION_1> with <PERSON_2>"


@pytest.mark.unit
@requires_anonymizer
@pytest.mark.parametrize("masking_strategy", ["per_type", "per_value"])
def test_no_detections_leaves_the_text_alone(masking_strategy):
    assert _anonymize(TWO_PEOPLE, [], ["PERSON"], masking_strategy) == TWO_PEOPLE


@pytest.mark.unit
def test_masking_strategy_defaults_to_per_type():
    assert SensitiveDataDetectionOptions().masking_strategy == "per_type"


@pytest.mark.unit
@pytest.mark.parametrize("masking_strategy", ["per_type", "per_value"])
def test_valid_strategies_are_accepted(masking_strategy):
    options = SensitiveDataDetectionOptions(entities=["PERSON"], masking_strategy=masking_strategy)
    assert options.masking_strategy == masking_strategy


@pytest.mark.unit
@pytest.mark.parametrize("masking_strategy", ["per_entity", "PER_VALUE", "", None])
def test_unknown_strategies_are_rejected_at_config_load(masking_strategy):
    with pytest.raises(ValidationError):
        SensitiveDataDetectionOptions(entities=["PERSON"], masking_strategy=masking_strategy)


@pytest.mark.unit
@requires_anonymizer
@pytest.mark.parametrize("masking_strategy", ["per_entity", "PER_VALUE", "", None])
def test_unknown_strategies_are_rejected_by_helper(masking_strategy):
    with pytest.raises(ValueError, match="Unsupported masking strategy"):
        _anonymize(TWO_PEOPLE, [], ["PERSON"], masking_strategy)


@pytest.mark.unit
@requires_anonymizer
@pytest.mark.parametrize(
    "text, detections, expected",
    [
        (
            "Alice Smith lives here. Bob visits.",
            [("PERSON", 0, 11, 0.9), ("LOCATION", 6, 16, 0.9), ("PERSON", 24, 27, 0.9)],
            "<PERSON_1><LOCATION_1>s here. <PERSON_2> visits.",
        ),
        (
            "aaaaaaaaaabbbbbcccccdddd",
            [("A", 0, 10, 0.9), ("B", 5, 15, 0.5), ("C", 8, 20, 0.4)],
            "<A_1><B_1><C_1>dddd",
        ),
        (
            "Alice Smith met Bob",
            [("PERSON", 0, 11, 0.9), ("LOCATION", 6, 11, 0.8), ("PERSON", 16, 19, 0.9)],
            "<PERSON_1> met <PERSON_2>",
        ),
    ],
    ids=["partial-overlap-before-another-entity", "three-intersecting-entities", "contained-entity"],
)
def test_per_value_resolves_overlaps_without_exposing_detected_text(text, detections, expected):
    spans = [RecognizerResult(*detection) for detection in detections]
    entities = list(dict.fromkeys(span.entity_type for span in spans))
    assert _anonymize(text, spans, entities, "per_value") == expected


@pytest.mark.unit
@requires_anonymizer
def test_per_value_tokens_are_scoped_to_each_call():
    assert _anonymize("Alice", [_person(0, 5)], ["PERSON"], "per_value") == "<PERSON_1>"
    assert _anonymize("Bob", [_person(0, 3)], ["PERSON"], "per_value") == "<PERSON_1>"


@pytest.mark.unit
@requires_anonymizer
def test_anonymization_does_not_require_analyzer():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["presidio_analyzer"] = None
from presidio_anonymizer.entities import RecognizerResult
from nemoguardrails.library.sensitive_data_detection.actions import _anonymize
assert _anonymize("Alice", [RecognizerResult("PERSON", 0, 5, 0.9)], ["PERSON"], "per_value") == "<PERSON_1>"
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.asyncio
@requires_anonymizer
@pytest.mark.parametrize(
    "source, target", [("input", "user_message"), ("output", "bot_message"), ("retrieval", "relevant_chunks")]
)
@pytest.mark.parametrize("strategy", [None, "per_type", "per_value"])
async def test_masking_strategy_is_applied_per_source(monkeypatch, source, target, strategy):
    from nemoguardrails.library.sensitive_data_detection import actions

    options = {"entities": ["PERSON"]}
    if strategy is not None:
        options["masking_strategy"] = strategy
    config = RailsConfig.from_content(
        config={"models": [], "rails": {"config": {"sensitive_data_detection": {source: options}}}}
    )
    analyzer = Mock()
    analyzer.analyze.return_value = [_person(*span) for span in TWO_PEOPLE_SPANS]
    monkeypatch.setattr(actions, "_get_analyzer", lambda: analyzer)
    monkeypatch.setattr(actions, "_get_ad_hoc_recognizers", lambda _: [])

    outcome = await actions.mask_sensitive_data(source, TWO_PEOPLE, config)

    expected = "<PERSON_1> met <PERSON_2>" if strategy == "per_value" else "<PERSON> met <PERSON>"
    assert outcome.transform_text == {target: expected}
