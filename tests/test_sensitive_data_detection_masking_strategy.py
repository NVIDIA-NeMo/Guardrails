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

import pytest
from pydantic import ValidationError

from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.rails.llm.config import SensitiveDataDetectionOptions

ANONYMIZER_PRESENT = check_optional_dependency("presidio_anonymizer")

pytestmark = pytest.mark.skipif(not ANONYMIZER_PRESENT, reason="presidio-anonymizer is not installed.")

if ANONYMIZER_PRESENT:
    from presidio_anonymizer.entities import RecognizerResult

    from nemoguardrails.library.sensitive_data_detection.actions import _anonymize


def _person(start: int, end: int) -> "RecognizerResult":
    return RecognizerResult("PERSON", start, end, 0.85)


TWO_PEOPLE = "Alice met Bob"
TWO_PEOPLE_SPANS = [(0, 5), (10, 13)]


@pytest.mark.unit
@pytest.mark.parametrize("masking_strategy", ["per_type", None])
def test_per_type_is_the_existing_behavior(masking_strategy):
    """The default must keep producing exactly "<ENTITY_TYPE>"."""
    spans = [_person(*s) for s in TWO_PEOPLE_SPANS]
    strategy = masking_strategy or SensitiveDataDetectionOptions().masking_strategy
    assert _anonymize(TWO_PEOPLE, spans, ["PERSON"], strategy) == "<PERSON> met <PERSON>"


@pytest.mark.unit
def test_per_value_numbers_distinct_values():
    spans = [_person(*s) for s in TWO_PEOPLE_SPANS]
    masked = _anonymize(TWO_PEOPLE, spans, ["PERSON"], "per_value")
    assert masked == "<PERSON_1> met <PERSON_2>"


@pytest.mark.unit
def test_per_value_numbers_in_text_order():
    """Presidio invokes operators right to left, so this is the regression that matters."""
    # Supplied deliberately out of text order, as a recognizer may well return them.
    spans = [_person(10, 13), _person(0, 5)]
    assert _anonymize(TWO_PEOPLE, spans, ["PERSON"], "per_value") == "<PERSON_1> met <PERSON_2>"


@pytest.mark.unit
def test_per_value_reuses_the_token_for_a_repeated_value():
    text = "Alice met Bob. Alice left."
    spans = [_person(0, 5), _person(10, 13), _person(15, 20)]
    masked = _anonymize(text, spans, ["PERSON"], "per_value")
    assert masked == "<PERSON_1> met <PERSON_2>. <PERSON_1> left."


@pytest.mark.unit
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
@pytest.mark.parametrize("masking_strategy", ["per_entity", "PER_VALUE", ""])
def test_unknown_strategies_are_rejected_at_config_load(masking_strategy):
    with pytest.raises(ValidationError):
        SensitiveDataDetectionOptions(entities=["PERSON"], masking_strategy=masking_strategy)
