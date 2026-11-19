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

"""Tests for masking when detected PII spans overlap or are nested.

Overlapping or nested spans used to corrupt the string offsets of the
reverse-order replacements (e.g., producing "[PERSON]ME]") and leak raw PII.
Masking must first resolve the spans to a non-overlapping set.
"""

from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.library.gliner.actions import (
    _mask_text_with_entities,
    _resolve_overlapping_entities,
    gliner_mask_pii,
)
from nemoguardrails.library.polygraf.actions import (
    _resolve_overlapping_spans,
    polygraf_mask_pii,
)

# ---------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------


def _gliner_entity(label: str, start: int, end: int, score: float = 0.9) -> dict:
    return {
        "value": "",
        "suggested_label": label,
        "start_position": start,
        "end_position": end,
        "score": score,
    }


@pytest.mark.unit
def test_gliner_masking_with_nested_spans():
    """A nested span must not corrupt the replacement of its enclosing span."""
    text = "Contact John Smith at john@smith.com."
    entities = [
        _gliner_entity("person", 8, 18),  # "John Smith"
        _gliner_entity("last_name", 13, 18, score=0.8),  # "Smith", nested in person
        _gliner_entity("email", 22, 36),  # "john@smith.com"
    ]

    masked = _mask_text_with_entities(text, entities)

    assert masked == "Contact [PERSON] at [EMAIL]."
    assert "John" not in masked
    assert "Smith" not in masked
    assert "john@smith.com" not in masked


@pytest.mark.unit
def test_gliner_masking_with_overlapping_spans():
    """Overlapping spans keep only the earlier one; no offset corruption or PII leak."""
    text = "SSN: 123-45-6789 done"
    entities = [
        _gliner_entity("ssn", 5, 16),  # "123-45-6789"
        _gliner_entity("person", 10, 21),  # "45-6789 done", overlaps the SSN tail
    ]

    masked = _mask_text_with_entities(text, entities)

    assert masked == "SSN: [SSN] done"
    assert "123-45-6789" not in masked
    assert "45-6789" not in masked
    # No leftover placeholder fragments from corrupted offsets.
    assert "][" not in masked
    assert masked.count("]") == masked.count("[")


@pytest.mark.unit
def test_gliner_masking_prefers_longest_span_on_same_start():
    """When spans share a start offset, the longer span wins."""
    text = "john@doe.com is here"
    entities = [
        _gliner_entity("email", 0, 12),  # "john@doe.com"
        _gliner_entity("person", 0, 4),  # "john", same start, shorter
    ]

    masked = _mask_text_with_entities(text, entities)

    assert masked == "[EMAIL] is here"


@pytest.mark.unit
def test_gliner_masking_deduplicates_identical_spans():
    """Identical duplicate spans must produce a single placeholder."""
    text = "Contact john@doe.com now"
    entities = [
        _gliner_entity("email", 8, 20),
        _gliner_entity("email", 8, 20),
    ]

    masked = _mask_text_with_entities(text, entities)

    assert masked == "Contact [EMAIL] now"


@pytest.mark.unit
def test_gliner_masking_without_entities_returns_text_unchanged():
    assert _mask_text_with_entities("Hello world", []) == "Hello world"


@pytest.mark.unit
def test_gliner_resolve_overlapping_entities_drops_unsafe_spans():
    """Degenerate spans (empty, reversed, negative) are dropped, not replaced."""
    entities = [
        _gliner_entity("email", 8, 20),
        _gliner_entity("person", 10, 10),  # empty
        _gliner_entity("person", 15, 12),  # reversed
        _gliner_entity("person", -3, 4),  # negative start
    ]

    resolved = _resolve_overlapping_entities(entities)

    assert resolved == [entities[0]]


@pytest.mark.unit
def test_gliner_resolve_overlapping_entities_returns_ascending_order():
    entities = [
        _gliner_entity("email", 30, 40),
        _gliner_entity("person", 0, 10),
    ]

    resolved = _resolve_overlapping_entities(entities)

    assert [e["start_position"] for e in resolved] == [0, 30]


def _gliner_config() -> RailsConfig:
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:1235/v1/extract
                  input:
                    entities:
                      - email
            """,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gliner_mask_pii_action_masks_nested_spans():
    """The mask action must resolve nested spans from a GLiNER response."""
    config = _gliner_config()
    response = {
        "entities": [
            {
                "value": "John Smith",
                "suggested_label": "person",
                "start_position": 8,
                "end_position": 18,
                "score": 0.95,
            },
            {
                "value": "Smith",
                "suggested_label": "last_name",
                "start_position": 13,
                "end_position": 18,
                "score": 0.9,
            },
        ],
        "total_entities": 2,
    }

    with patch(
        "nemoguardrails.library.gliner.actions.gliner_request",
        new=AsyncMock(return_value=response),
    ):
        result = await gliner_mask_pii(source="input", text="Contact John Smith now", config=config)

    assert result.transform_text["user_message"] == "Contact [PERSON] now"


# ---------------------------------------------------------------------------
# Polygraf
# ---------------------------------------------------------------------------


def _polygraf_config() -> RailsConfig:
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
            """,
    )


def _patch_polygraf_request(monkeypatch, entities):
    async def mock_request(text, server_endpoint, api_key, http_client=None):
        return entities

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygraf_mask_pii_masks_nested_spans(monkeypatch):
    """A nested span must not corrupt the replacement of its enclosing span."""
    _patch_polygraf_request(
        monkeypatch,
        [
            {"entity_type": "Email", "entity_text": "john@doe.com", "start": 6, "end": 18},
            {"entity_type": "Person", "entity_text": "john", "start": 6, "end": 10},
        ],
    )

    result = await polygraf_mask_pii("input", "Email john@doe.com sent", _polygraf_config())

    assert result.transform_text["user_message"] == "Email <Email> sent"
    assert "john@doe.com" not in result.transform_text["user_message"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygraf_mask_pii_masks_overlapping_spans(monkeypatch):
    """Overlapping spans keep only the earlier one; no offset corruption or PII leak."""
    _patch_polygraf_request(
        monkeypatch,
        [
            # "123-45-6789" (entity type allowed by the unrestricted filter below)
            {"entity_type": "SSN", "entity_text": "123-45-6789", "start": 5, "end": 16},
            # "45-6789 done" overlaps the SSN tail
            {"entity_type": "Person", "entity_text": "45-6789 done", "start": 10, "end": 21},
        ],
    )
    # No entity filter: both reported types are selected for masking.
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input: {}
            """,
    )

    result = await polygraf_mask_pii("input", "SSN: 123-45-6789 done", config)

    assert result.transform_text["user_message"] == "SSN: <SSN> done"
    assert "123-45-6789" not in result.transform_text["user_message"]
    assert "45-6789" not in result.transform_text["user_message"]
    # No leftover placeholder fragments from corrupted offsets.
    assert ">" not in result.transform_text["user_message"].replace("<SSN>", "")


@pytest.mark.unit
def test_polygraf_resolve_overlapping_spans_keeps_non_overlapping_set():
    """Nested, overlapping, and duplicate spans reduce to one span per region."""
    spans = [
        (10, 21, "Person"),  # overlaps the SSN tail
        (5, 16, "SSN"),
        (5, 16, "SSN"),  # duplicate
        (22, 26, "Email"),  # disjoint
    ]

    resolved = _resolve_overlapping_spans(spans)

    assert resolved == [(5, 16, "SSN"), (22, 26, "Email")]


@pytest.mark.unit
def test_polygraf_resolve_overlapping_spans_prefers_longest_on_same_start():
    assert _resolve_overlapping_spans([(0, 4, "Person"), (0, 12, "Email")]) == [(0, 12, "Email")]
