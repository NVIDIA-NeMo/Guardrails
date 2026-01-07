# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for PII utility functions.

These tests do not require the GLiNER model or server to be running.
"""

import pytest
from gliner_server.pii_utils import (
    DEFAULT_CATEGORIES,
    DEFAULT_LABELS,
    adjust_entity_positions,
    create_tagged_text,
    create_text_chunks,
    deduplicate_entities_by_score,
    process_raw_entities,
    remove_subset_entities,
)


class TestDefaultLabelsAndCategories:
    """Tests for validating DEFAULT_LABELS and DEFAULT_CATEGORIES consistency."""

    def test_all_category_labels_are_valid(self):
        """Ensure all labels in DEFAULT_CATEGORIES exist in DEFAULT_LABELS."""
        invalid_labels = []
        for category, labels in DEFAULT_CATEGORIES.items():
            for label in labels:
                if label not in DEFAULT_LABELS:
                    invalid_labels.append((category, label))

        assert not invalid_labels, (
            f"Found labels in DEFAULT_CATEGORIES that are not in DEFAULT_LABELS: {invalid_labels}"
        )


class TestCreateTaggedText:
    """Tests for the create_tagged_text function."""

    def test_empty_entities(self):
        """Test with no entities returns original text."""
        text = "Hello, my name is John."
        result = create_tagged_text(text, [])
        assert result == text

    def test_single_entity(self):
        """Test tagging a single entity."""
        text = "Hello, my name is John."
        entities = [{"value": "John", "suggested_label": "first_name", "start_position": 18, "end_position": 22}]
        result = create_tagged_text(text, entities)
        assert result == "Hello, my name is [John](first_name)."

    def test_multiple_entities(self):
        """Test tagging multiple entities."""
        text = "John's email is john@example.com"
        entities = [
            {"value": "John", "suggested_label": "first_name", "start_position": 0, "end_position": 4},
            {"value": "john@example.com", "suggested_label": "email", "start_position": 16, "end_position": 32},
        ]
        result = create_tagged_text(text, entities)
        assert result == "[John](first_name)'s email is [john@example.com](email)"

    def test_entities_out_of_order(self):
        """Test that entities are sorted by position."""
        text = "John's email is john@example.com"
        # Entities provided out of order
        entities = [
            {"value": "john@example.com", "suggested_label": "email", "start_position": 16, "end_position": 32},
            {"value": "John", "suggested_label": "first_name", "start_position": 0, "end_position": 4},
        ]
        result = create_tagged_text(text, entities)
        assert result == "[John](first_name)'s email is [john@example.com](email)"

    def test_overlapping_entities_skipped(self):
        """Test that overlapping entities are skipped."""
        text = "John Doe is here"
        entities = [
            {"value": "John Doe", "suggested_label": "full_name", "start_position": 0, "end_position": 8},
            {"value": "Doe", "suggested_label": "last_name", "start_position": 5, "end_position": 8},
        ]
        result = create_tagged_text(text, entities)
        # The second entity overlaps and should be skipped
        assert result == "[John Doe](full_name) is here"

    def test_custom_label_key(self):
        """Test using a custom label key."""
        text = "Hello, John!"
        entities = [{"value": "John", "custom_label": "person", "start_position": 7, "end_position": 11}]
        result = create_tagged_text(text, entities, label_key="custom_label")
        assert result == "Hello, [John](person)!"

    def test_entity_at_start(self):
        """Test entity at the start of text."""
        text = "John went home"
        entities = [{"value": "John", "suggested_label": "first_name", "start_position": 0, "end_position": 4}]
        result = create_tagged_text(text, entities)
        assert result == "[John](first_name) went home"

    def test_entity_at_end(self):
        """Test entity at the end of text."""
        text = "Contact John"
        entities = [{"value": "John", "suggested_label": "first_name", "start_position": 8, "end_position": 12}]
        result = create_tagged_text(text, entities)
        assert result == "Contact [John](first_name)"


class TestRemoveSubsetEntities:
    """Tests for the remove_subset_entities function."""

    def test_empty_list(self):
        """Test with empty list."""
        result = remove_subset_entities([])
        assert result == []

    def test_no_subsets(self):
        """Test with no subset entities."""
        entities = [
            {"start": 0, "end": 4, "label": "first_name"},
            {"start": 10, "end": 20, "label": "email"},
        ]
        result = remove_subset_entities(entities)
        assert len(result) == 2

    def test_remove_subset(self):
        """Test removing a subset entity."""
        entities = [
            {"start": 0, "end": 10, "label": "full_name"},  # Superset
            {"start": 5, "end": 10, "label": "last_name"},  # Subset - should be removed
        ]
        result = remove_subset_entities(entities)
        assert len(result) == 1
        assert result[0]["label"] == "full_name"

    def test_exact_overlap_keeps_first(self):
        """Test that exact overlaps keep one entity."""
        entities = [
            {"start": 0, "end": 10, "label": "name1"},
            {"start": 0, "end": 10, "label": "name2"},
        ]
        result = remove_subset_entities(entities)
        # One should be kept (the second is a "subset" of the first with same bounds)
        assert len(result) == 1

    def test_multiple_subsets(self):
        """Test removing multiple subsets."""
        entities = [
            {"start": 0, "end": 20, "label": "full_address"},
            {"start": 0, "end": 5, "label": "number"},
            {"start": 6, "end": 15, "label": "street"},
            {"start": 16, "end": 20, "label": "unit"},
        ]
        result = remove_subset_entities(entities)
        assert len(result) == 1
        assert result[0]["label"] == "full_address"

    def test_does_not_modify_original(self):
        """Test that original list is not modified."""
        original = [
            {"start": 0, "end": 10, "label": "full_name"},
            {"start": 5, "end": 10, "label": "last_name"},
        ]
        original_copy = [dict(e) for e in original]
        remove_subset_entities(original)
        assert original == original_copy


class TestDeduplicateEntitiesByScore:
    """Tests for the deduplicate_entities_by_score function."""

    def test_empty_list(self):
        """Test with empty list."""
        result = deduplicate_entities_by_score([])
        assert result == {}

    def test_single_entity(self):
        """Test with a single entity."""
        entities = [{"label": "email", "text": "john@example.com", "start": 0, "end": 16, "score": 0.95}]
        result = deduplicate_entities_by_score(entities)
        assert len(result) == 1
        key = ("email", "john@example.com")
        assert key in result
        assert result[key]["score"] == 0.95

    def test_keeps_higher_score(self):
        """Test that higher scored entity is kept."""
        entities = [
            {"label": "email", "text": "john@example.com", "start": 0, "end": 16, "score": 0.8},
            {"label": "email", "text": "john@example.com", "start": 50, "end": 66, "score": 0.95},
        ]
        result = deduplicate_entities_by_score(entities)
        assert len(result) == 1
        key = ("email", "john@example.com")
        assert result[key]["score"] == 0.95
        assert result[key]["start_position"] == 50

    def test_case_insensitive_dedup(self):
        """Test that deduplication is case-insensitive."""
        entities = [
            {"label": "first_name", "text": "John", "start": 0, "end": 4, "score": 0.9},
            {"label": "first_name", "text": "john", "start": 20, "end": 24, "score": 0.8},
        ]
        result = deduplicate_entities_by_score(entities)
        assert len(result) == 1
        # Higher score should be kept
        assert result[("first_name", "john")]["score"] == 0.9

    def test_different_labels_not_deduped(self):
        """Test that same text with different labels is not deduplicated."""
        entities = [
            {"label": "first_name", "text": "John", "start": 0, "end": 4, "score": 0.9},
            {"label": "user_name", "text": "John", "start": 0, "end": 4, "score": 0.8},
        ]
        result = deduplicate_entities_by_score(entities)
        assert len(result) == 2

    def test_skips_entities_without_label(self):
        """Test that entities without a label are skipped."""
        entities = [
            {"text": "John", "start": 0, "end": 4, "score": 0.9},
            {"label": "email", "text": "john@example.com", "start": 10, "end": 26, "score": 0.85},
        ]
        result = deduplicate_entities_by_score(entities)
        assert len(result) == 1
        assert ("email", "john@example.com") in result

    def test_output_format(self):
        """Test the output format is correct."""
        entities = [{"label": "email", "text": "test@test.com", "start": 5, "end": 18, "score": 0.876}]
        result = deduplicate_entities_by_score(entities)
        key = ("email", "test@test.com")
        assert result[key]["value"] == "test@test.com"
        assert result[key]["suggested_label"] == "email"
        assert result[key]["start_position"] == 5
        assert result[key]["end_position"] == 18
        assert result[key]["score"] == 0.876


class TestAdjustEntityPositions:
    """Tests for the adjust_entity_positions function."""

    def test_empty_list(self):
        """Test with empty list."""
        result = adjust_entity_positions([], 100)
        assert result == []

    def test_positive_offset(self):
        """Test adjusting with positive offset."""
        entities = [{"start": 0, "end": 5}, {"start": 10, "end": 15}]
        result = adjust_entity_positions(entities, 100)
        assert result[0]["start"] == 100
        assert result[0]["end"] == 105
        assert result[1]["start"] == 110
        assert result[1]["end"] == 115

    def test_zero_offset(self):
        """Test adjusting with zero offset."""
        entities = [{"start": 5, "end": 10}]
        result = adjust_entity_positions(entities, 0)
        assert result[0]["start"] == 5
        assert result[0]["end"] == 10

    def test_modifies_in_place(self):
        """Test that the function modifies entities in place."""
        entities = [{"start": 0, "end": 5}]
        result = adjust_entity_positions(entities, 10)
        assert entities[0]["start"] == 10  # Original is modified
        assert result is entities  # Same object


class TestProcessRawEntities:
    """Tests for the process_raw_entities function."""

    def test_empty_entities(self):
        """Test with empty entities."""
        result = process_raw_entities([], "Hello world")
        assert result.total_entities == 0
        assert result.entities == []
        assert result.tagged_text == "Hello world"

    def test_full_pipeline(self):
        """Test the full processing pipeline."""
        entities = [
            {"label": "first_name", "text": "John", "start": 0, "end": 4, "score": 0.95},
            {"label": "email", "text": "john@example.com", "start": 16, "end": 32, "score": 0.9},
        ]
        text = "John's email is john@example.com"
        result = process_raw_entities(entities, text)

        assert result.total_entities == 2
        assert len(result.entities) == 2
        assert "[John](first_name)" in result.tagged_text
        assert "[john@example.com](email)" in result.tagged_text

    def test_removes_subsets_and_deduplicates(self):
        """Test that subsets are removed and duplicates are deduplicated."""
        entities = [
            {"label": "full_name", "text": "John Doe", "start": 0, "end": 8, "score": 0.9},
            {"label": "first_name", "text": "John", "start": 0, "end": 4, "score": 0.95},  # Subset
            {"label": "email", "text": "john@example.com", "start": 20, "end": 36, "score": 0.8},
            {"label": "email", "text": "john@example.com", "start": 50, "end": 66, "score": 0.85},  # Duplicate
        ]
        text = "John Doe's email is john@example.com and also john@example.com"
        result = process_raw_entities(entities, text)

        # Should have 2 entities: full_name and email (deduplicated)
        assert result.total_entities == 2
        labels = {e.suggested_label for e in result.entities}
        assert labels == {"full_name", "email"}


class TestCreateTextChunks:
    """Tests for the create_text_chunks function."""

    def test_short_text_single_chunk(self):
        """Test that text shorter than chunk_length produces a single chunk."""
        text = "Hello world"
        chunks, offsets = create_text_chunks(text, chunk_length=100, overlap=20)

        assert len(chunks) == 1
        assert chunks[0] == "Hello world"
        assert offsets == [0]

    def test_text_exactly_chunk_length(self):
        """Test text exactly equal to chunk_length."""
        text = "A" * 100
        chunks, offsets = create_text_chunks(text, chunk_length=100, overlap=20)

        # With overlap, we get 2 chunks: 0-100 and 80-100
        assert len(chunks) == 2
        assert offsets == [0, 80]
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 20

    def test_long_text_multiple_chunks(self):
        """Test that long text is split into multiple overlapping chunks."""
        text = "A" * 250
        chunks, offsets = create_text_chunks(text, chunk_length=100, overlap=20)

        # With step = 100 - 20 = 80, chunks at: 0, 80, 160, 240
        assert len(chunks) == 4
        assert offsets == [0, 80, 160, 240]
        assert len(chunks[0]) == 100
        assert len(chunks[1]) == 100
        assert len(chunks[2]) == 90  # 250 - 160 = 90
        assert len(chunks[3]) == 10  # 250 - 240 = 10

    def test_overlap_creates_redundancy(self):
        """Test that overlapping regions contain the same content."""
        text = "0123456789ABCDEFGHIJ"  # 20 chars
        chunks, offsets = create_text_chunks(text, chunk_length=12, overlap=4)

        # With step = 12 - 4 = 8, chunks at: 0, 8, 16
        assert len(chunks) == 3
        assert offsets == [0, 8, 16]

        # Verify overlap region between chunk 0 and chunk 1
        # Chunk 0: positions 0-12, chunk 1: positions 8-20
        # Overlap is positions 8-12: "89AB"
        overlap_from_chunk0 = chunks[0][8:12]  # Last 4 chars of chunk 0
        overlap_from_chunk1 = chunks[1][0:4]  # First 4 chars of chunk 1
        assert overlap_from_chunk0 == overlap_from_chunk1 == "89AB"

    def test_default_parameters(self):
        """Test with default chunk_length=384 and overlap=128."""
        text = "X" * 500
        chunks, offsets = create_text_chunks(text)

        # With chunk_length=384 and overlap=128, step is 256
        # Chunks: 0-384, 256-500
        assert len(chunks) == 2
        assert offsets == [0, 256]
        assert len(chunks[0]) == 384
        assert len(chunks[1]) == 244  # 500 - 256

    def test_empty_text(self):
        """Test with empty text."""
        chunks, offsets = create_text_chunks("", chunk_length=100, overlap=20)

        assert chunks == []
        assert offsets == []

    def test_returns_tuple(self):
        """Test that the function returns a tuple of two lists."""
        result = create_text_chunks("test", chunk_length=10, overlap=2)

        assert isinstance(result, tuple)
        assert len(result) == 2
        chunks, offsets = result
        assert isinstance(chunks, list)
        assert isinstance(offsets, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
