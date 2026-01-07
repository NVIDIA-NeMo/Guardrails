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

"""PII detection utilities and helper functions for GLiNER entity extraction."""

from typing import Any

from .models import EntitySpan, GLiNERResponse

# Comprehensive PII labels
DEFAULT_LABELS = [
    "occupation",
    "certificate_license_number",
    "first_name",
    "date_of_birth",
    "ssn",
    "medical_record_number",
    "password",
    "unique_id",
    "phone_number",
    "national_id",
    "swift_bic",
    "company_name",
    "country",
    "license_plate",
    "tax_id",
    "employee_id",
    "pin",
    "state",
    "email",
    "date_time",
    "api_key",
    "biometric_identifier",
    "credit_debit_card",
    "coordinate",
    "device_identifier",
    "city",
    "postcode",
    "bank_routing_number",
    "vehicle_identifier",
    "health_plan_beneficiary_number",
    "url",
    "ipv4",
    "last_name",
    "cvv",
    "customer_id",
    "date",
    "user_name",
    "street_address",
    "ipv6",
    "account_number",
    "time",
    "age",
    "fax_number",
    "county",
    "gender",
    "sexuality",
    "political_view",
    "race_ethnicity",
    "religious_belief",
    "language",
    "blood_type",
    "mac_address",
    "http_cookie",
    "employment_status",
    "education_level",
]

DEFAULT_CATEGORIES = {
    "personal_identifiers": ["first_name", "last_name", "ssn", "date_of_birth"],
    "contact_info": [
        "email",
        "phone_number",
        "street_address",
        "city",
        "state",
    ],
    "financial": [
        "credit_debit_card",
        "cvv",
        "bank_routing_number",
        "account_number",
    ],
    "technical": ["ipv4", "ipv6", "mac_address", "url", "api_key"],
    "sensitive_attributes": [
        "gender",
        "sexuality",
        "race_ethnicity",
        "religious_belief",
    ],
}


def create_tagged_text(text: str, entities: list[dict[str, Any]], label_key: str = "suggested_label") -> str:
    """
    Create tagged text from original text and entities with positions.

    Args:
        text: Original text
        entities: List of entity dictionaries with 'value', label_key, 'start_position', 'end_position' keys
        label_key: The key to use for the entity label (default: 'suggested_label')

    Returns:
        Tagged text with format: [entity_text](entity_label)
    """
    if not entities:
        return text

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x["start_position"])

    tagged_text = ""
    position = 0

    for entity in sorted_entities:
        start = entity["start_position"]
        end = entity["end_position"]
        entity_text = entity["value"]
        entity_label = entity[label_key]

        # Skip if this entity starts before our current position (overlap)
        if start < position:
            continue

        # Add text before the entity
        tagged_text += text[position:start]

        # Add the tagged entity
        tagged_text += f"[{entity_text}]({entity_label})"

        # Update position
        position = end

    # Add remaining text
    tagged_text += text[position:]

    return tagged_text


def remove_subset_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove entities that are subsets of other entities (based on position).

    An entity is considered a subset if another entity fully contains its span.

    Args:
        entities: List of entity dictionaries with 'start' and 'end' keys

    Returns:
        List of entities with subset entities removed
    """
    if not entities:
        return []

    # Create a copy to avoid modifying the original
    entities = list(entities)
    entities_to_delete = []

    for idx, ent in enumerate(entities):
        has_superset = any(
            i != idx and i not in entities_to_delete and e["start"] <= ent["start"] and e["end"] >= ent["end"]
            for i, e in enumerate(entities)
        )
        if has_superset:
            entities_to_delete.append(idx)

    for idx in sorted(entities_to_delete, reverse=True):
        del entities[idx]

    return entities


def deduplicate_entities_by_score(entities: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """
    Deduplicate entities by keeping the highest scoring one for each (label, text) pair.

    Args:
        entities: List of entity dictionaries with 'label', 'text', 'start', 'end', 'score' keys

    Returns:
        Dictionary mapping (label, normalized_text) to entity span dictionaries
    """
    dedup_map = {}

    for ent in entities:
        label = ent.get("label")
        if not label:
            continue

        text_val = ent.get("text", "")
        score_val = float(ent.get("score", 0.0))
        key = (label, text_val.strip().lower())

        if key not in dedup_map or score_val > dedup_map[key]["score"]:
            dedup_map[key] = {
                "value": text_val,
                "suggested_label": label,
                "start_position": int(ent.get("start", 0)),
                "end_position": int(ent.get("end", 0)),
                "score": round(score_val, 3),
            }

    return dedup_map


def adjust_entity_positions(entities: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    """
    Adjust entity start/end positions by a given offset.

    Args:
        entities: List of entity dictionaries with 'start' and 'end' keys
        offset: The offset to add to start and end positions

    Returns:
        List of entities with adjusted positions
    """
    for entity in entities:
        entity["start"] += offset
        entity["end"] += offset
    return entities


def process_raw_entities(entities: list[dict[str, Any]], text: str) -> GLiNERResponse:
    """
    Process raw GLiNER entities into the final response format.

    This function:
    1. Removes subset entities
    2. Deduplicates by score
    3. Creates EntitySpan objects
    4. Generates tagged text

    Args:
        entities: Raw entities from GLiNER with 'start', 'end', 'label', 'text', 'score' keys
        text: The original text

    Returns:
        GLiNERResponse object
    """
    # Remove subset entities
    entities = remove_subset_entities(entities)

    # Deduplicate by score
    dedup_map = deduplicate_entities_by_score(entities)

    # Convert to list of EntitySpan objects
    entity_spans = [EntitySpan(**ent) for ent in dedup_map.values()]

    # Create tagged text
    tagged_text = create_tagged_text(text, list(dedup_map.values()))

    return GLiNERResponse(
        total_entities=len(entity_spans),
        entities=entity_spans,
        tagged_text=tagged_text,
    )


def create_text_chunks(text: str, chunk_length: int = 384, overlap: int = 128) -> tuple:
    """
    Split text into overlapping chunks for processing.

    Args:
        text: The text to split into chunks
        chunk_length: Maximum length of each chunk
        overlap: Number of characters to overlap between consecutive chunks

    Returns:
        Tuple of (chunks, offsets) where:
        - chunks: List of text chunks
        - offsets: List of starting positions for each chunk in the original text
    """
    chunks = []
    offsets = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_length])
        offsets.append(start)
        start += chunk_length - overlap
    return chunks, offsets
