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

"""PII detection using Polygraf."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import action
from nemoguardrails.library.polygraf.request import polygraf_request
from nemoguardrails.rails.llm.config import PolygrafDetection

log = logging.getLogger(__name__)

# Placeholder returned when masking cannot complete safely (provider failure
# or a configured entity span is malformed). Replacing the entire payload is
# the only fail-closed option that guarantees no raw PII passes downstream.
FAILSAFE_MASK_PLACEHOLDER = "<REDACTED>"


def detect_pii_mapping(result: bool) -> bool:
    """
    Mapping for polygraf_detect_pii.

    Since the function returns True when PII is detected,
    we block if result is True.
    """
    return result


def _get_polygraf_api_key() -> Optional[str]:
    api_key = os.environ.get("POLYGRAF_API_KEY")
    if not api_key:
        log.warning(
            "POLYGRAF_API_KEY environment variable is not set. "
            "Polygraf cloud endpoints may reject unauthenticated requests."
        )
    return api_key


def _entity_shape(entity: Any) -> str:
    """Return a PII-free structural description of an entity for logging."""
    if isinstance(entity, dict):
        return f"dict(keys={sorted(entity.keys())})"
    return type(entity).__name__


def _is_int(value: Any) -> bool:
    """Strict integer check.

    ``bool`` is a subclass of ``int`` in Python, so a plain ``isinstance(x, int)``
    would accept ``True``/``False`` as valid offsets. Explicitly reject booleans
    so a Polygraf response cannot smuggle bogus span coordinates past validation.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _classify_entities(
    entities: List[Any],
    enabled_entities: Optional[List[str]],
    text_length: int,
) -> Tuple[List[Tuple[int, int, str]], bool]:
    """Split Polygraf entities into safe spans and report whether any
    span is unsafe enough to require failing closed.

    Args:
        entities: Raw entity records returned by Polygraf.
        enabled_entities: The configured entity-type filter (or ``None`` to
            accept every Polygraf-reported type).
        text_length: Length of the original payload, used to validate that
            integer offsets actually point inside the text.

    Returns:
        (safe_spans, has_malformed_selected)

        - safe_spans: ``(start, end, entity_type)`` triples for entities that
          pass the entity-type filter AND have a non-empty type AND have
          strict integer offsets satisfying ``0 <= start < end <= text_length``.
        - has_malformed_selected: ``True`` if any entity that *might* be a
          selected PII span is malformed. Callers must treat this as a
          fail-closed signal because there is no safe way to either trust
          a missing-type entity or silently drop it.
    """
    safe_spans: List[Tuple[int, int, str]] = []
    has_malformed_selected = False

    for entity in entities:
        if not isinstance(entity, dict):
            # Non-dict entries can't carry an entity_type or offsets we can
            # validate, so they always count as a malformed-selected span.
            # Log only the shape, never the value.
            log.warning("Skipping malformed Polygraf entity: shape=%s", _entity_shape(entity))
            has_malformed_selected = True
            continue

        entity_type = entity.get("entity_type")
        start = entity.get("start")
        end = entity.get("end")

        invalid_fields: List[str] = []
        if not isinstance(entity_type, str) or not entity_type:
            invalid_fields.append("entity_type")
        if not _is_int(start):
            invalid_fields.append("start")
        if not _is_int(end):
            invalid_fields.append("end")

        if invalid_fields:
            log.warning(
                "Skipping malformed Polygraf entity: invalid_fields=%s keys=%s",
                invalid_fields,
                sorted(entity.keys()),
            )
            # Fail closed conservatively:
            #  - Unknown entity_type: we cannot tell whether the filter would
            #    have selected it, so assume it would have.
            #  - Known entity_type missing from the filter: silently skip.
            if "entity_type" in invalid_fields:
                has_malformed_selected = True
            elif enabled_entities is None or entity_type in enabled_entities:
                has_malformed_selected = True
            continue

        # Offsets are now known-good integers. Validate they actually point
        # inside the text and form a non-empty, non-reversed span.
        if not (0 <= start < end <= text_length):
            log.warning(
                "Skipping malformed Polygraf entity: out-of-range span (start=%d end=%d text_length=%d) keys=%s",
                start,
                end,
                text_length,
                sorted(entity.keys()),
            )
            if enabled_entities is None or entity_type in enabled_entities:
                has_malformed_selected = True
            continue

        if enabled_entities and entity_type not in enabled_entities:
            continue

        safe_spans.append((start, end, entity_type))

    return safe_spans, has_malformed_selected


def _resolve_source_config(config: RailsConfig, source: str) -> Tuple[PolygrafDetection, Any, Optional[List[str]]]:
    """Resolve the Polygraf config and per-source entity filter, validating ``source``."""
    polygraf_config: PolygrafDetection = getattr(config.rails.config, "polygraf")
    source_config = getattr(polygraf_config, source, None)
    if source_config is None:
        valid_sources = ["input", "output", "retrieval"]
        raise ValueError(
            f"Polygraf can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source}', is not allowed."
        )
    enabled_entities = source_config.entities if source_config.entities else None
    return polygraf_config, source_config, enabled_entities


@action(is_system_action=False, output_mapping=detect_pii_mapping)
async def polygraf_detect_pii(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
) -> bool:
    """Checks whether the provided text contains any PII using Polygraf.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        True if PII is detected (or if the detection cannot complete safely),
        False otherwise.

    Raises:
        ValueError: Only if ``source`` is not one of the allowed flows.
            Provider/network failures are caught and treated as fail-closed
            (the action returns True so the rail blocks the message).
    """
    polygraf_config, _source_config, enabled_entities = _resolve_source_config(config, source)
    server_endpoint = polygraf_config.server_endpoint
    api_key = _get_polygraf_api_key()
    session = kwargs.get("session")

    try:
        entities: List[Dict[str, Any]] = await polygraf_request(text, server_endpoint, api_key, session=session)
    except ValueError as err:
        # Fail closed: a provider failure must not allow potentially-PII text
        # through. Log only the failure category, never the input text or
        # exception chain (which can contain response bodies with PII).
        log.warning("Polygraf detection failed (%s); failing closed and blocking text.", type(err).__name__)
        return True

    if not entities:
        return False

    safe_spans, has_malformed_selected = _classify_entities(entities, enabled_entities, len(text))

    # If a *selected* entity was malformed, treat the whole result as untrusted
    # and fail closed even if other valid entities had no enabled match.
    if has_malformed_selected:
        log.warning("Polygraf returned a malformed selected entity; failing closed and blocking text.")
        return True

    return len(safe_spans) > 0


@action(is_system_action=False)
async def polygraf_mask_pii(source: str, text: str, config: RailsConfig, **kwargs) -> str:
    """Masks any detected PII in the provided text using Polygraf.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        The altered text with PII masked. Returns ``FAILSAFE_MASK_PLACEHOLDER``
        when masking cannot complete safely (provider failure or a configured
        entity span is malformed), so raw PII is never sent downstream.

    Raises:
        ValueError: Only if ``source`` is not one of the allowed flows.
            Provider/network failures are caught and treated as fail-closed.
    """
    polygraf_config, _source_config, enabled_entities = _resolve_source_config(config, source)
    server_endpoint = polygraf_config.server_endpoint
    api_key = _get_polygraf_api_key()
    session = kwargs.get("session")

    try:
        entities: List[Dict[str, Any]] = await polygraf_request(text, server_endpoint, api_key, session=session)
    except ValueError as err:
        # Fail closed: if we cannot run masking at all, redact the entire text
        # rather than risk forwarding raw PII downstream.
        log.warning("Polygraf masking failed (%s); replacing payload with redaction placeholder.", type(err).__name__)
        return FAILSAFE_MASK_PLACEHOLDER

    if not entities:
        return text

    safe_spans, has_malformed_selected = _classify_entities(entities, enabled_entities, len(text))

    if has_malformed_selected:
        # A configured entity was reported with invalid offsets / type. We
        # cannot guarantee in-place masking, so fail closed by redacting the
        # entire payload instead of returning partially-masked text.
        log.warning("Polygraf returned a malformed selected entity; replacing payload with redaction placeholder.")
        return FAILSAFE_MASK_PLACEHOLDER

    masked_text = text
    for start, end, entity_type in sorted(safe_spans, key=lambda x: x[0], reverse=True):
        masked_text = masked_text[:start] + f"<{entity_type}>" + masked_text[end:]

    return masked_text
