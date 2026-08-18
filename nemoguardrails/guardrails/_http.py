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

"""Shared aiohttp helpers for IORails engine HTTP clients."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import aiohttp

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT_TOTAL = 30
DEFAULT_TIMEOUT_CONNECT = 5
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def merge_headers_case_insensitive(base: Mapping[str, str], overrides: Optional[Mapping[str, str]]) -> dict[str, str]:
    """Merge ``overrides`` onto a copy of ``base``, matching header names case-insensitively.

    HTTP header names are case-insensitive, so an override replaces any base
    header with the same name regardless of case, adopting the override's
    casing (mirrors ``BaseClient._build_headers``). ``base`` is not mutated;
    ``None`` or empty ``overrides`` yields a plain copy of ``base``.
    """
    merged = dict(base)
    for name, value in (overrides or {}).items():
        for existing in [key for key in merged if key.lower() == name.lower()]:
            del merged[existing]
        merged[name] = value
    return merged


def normalize_query_params(query: Any) -> tuple[tuple[str, str], ...]:
    """Normalize a configured ``default_query`` mapping into ``(name, value)`` pairs for aiohttp's ``params=``."""
    if query is None:
        return ()
    if not isinstance(query, Mapping):
        raise ValueError(f"default_query must be a mapping of parameter names to values, got {type(query).__name__}.")

    pairs: list[tuple[str, str]] = []
    for key, value in query.items():
        name = str(key)
        # A list repeats the key, as httpx does; str and bytes are scalars, not
        # sequences of characters or ints.
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pairs.extend((name, _query_value_to_str(name, element)) for element in value)
        else:
            pairs.append((name, _query_value_to_str(name, value)))
    return tuple(pairs)


def _query_value_to_str(name: str, value: Any) -> str:
    """Coerce one query-parameter value to the string form httpx would put on the wire."""
    # httpx lowercases booleans and renders None as an empty value; yarl raises
    # TypeError on both, so the coercion happens here to keep the two engines in
    # step.  bool is an int subclass, so it must be tested before the str() fallback.
    if isinstance(value, Mapping):
        raise ValueError(
            f"default_query['{name}'] must be a scalar or a list of scalars, not a nested object. "
            "Flatten nested values into separate query parameters."
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


async def safe_read_body(response: aiohttp.ClientResponse, max_chars: int = 500) -> str:
    """Read response body for error messages, truncating if too large."""
    try:
        text = await response.text()
        return text[:max_chars] if len(text) > max_chars else text
    except Exception:
        return "<could not read response body>"
