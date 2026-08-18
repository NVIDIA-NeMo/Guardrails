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

from typing import Mapping, Optional

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


async def safe_read_body(response: aiohttp.ClientResponse, max_chars: Optional[int] = 500) -> str:
    """Read response body for error messages, truncating unless ``max_chars`` is None.

    Error classification passes None: a body cut mid-JSON no longer parses, and the
    provider's ``message``/``code``/``param`` would be lost in favour of the raw text.
    """
    try:
        text = await response.text()
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text
    except Exception:
        return "<could not read response body>"
