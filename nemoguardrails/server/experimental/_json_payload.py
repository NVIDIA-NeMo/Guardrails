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
import math
from collections.abc import Mapping, Sequence
from typing import Any

from nemoguardrails.server.experimental.provider.types import JsonObject


class InvalidJson(ValueError):
    """Report a payload that is not standards-compliant JSON."""


class UnsupportedJsonShape(ValueError):
    """Report valid JSON that cannot be handled without ambiguity."""


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> JsonObject:
    """Build an object while rejecting duplicate member names."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise UnsupportedJsonShape(f"Duplicate JSON member {key!r} is not supported.")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str):
    """Reject JSON constants that are not part of the standard grammar."""
    raise InvalidJson(f"Non-standard JSON number {value!r} is not supported.")


def _parse_finite_float(value: str) -> float:
    """Parse a JSON float only when it has a finite value."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise InvalidJson(f"Non-finite JSON number {value!r} is not supported.")
    return parsed


def parse_json_object(body: bytes) -> JsonObject:
    """Parse a JSON object without accepting ambiguous or nonstandard input."""

    if not isinstance(body, bytes):
        raise TypeError("A provider JSON body must be bytes.")
    try:
        payload = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_number,
            parse_float=_parse_finite_float,
        )
    except UnsupportedJsonShape:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, InvalidJson) as error:
        raise InvalidJson("The payload must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise UnsupportedJsonShape("The JSON payload must be an object.")
    return payload


def encode_json_object(payload: Mapping[str, Any]) -> bytes:
    """Serialize a modified provider object as compact UTF-8 JSON."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
