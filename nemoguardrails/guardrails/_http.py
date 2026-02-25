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

import aiohttp

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT_TOTAL = 30
DEFAULT_TIMEOUT_CONNECT = 5
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


async def safe_read_body(response: aiohttp.ClientResponse, max_chars: int = 500) -> str:
    """Read response body for error messages, truncating if too large."""
    try:
        text = await response.text()
        return text[:max_chars] if len(text) > max_chars else text
    except Exception:
        return "<could not read response body>"
