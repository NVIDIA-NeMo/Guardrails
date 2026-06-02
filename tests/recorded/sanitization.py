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

from __future__ import annotations

import re

FILTERED_HEADERS = {
    "authorization",
    "content-length",
    "x-api-key",
    "api-key",
    "openai-organization",
    "openai-project",
    "cookie",
    "set-cookie",
    "x-xet-access-token",
}

FILTERED_QUERY_PARAMETERS = ("api_key", "key", "token")

FILTERED_HEADER_PREFIXES = ("x-", "cf-", "openai-")

ALLOWED_HEADERS = {"content-type"}

VOLATILE_RESPONSE_HEADERS = {
    "content-length",
    "date",
    "nvcf-reqid",
    "request-id",
    "server-timing",
}

SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{12,}"), "[OPENAI_API_KEY]"),
    (re.compile(r"nvapi-[A-Za-z0-9_-]{12,}"), "[NVIDIA_API_KEY]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"\borg-[A-Za-z0-9_-]{6,}\b"), "[OPENAI_ORG]"),
    (re.compile(r"\bproj_[A-Za-z0-9_-]{6,}\b"), "[OPENAI_PROJECT]"),
)

JSON_SECRET_KEYS = {
    "access_token",
    "accesstoken",
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "token",
    "xet_access_token",
    "xetaccesstoken",
}

VOLATILE_RESPONSE_JSON_FIELDS = {
    "created": 0,
    "id": "[RECORDED_RESPONSE_ID]",
}

NULLABLE_VOLATILE_RESPONSE_JSON_FIELDS = {
    "system_fingerprint": "[RECORDED_SYSTEM_FINGERPRINT]",
}

VOLATILE_RESPONSE_METADATA_FIELDS = set(VOLATILE_RESPONSE_JSON_FIELDS) | set(NULLABLE_VOLATILE_RESPONSE_JSON_FIELDS)
