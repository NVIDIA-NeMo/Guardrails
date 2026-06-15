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

from typing import Any


def normalize_chat_completion(response: dict[str, Any]) -> dict[str, Any]:
    """Strip server-generated volatile fields from a chat-completion response.

    The server mints its own ``id`` / ``created`` / ``system_fingerprint`` independent of
    the recorded provider traffic, so they are normalized to fixed sentinels before a
    snapshot. Everything else (object, model, choices, usage, guardrails) is left intact.
    """
    normalized = dict(response)
    if "id" in normalized:
        normalized["id"] = "[RECORDED_RESPONSE_ID]"
    if normalized.get("created") is not None:
        normalized["created"] = 0
    if normalized.get("system_fingerprint") is not None:
        normalized["system_fingerprint"] = "[RECORDED_SYSTEM_FINGERPRINT]"
    return normalized
