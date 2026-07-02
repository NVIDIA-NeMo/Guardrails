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

import logging
from typing import Optional

from nemoguardrails.actions import action
from nemoguardrails.http import HTTPClient, http_call, resolve_http_client

log = logging.getLogger(__name__)


@action(name="alignscore request")
async def alignscore_request(
    api_url: str = "http://localhost:5000/alignscore_large",
    evidence: Optional[list] = None,
    response: Optional[str] = None,
    http_client: Optional[HTTPClient] = None,
):
    """Checks the facts for the bot response by making a request to the AlignScore API."""
    if not evidence:
        return 1.0

    payload = {"evidence": evidence, "claim": response}

    async with resolve_http_client(http_client) as client:
        http_response = await http_call(
            client,
            "POST",
            api_url,
            json=payload,
            raise_for_status=False,
        )
    if http_response.status_code != 200:
        log.error(f"AlignScore API request failed with status {http_response.status_code}")
        return None

    result = http_response.json()

    log.info(f"AlignScore was {result}.")
    try:
        result = result["alignscore"]
    except Exception:
        result = None
    return result
