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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from urllib.parse import urljoin

from nemoguardrails.http import HTTPClient, HTTPClientError, HTTPTimeoutError, http_call

log = logging.getLogger(__name__)


def join_nim_url(base_url: str, classification_path: str) -> str:
    """Join NIM base URL with classification path, handling trailing/leading slashes.

    Args:
        base_url: The base NIM URL (with or without trailing slash)
        classification_path: The classification endpoint path (with or without leading slash)

    Returns:
        Properly joined URL
    """
    # Ensure base_url ends with '/' for proper urljoin behavior
    normalized_base = base_url.rstrip("/") + "/"
    # Remove leading slash from classification path to ensure relative joining
    normalized_path = classification_path.lstrip("/")
    return urljoin(normalized_base, normalized_path)


async def jailbreak_detection_heuristics_request(
    prompt: str,
    api_url: str = "http://localhost:1337/heuristics",
    lp_threshold: Optional[float] = None,
    ps_ppl_threshold: Optional[float] = None,
    http_client: Optional[HTTPClient] = None,
):
    payload = {
        "prompt": prompt,
        "lp_threshold": lp_threshold,
        "ps_ppl_threshold": ps_ppl_threshold,
    }

    response = await http_call(http_client, "POST", api_url, json=payload, raise_for_status=False)
    if response.status_code != 200:
        log.error(f"Jailbreak check API request failed with status {response.status_code}")
        return None

    result = response.json()

    log.info(f"Prompt jailbreak check: {result}.")
    try:
        result = result["jailbreak"]
    except KeyError:
        log.exception("No jailbreak field in result.")
        result = None
    return result


async def jailbreak_detection_model_request(
    prompt: str,
    api_url: str = "http://localhost:1337/model",
    http_client: Optional[HTTPClient] = None,
):
    payload = {
        "prompt": prompt,
    }

    response = await http_call(http_client, "POST", api_url, json=payload, raise_for_status=False)
    if response.status_code != 200:
        log.error(f"Jailbreak check API request failed with status {response.status_code}")
        return None

    result = response.json()

    log.info(f"Prompt jailbreak check: {result}.")
    try:
        result = result["jailbreak"]
    except KeyError:
        log.exception("No jailbreak field in result.")
        result = None
    return result


async def jailbreak_nim_request(
    prompt: str,
    nim_url: str,
    nim_auth_token: Optional[str],
    nim_classification_path: str,
    http_client: Optional[HTTPClient] = None,
):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "input": prompt,
    }

    endpoint = join_nim_url(nim_url, nim_classification_path)
    try:
        if nim_auth_token is not None:
            headers["Authorization"] = f"Bearer {nim_auth_token}"
        response = await http_call(
            http_client,
            "POST",
            endpoint,
            json=payload,
            headers=headers,
            timeout=30,
            raise_for_status=False,
        )
        if response.status_code != 200:
            log.error(f"NemoGuard JailbreakDetect NIM request failed with status {response.status_code}")
            return None

        result = response.json()

        log.info(f"Prompt jailbreak check: {result}.")
        try:
            result = result["jailbreak"]
        except KeyError:
            log.exception("No jailbreak field in result.")
            result = None
        return result
    except HTTPTimeoutError:
        log.error("NemoGuard JailbreakDetect NIM request timed out")
        return None
    except HTTPClientError as e:
        log.error(f"NemoGuard JailbreakDetect NIM connection error: {str(e)}")
        return None
    except Exception as e:
        log.error(f"Unexpected error during NIM request: {str(e)}")
        return None
