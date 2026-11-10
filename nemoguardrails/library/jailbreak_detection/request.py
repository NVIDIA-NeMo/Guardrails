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

from typing import Optional
from urllib.parse import urljoin

from nemoguardrails.http import HTTPClient, HTTPClientError, HTTPResponse, HTTPResponseDecodeError, http_call
from nemoguardrails.library.jailbreak_detection.errors import JailbreakDetectionError
from nemoguardrails.library.jailbreak_detection.model_based.errors import JailbreakInputTooLongError
from nemoguardrails.library.jailbreak_detection.model_based.input_length import validate_input_length


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

    response = await _request_detector(http_client, api_url, payload)
    return _parse_jailbreak_response(response)


async def jailbreak_detection_model_request(
    prompt: str,
    api_url: str = "http://localhost:1337/model",
    http_client: Optional[HTTPClient] = None,
):
    validate_input_length(prompt)
    payload = {
        "prompt": prompt,
    }

    response = await _request_detector(http_client, api_url, payload)
    if response.status_code == 413:
        raise JailbreakInputTooLongError("Input exceeds the jailbreak detector's input limit.")
    return _parse_jailbreak_response(response)


async def jailbreak_nim_request(
    prompt: str,
    nim_url: str,
    nim_auth_token: Optional[str],
    nim_classification_path: str,
    http_client: Optional[HTTPClient] = None,
):
    validate_input_length(prompt)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "input": prompt,
    }

    endpoint = join_nim_url(nim_url, nim_classification_path)
    if nim_auth_token is not None:
        headers["Authorization"] = f"Bearer {nim_auth_token}"
    response = await _request_detector(http_client, endpoint, payload, headers=headers, timeout=30)
    if response.status_code == 413:
        raise JailbreakInputTooLongError("Input exceeds the jailbreak detector's input limit.")
    return _parse_jailbreak_response(response)


async def _request_detector(
    http_client: Optional[HTTPClient],
    endpoint: str,
    payload: dict,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> HTTPResponse:
    try:
        return await http_call(
            http_client, "POST", endpoint, json=payload, headers=headers, timeout=timeout, raise_for_status=False
        )
    except HTTPClientError as exc:
        raise JailbreakDetectionError("Jailbreak detector request failed.") from exc


def _parse_jailbreak_response(response: HTTPResponse) -> bool:
    if response.status_code != 200:
        raise JailbreakDetectionError(f"Jailbreak detector request failed with HTTP {response.status_code}.")
    try:
        result = response.json()
    except HTTPResponseDecodeError as exc:
        raise JailbreakDetectionError("Jailbreak detector returned invalid JSON.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("jailbreak"), bool):
        raise JailbreakDetectionError("Jailbreak detector response must contain a boolean jailbreak field.")
    return result["jailbreak"]
