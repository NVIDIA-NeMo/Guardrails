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

"""Client for decision model endpoints using the state/questions/answers protocol."""

from dataclasses import dataclass
from typing import Dict, Optional

from nemoguardrails.http import (
    HTTPClient,
    HTTPClientError,
    HTTPResponseDecodeError,
    HTTPTimeoutError,
    HTTPTLSConfig,
    create_http_client,
    http_call,
)

AUTH_FAILURE_STATUS_CODES = frozenset({401, 403})


class DecisionModelAuthError(ValueError):
    """The decision model rejected the credentials (HTTP 401 or 403)."""


@dataclass(frozen=True)
class DecisionModelAssessment:
    scores: Dict[str, float]
    model: Optional[str] = None


async def decision_model_request(
    text: str,
    questions: Dict[str, str],
    server_endpoint: str,
    model: str,
    api_key: Optional[str],
    http_client: Optional[HTTPClient] = None,
    timeout: float = 10.0,
    verify_ssl: bool = True,
) -> DecisionModelAssessment:
    """Ask the decision model a batch of yes/no (Noul) questions about ``text``.

    All questions go in a single request to the configured endpoint.

    Args:
        text: The text to assess, sent in the "state" field.
        questions: Mapping of question id to the yes/no question instructions.
        server_endpoint: The assessment endpoint URL.
        model: The decision model identifier.
        api_key: The API key, sent as a bearer token. When ``None``, no Authorization header is sent.
        http_client: Optional shared HTTP client. Ignored when ``verify_ssl`` is false, because the
            shared client verifies certificates.
        timeout: Per-request timeout in seconds.
        verify_ssl: Verify the endpoint's TLS certificate. When false, the request uses its own
            client with verification disabled.

    Returns:
        Per-question probabilities (0-1) and the resolved model identifier, when supplied.

    Raises:
        DecisionModelAuthError: If the service answers 401 or 403.
        ValueError: If the call fails, times out, or the response is malformed.
            Messages never include the request text.
    """
    payload = {
        "state": text,
        "model": model,
        "questions": {qid: {"type": "noul", "instructions": instructions} for qid, instructions in questions.items()},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = await http_call(
            http_client if verify_ssl else None,
            "POST",
            server_endpoint,
            json=payload,
            headers=headers,
            timeout=timeout,
            raise_for_status=False,
            factory=lambda: create_http_client(timeout=timeout, tls=HTTPTLSConfig(verify=verify_ssl)),
        )
    except HTTPTimeoutError as err:
        raise ValueError(f"Decision model call timed out after {timeout} seconds.") from err
    except HTTPClientError as err:
        raise ValueError(f"Decision model call failed: {type(err).__name__}") from err

    if response.status_code in AUTH_FAILURE_STATUS_CODES:
        raise DecisionModelAuthError(
            f"Decision model authentication failed (HTTP {response.status_code}); "
            "check the API key and api_key_env_var."
        )
    if response.status_code != 200:
        # Deliberately omit the response body: it can echo request content.
        raise ValueError(f"Decision model call failed with status code {response.status_code}.")

    try:
        data = response.json()
    except HTTPResponseDecodeError as err:
        raise ValueError("Failed to parse Decision model response as JSON.") from err

    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("Invalid response from the decision model: missing 'answers' object.")

    scores: Dict[str, float] = {}
    for qid in questions:
        answer = answers.get(qid)
        value = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid response from the decision model: bad or missing noul answer for question '{qid}'."
            )
        scores[qid] = float(value)
    resolved_model = data.get("model")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        resolved_model = None
    return DecisionModelAssessment(scores=scores, model=resolved_model)
