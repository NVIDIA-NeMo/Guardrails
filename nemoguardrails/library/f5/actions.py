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

import asyncio
import logging
import os
from typing import Any, Optional

from typing_extensions import cast

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.http import (
    HTTPClient,
    HTTPConnectionError,
    HTTPTimeoutError,
    RetryingHTTPClient,
    RetryPolicy,
    create_http_client,
    http_call,
    resolve_http_client,
)
from nemoguardrails.rails.llm.config import F5GuardrailsRailConfig, RailsConfig

log = logging.getLogger(__name__)


def _scan_outcome(result: dict) -> RailOutcome:
    if result.get("result", {}).get("outcome") != "cleared":
        return RailOutcome.block(metadata=result)
    return RailOutcome.allow(metadata=result)


def _get_f5_config(config: Optional[RailsConfig]) -> F5GuardrailsRailConfig:
    if config is None:
        return F5GuardrailsRailConfig()
    rails_config = getattr(config.rails, "config", None)
    f5_config = getattr(rails_config, "f5", None) if rails_config is not None else None
    if f5_config is None:
        return F5GuardrailsRailConfig()
    return cast(F5GuardrailsRailConfig, f5_config)


def _fail_open_outcome() -> RailOutcome:
    """Outcome returned when the API is unreachable and fail_open is enabled.

    The ``fail_open`` marker makes this distinguishable from a real cleared
    scan in logs and traces.
    """
    return RailOutcome.allow(metadata={"result": {"outcome": "cleared"}, "fail_open": True})


def _retry_policy(f5_config: F5GuardrailsRailConfig) -> RetryPolicy:
    max_delay = f5_config.max_retry_after_seconds
    return RetryPolicy(
        max_attempts=f5_config.max_retries + 1,
        retryable_methods=frozenset({"POST"}),
        retryable_status_codes=frozenset({429}),
        initial_delay=min(f5_config.retry_backoff_seconds, max_delay),
        max_delay=max_delay,
        max_retry_after=max_delay,
        retry_transport_errors=False,
        honor_retry_override_header=False,
        clamp_retry_after=True,
    )


def _create_http_client() -> HTTPClient:
    return create_http_client(timeout=30.0, retry_policy=RetryPolicy(max_attempts=1))


@action(name="f5_guardrails_scan", is_system_action=True)
async def f5_guardrails_scan(
    text: str,
    config: Optional[RailsConfig] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs: Any,
) -> RailOutcome:
    """
    Scans the provided text using the F5 Guardrails API.

    Args:
        text: The text to scan.
        config: The active RailsConfig; used to resolve ``rails.config.f5``.

    Returns:
        The rail decision with the F5 Guardrails API response as metadata.
        Fail-open outcomes include ``fail_open=True`` in their metadata.

    Resolution order for the API base URL:
        1. ``F5_GUARDRAILS_API_URL`` environment variable.
        2. ``rails.config.f5.api_url`` if set.
        3. The built-in default ``https://us1.calypsoai.app``.

    On HTTP 429 responses the action honors the ``Retry-After`` header
    (delta-seconds or HTTP-date) up to
    ``rails.config.f5.max_retry_after_seconds`` and retries up to
    ``rails.config.f5.max_retries`` additional times before applying the
    configured fail_open / fail_closed behavior.
    """
    f5_config = _get_f5_config(config)
    api_url = os.getenv("F5_GUARDRAILS_API_URL") or f5_config.api_url
    fail_open = f5_config.fail_open
    api_key = os.getenv("F5_GUARDRAILS_API_KEY")

    if not api_key:
        raise ValueError("F5 Guardrails API key not found. Please set F5_GUARDRAILS_API_KEY.")

    endpoint = f"{api_url.rstrip('/')}/backend/v1/scans"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": text,
    }

    try:
        async with resolve_http_client(http_client, factory=_create_http_client) as client:
            retrying_client = RetryingHTTPClient(
                client,
                _retry_policy(f5_config),
                sleep=asyncio.sleep,
                random_value=lambda: 1.0,
            )
            response = await http_call(
                retrying_client,
                "POST",
                endpoint,
                headers=headers,
                json=payload,
                timeout=30.0,
                raise_for_status=False,
            )
    except (HTTPTimeoutError, asyncio.TimeoutError):
        log.error("F5 Guardrails API call timed out after 30 seconds")

        if fail_open:
            log.warning("F5 Guardrails API call timed out; fail_open is enabled, allowing content.")
            return _fail_open_outcome()

        raise RuntimeError("F5 Guardrails API request timed out") from None
    except HTTPConnectionError as e:
        log.error("Error connecting to F5 Guardrails API: %s", type(e).__name__)

        if fail_open:
            log.warning("F5 Guardrails API call failed; fail_open is enabled, allowing content.")
            return _fail_open_outcome()

        raise RuntimeError(f"Connection error to F5 Guardrails API: {type(e).__name__}") from e

    if response.status_code != 200:
        content_type = next(
            (value for name, value in response.headers.items() if name.lower() == "content-type"),
            "unknown",
        )
        log.error(
            "F5 Guardrails API call failed: status=%s content_type=%s body_length=%s",
            response.status_code,
            content_type,
            len(response.content),
        )

        if fail_open:
            log.warning("F5 Guardrails API call failed; fail_open is enabled, allowing content.")
            return _fail_open_outcome()

        if response.status_code == 429:
            raise RuntimeError("F5 Guardrails API rate limited (429) after exhausting retries")
        raise RuntimeError(f"F5 Guardrails API error: {response.status_code}")

    return _scan_outcome(response.json())
