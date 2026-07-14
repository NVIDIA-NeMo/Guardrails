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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import aiohttp
from typing_extensions import cast

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.rails.llm.config import F5GuardrailsRailConfig, RailsConfig

log = logging.getLogger(__name__)


def _scan_outcome(result: dict) -> RailOutcome:
    if result.get("result", {}).get("outcome") != "cleared":
        return RailOutcome.block(**result)
    return RailOutcome.allow(**result)


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
    return RailOutcome.allow(result={"outcome": "cleared"}, fail_open=True)


def _parse_retry_after(raw: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header value.

    Supports both delta-seconds (numeric) and HTTP-date forms per RFC 7231.
    Returns ``None`` if the value is missing or unparseable.
    """
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(str(raw))
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(tz=timezone.utc)).total_seconds()


def _compute_retry_delay(
    retry_after_raw: Optional[str],
    attempt: int,
    f5_config: F5GuardrailsRailConfig,
) -> float:
    """Compute the sleep duration before the next 429 retry."""
    parsed = _parse_retry_after(retry_after_raw)
    if parsed is None:
        delay = f5_config.retry_backoff_seconds * (2**attempt)
    else:
        delay = parsed
    if delay < 0:
        delay = 0.0
    if delay > f5_config.max_retry_after_seconds:
        delay = f5_config.max_retry_after_seconds
    return delay


@action(name="f5_guardrails_scan", is_system_action=True)
async def f5_guardrails_scan(
    text: str,
    config: Optional[RailsConfig] = None,
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

    timeout = aiohttp.ClientTimeout(total=30)
    max_attempts = f5_config.max_retries + 1
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(max_attempts):
            try:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status == 429 and attempt < max_attempts - 1:
                        retry_after_raw = response.headers.get("Retry-After")
                        delay = _compute_retry_delay(retry_after_raw, attempt, f5_config)
                        log.warning(
                            "F5 Guardrails API rate limited: status=429 attempt=%s/%s sleep_seconds=%.3f retry_after_present=%s",
                            attempt + 1,
                            max_attempts,
                            delay,
                            retry_after_raw is not None,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if response.status != 200:
                        content_type = response.headers.get("Content-Type", "unknown")
                        body_length = response.content_length if response.content_length is not None else "unknown"
                        log.error(
                            "F5 Guardrails API call failed: status=%s content_type=%s body_length=%s",
                            response.status,
                            content_type,
                            body_length,
                        )

                        if fail_open:
                            log.warning("F5 Guardrails API call failed; fail_open is enabled, allowing content.")
                            return _fail_open_outcome()

                        if response.status == 429:
                            raise RuntimeError("F5 Guardrails API rate limited (429) after exhausting retries")
                        raise RuntimeError(f"F5 Guardrails API error: {response.status}")

                    result = await response.json()
                    return _scan_outcome(result)
            except asyncio.TimeoutError:
                log.error("F5 Guardrails API call timed out after 30 seconds")

                if fail_open:
                    log.warning("F5 Guardrails API call timed out; fail_open is enabled, allowing content.")
                    return _fail_open_outcome()

                raise RuntimeError("F5 Guardrails API request timed out") from None
            except aiohttp.ClientError as e:
                log.error("Error connecting to F5 Guardrails API: %s", type(e).__name__)

                if fail_open:
                    log.warning("F5 Guardrails API call failed; fail_open is enabled, allowing content.")
                    return _fail_open_outcome()

                raise RuntimeError(f"Connection error to F5 Guardrails API: {type(e).__name__}") from e

    if fail_open:
        log.warning("F5 Guardrails API rate limited after retries; fail_open is enabled, allowing content.")
        return _fail_open_outcome()
    raise RuntimeError("F5 Guardrails API rate limited (429) after exhausting retries")
