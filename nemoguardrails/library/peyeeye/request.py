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

"""Module for handling Peyeeye PII redaction & rehydration requests."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

log = logging.getLogger(__name__)


class PEyeEyeGuardrailMissingSecrets(Exception):
    """Raised when the peyeeye API key is missing."""


class PEyeEyeGuardrailAPIError(Exception):
    """Raised when the peyeeye API returns an error."""


_BODY_SNIPPET_MAX = 200


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _safe_snippet(body_text: str, limit: int = _BODY_SNIPPET_MAX) -> str:
    """Return a truncated body snippet safe to embed in an exception message.

    Avoids dumping the full provider response (which may include detail strings
    a caller would prefer not to see in logs) while keeping enough context for
    debugging.
    """
    if body_text is None:
        return "<empty body, len=0>"
    text = body_text.strip()
    length = len(text)
    if length <= limit:
        return text
    return f"{text[:limit]}… <truncated, total len={length}>"


async def peyeeye_redact_request(
    texts: List[str],
    api_base: str,
    api_key: str,
    locale: str = "auto",
    entities: Optional[List[str]] = None,
    session_mode: str = "stateful",
    timeout: float = 15.0,
):
    """Send a redaction request to the peyeeye API.

    Args:
        texts: The list of texts to redact.
        api_base: The peyeeye API base URL (e.g. ``https://api.peyeeye.ai``).
        api_key: The bearer API key.
        locale: BCP-47 locale or ``"auto"``.
        entities: Optional list of entity IDs to restrict detection to.
        session_mode: ``"stateful"`` (default) or ``"stateless"``.
        timeout: Total request timeout in seconds.

    Returns:
        Tuple ``(redacted_texts, session_id)`` where ``session_id`` is either a
        ``ses_…`` (stateful) or ``skey_…`` (stateless) string, or ``None`` if
        the API didn't return one.

    Raises:
        PEyeEyeGuardrailMissingSecrets: If the API key is missing/invalid.
        PEyeEyeGuardrailAPIError: For all other API or transport failures.
    """
    if not api_key:
        raise PEyeEyeGuardrailMissingSecrets(
            "Peyeeye API key is required. Set the `PEYEEYE_API_KEY` environment variable."
        )

    body: Dict[str, Any] = {
        "text": list(texts),
        "locale": locale,
    }
    if entities:
        body["entities"] = list(entities)
    if session_mode == "stateless":
        body["session"] = "stateless"

    url = f"{api_base.rstrip('/')}/v1/redact"
    payload = await _post_json(url, body, _auth_headers(api_key), timeout=timeout)

    out_text = payload.get("text")
    if isinstance(out_text, str):
        redacted = [out_text]
    elif isinstance(out_text, list):
        redacted = [str(x) for x in out_text]
    else:
        raise PEyeEyeGuardrailAPIError(
            "peyeeye /v1/redact returned unexpected response shape; refusing to forward unredacted text"
        )

    if session_mode == "stateless":
        session_id = payload.get("rehydration_key")
    else:
        session_id = payload.get("session_id") or payload.get("session")

    return redacted, session_id


async def peyeeye_rehydrate_request(
    text: str,
    session_id: str,
    api_base: str,
    api_key: str,
    timeout: float = 15.0,
):
    """Send a rehydration request to the peyeeye API.

    Returns a dict ``{"text": <rehydrated>, "replaced": <int>}``.
    """
    if not api_key:
        raise PEyeEyeGuardrailMissingSecrets(
            "Peyeeye API key is required. Set the `PEYEEYE_API_KEY` environment variable."
        )
    if not session_id:
        raise PEyeEyeGuardrailAPIError("Cannot rehydrate without a session id.")

    url = f"{api_base.rstrip('/')}/v1/rehydrate"
    body = {"text": text, "session": session_id}
    payload = await _post_json(url, body, _auth_headers(api_key), timeout=timeout)

    # Use ``or text`` so a JSON ``null`` (key present, value None) falls back to
    # the original input rather than propagating ``None`` into ``$bot_message``.
    rehydrated = payload.get("text") or text
    replaced = payload.get("replaced", 0)
    try:
        replaced = int(replaced)
    except (TypeError, ValueError):
        replaced = 0
    return {"text": rehydrated, "replaced": replaced}


async def peyeeye_delete_session(
    session_id: str,
    api_base: str,
    api_key: str,
    timeout: float = 10.0,
) -> None:
    """Best-effort delete of a stateful peyeeye session.

    Errors are swallowed and logged at debug; this is cleanup.
    """
    if not session_id or not session_id.startswith("ses_"):
        return
    url = f"{api_base.rstrip('/')}/v1/sessions/{session_id}"
    headers = _auth_headers(api_key)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)):
                pass
    except Exception as e:  # noqa: BLE001 - best-effort cleanup
        log.debug("peyeeye: best-effort session cleanup failed: %s", e)


async def _post_json(
    url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    """POST JSON, raise ``PEyeEyeGuardrail*`` on failure, return parsed body."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                status = resp.status
                if status == 401:
                    raise PEyeEyeGuardrailMissingSecrets(f"Invalid peyeeye API key (401 from {url})")
                if status >= 400:
                    body_text = await resp.text()
                    snippet = _safe_snippet(body_text)
                    raise PEyeEyeGuardrailAPIError(f"peyeeye request to {url} failed with status {status}: {snippet}")
                try:
                    payload = await resp.json()
                except aiohttp.ContentTypeError as e:
                    body_text = await resp.text()
                    snippet = _safe_snippet(body_text)
                    raise PEyeEyeGuardrailAPIError(
                        f"peyeeye {url} returned non-JSON response (status={status}): {snippet}"
                    ) from e
                if not isinstance(payload, dict):
                    raise PEyeEyeGuardrailAPIError(
                        f"peyeeye {url} returned non-object JSON payload (status={status}, "
                        f"type={type(payload).__name__})"
                    )
                return payload
    except (PEyeEyeGuardrailMissingSecrets, PEyeEyeGuardrailAPIError):
        raise
    except aiohttp.ClientError as e:
        raise PEyeEyeGuardrailAPIError(f"peyeeye request to {url} failed: {e}") from e
    except TimeoutError as e:
        raise PEyeEyeGuardrailAPIError(f"peyeeye request to {url} timed out") from e
