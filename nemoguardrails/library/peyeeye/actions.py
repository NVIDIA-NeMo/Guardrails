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

"""PII redaction & rehydration using the Peyeeye API."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.peyeeye.request import (
    PEyeEyeGuardrailAPIError,
    PEyeEyeGuardrailMissingSecrets,
    peyeeye_delete_session,
    peyeeye_redact_request,
    peyeeye_rehydrate_request,
)

log = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.peyeeye.ai"
VALID_SOURCES = ("input", "output", "retrieval")


def _resolve_config(config: RailsConfig, source: str):
    """Pull the peyeeye config off the rails config object.

    Returns the per-source options object plus the shared api base.
    """
    peyeeye_cfg = getattr(getattr(config, "rails", None) and config.rails.config, "peyeeye", None)
    if peyeeye_cfg is None:
        raise PEyeEyeGuardrailAPIError("Peyeeye configuration is missing. Add `rails.config.peyeeye` to your config.")
    if source not in VALID_SOURCES:
        raise ValueError(f"Peyeeye source must be one of {VALID_SOURCES!r}; got {source!r}.")
    options = getattr(peyeeye_cfg, source)
    api_base = getattr(peyeeye_cfg, "api_base", None) or os.environ.get("PEYEEYE_API_BASE") or DEFAULT_API_BASE
    return options, api_base


def _resolve_api_key() -> str:
    api_key = os.environ.get("PEYEEYE_API_KEY")
    if not api_key:
        raise PEyeEyeGuardrailMissingSecrets(
            "PEYEEYE_API_KEY environment variable is required to use the peyeeye guardrail."
        )
    return api_key


def _normalize_texts(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, str):
        return [text]
    if isinstance(text, list):
        return [t if isinstance(t, str) else str(t) for t in text]
    raise ValueError(f"peyeeye_redact: `text` must be str or list[str]; got {type(text).__name__}.")


@action(is_system_action=False)
async def peyeeye_redact(
    source: str,
    text: Union[str, List[str]],
    config: RailsConfig,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Redact PII from a piece of input/output/retrieval text using peyeeye.

    The returned dict contains:

    * ``redacted_text``: the text with placeholders substituted (or the
      original ``list[str]`` input flattened to a list).
    * ``session_id``: a ``ses_…`` (stateful) or ``skey_…`` (stateless) string
      that callers must pass to :func:`peyeeye_rehydrate` to swap real values
      back into the model's response.
    * ``redacted``: ``True`` when peyeeye returned a usable session id.

    The function refuses to silently forward unredacted text: if the upstream
    response is malformed or the redacted/source list lengths disagree it
    raises :class:`PEyeEyeGuardrailAPIError`.
    """
    options, api_base = _resolve_config(config, source)
    api_key = _resolve_api_key()

    text_parts = _normalize_texts(text)
    if not any(t for t in text_parts):
        # Nothing to redact; preserve the input shape.
        return {
            "redacted_text": text,
            "session_id": None,
            "redacted": False,
        }

    redacted, session_id = await peyeeye_redact_request(
        texts=text_parts,
        api_base=api_base,
        api_key=api_key,
        locale=options.locale,
        entities=options.entities,
        session_mode=options.session_mode,
    )

    if len(redacted) != len(text_parts):
        raise PEyeEyeGuardrailAPIError(
            f"peyeeye /v1/redact returned {len(redacted)} texts for "
            f"{len(text_parts)} inputs; refusing to forward partially-"
            "redacted data."
        )

    if isinstance(text, str):
        redacted_text: Union[str, List[str]] = redacted[0]
    else:
        redacted_text = redacted

    if context is not None and session_id:
        # Stash the session id so the rehydrate flow can pick it up without
        # the caller having to thread it through every Colang variable.
        context_key = f"peyeeye_session_{source}"
        try:
            context[context_key] = session_id
        except Exception:  # noqa: BLE001 - context may be a read-only mapping
            log.debug("peyeeye: could not stash session id on context")

    return {
        "redacted_text": redacted_text,
        "session_id": session_id,
        "redacted": bool(session_id),
    }


@action(is_system_action=False)
async def peyeeye_rehydrate(
    text: str,
    session_id: str,
    config: RailsConfig,
    cleanup: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Rehydrate redacted text by swapping placeholders back to original PII.

    Returns ``{"text": <rehydrated>, "replaced": <int>}``. When ``cleanup`` is
    true and a stateful session was used, the server-side mapping is deleted
    on a best-effort basis after rehydration.

    Falls through with ``replaced=0`` if either ``text`` or ``session_id`` is
    empty/falsy — calling sites can safely invoke this on every response
    without first checking whether redaction happened.
    """
    if not text or not session_id:
        return {"text": text or "", "replaced": 0}

    peyeeye_cfg = getattr(
        getattr(config, "rails", None) and config.rails.config,
        "peyeeye",
        None,
    )
    api_base = (
        (getattr(peyeeye_cfg, "api_base", None) if peyeeye_cfg else None)
        or os.environ.get("PEYEEYE_API_BASE")
        or DEFAULT_API_BASE
    )
    api_key = _resolve_api_key()

    try:
        result = await peyeeye_rehydrate_request(
            text=text,
            session_id=session_id,
            api_base=api_base,
            api_key=api_key,
        )
    except PEyeEyeGuardrailMissingSecrets:
        raise
    except PEyeEyeGuardrailAPIError as e:
        log.warning("peyeeye: rehydrate failed: %s", e)
        return {"text": text, "replaced": 0}

    if cleanup and isinstance(session_id, str) and session_id.startswith("ses_"):
        await peyeeye_delete_session(
            session_id=session_id,
            api_base=api_base,
            api_key=api_key,
        )

    return result
