# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from typing import Optional

import requests

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


@action(name="check_anchor_drift", is_system_action=False)
async def check_anchor_drift(
    context: Optional[dict] = None, threshold: float = 0.95
):
    """
    Checks the last bot message for intent drift using the anchor-engine API.

    Args:
        context: The current conversation context, containing the bot message
            and ground truth.
        threshold: The HHEM score threshold below which the action triggers a
            drift alert.
    """
    api_key = os.environ.get("ANCHOR_API_KEY")
    base_url = os.environ.get("ANCHOR_BASE_URL", "http://localhost:3000")
    if not api_key:
        raise ValueError(
            "Missing ANCHOR_API_KEY. Please provide your API key and set "
            "ANCHOR_BASE_URL to your on-premise Anchor Engine instance."
        )

    # Get the last bot message and the context/reference
    context = context or {}
    last_bot_message = context.get("last_bot_message")
    # Typical NeMo context key
    source_context = context.get("relevant_chunks", "")

    if not last_bot_message or not source_context:
        return True

    try:
        # Call the Anchor Scoring API
        # Target: On-premise Anchor Engine score endpoint
        response = await asyncio.to_thread(
            requests.post,
            f"{base_url}/api/score",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "action": last_bot_message,
                "context": source_context,
                "threshold": threshold,
            },
            timeout=5,
        )

        if response.status_code == 200:
            result = response.json()
            # If 'allow' is False, drift was detected
            return bool(result.get("allow", True))
        else:
            log.warning(
                "Anchor API returned non-200 status: %s",
                response.status_code,
            )
            return True

    except (requests.RequestException, ValueError, TypeError) as e:
        # Fail open to preserve UX, but keep diagnostics.
        log.warning("Anchor drift check failed: %s", e)
        return True

    return True
