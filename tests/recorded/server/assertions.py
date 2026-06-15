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
