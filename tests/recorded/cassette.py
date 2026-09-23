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

from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SMART_CHAR_MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "--",
    "…": "...",
}
SMART_CHAR_TRANS = str.maketrans(SMART_CHAR_MAP)


def normalize_smart_chars(text: str) -> str:
    """Map smart punctuation to stable ASCII forms for comparisons."""
    return unicodedata.normalize("NFKC", text.translate(SMART_CHAR_TRANS))


def normalize_body(value: Any) -> Any:
    """Normalize JSON-like payloads for stable assertions and request matching."""
    if isinstance(value, str):
        return normalize_smart_chars(value)
    if isinstance(value, dict):
        return {key: normalize_body(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [normalize_body(item) for item in value]
    return value


@dataclass(frozen=True)
class RecordedChatResponse:
    """Parsed chat-completion response extracted from a cassette interaction.

    ``usage`` is normalized across provider conventions (input/output/total tokens);
    ``raw_usage`` keeps the original payload for assertions that need it.
    """

    content: str
    usage: dict[str, int | None]
    model: str | None
    finish_reason: str | None
    request_id: str | None
    raw_usage: dict[str, Any] | None


def decode_body_json(body: Any) -> Any:
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if isinstance(body, str):
        return json.loads(body)
    if isinstance(body, dict):
        if "parsed_body" in body:
            return body["parsed_body"]
        if "string" in body:
            return decode_body_json(body["string"])
        return body
    return None


def decode_body_text(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8")
    if isinstance(body, str):
        return body
    if isinstance(body, dict) and "string" in body:
        return decode_body_text(body["string"])
    return ""


def _request_json(interaction: dict[str, Any]) -> Any:
    request = interaction.get("request", {})
    if "parsed_body" in request:
        return request["parsed_body"]
    return decode_body_json(request.get("body"))


def _header_values(headers: dict[str, Any], name: str) -> list[str]:
    for key, value in headers.items():
        if key.lower() == name:
            return value if isinstance(value, list) else [value]
    return []


def _is_sse_response(response: dict[str, Any]) -> bool:
    content_types = [value.lower() for value in _header_values(response.get("headers", {}), "content-type")]
    return any("text/event-stream" in value for value in content_types)


def _json_body(value: Any) -> Any:
    try:
        return decode_body_json(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None


def _sse_body_payloads(text: str) -> list[Any] | None:
    """Parse only lossless single-line ``data:`` SSE streams.

    Returning ``None`` leaves the original body untouched, which avoids rewriting
    event ids, comments, multi-line data blocks, or other SSE features this
    serializer cannot round-trip exactly.
    """
    if not text.endswith("\n\n"):
        return None

    parts = text.split("\n\n")
    if parts[-1] != "" or any(not event for event in parts[:-1]):
        return None

    payloads = []
    for event in parts[:-1]:
        lines = event.splitlines()
        if len(lines) != 1:
            return None

        line = lines[0]
        if not line.startswith("data: "):
            return None

        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            payloads.append("[DONE]")
            continue
        try:
            payloads.append(json.loads(payload))
        except json.JSONDecodeError:
            return None
    return payloads


def _sse_payloads_body(payloads: list[Any]) -> str:
    """Rehydrate parsed SSE payloads using the strict format accepted above."""
    events = []
    for payload in payloads:
        if payload == "[DONE]":
            events.append("data: [DONE]")
        else:
            events.append("data: " + json.dumps(payload, separators=(",", ":")))
    return "\n\n".join(events) + "\n\n"


def _json_body_text(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


def cassette_with_parsed_bodies(cassette: dict[str, Any]) -> dict[str, Any]:
    """Store JSON bodies as readable ``parsed_body`` blocks in committed cassettes.

    SSE responses are converted only when they match the strict format that can
    be rehydrated without changing stream semantics.
    """
    cassette = deepcopy(cassette)
    for interaction in cassette.get("interactions") or []:
        request = interaction.get("request", {})
        request_body = request.get("body")
        request_data = _json_body(request_body)
        if request_data is not None:
            request["parsed_body"] = request_data
            request.pop("body", None)

        response = interaction.get("response", {})
        body = response.get("body")
        if not isinstance(body, dict):
            continue
        body_text = decode_body_text(body)
        if _is_sse_response(response) and body_text:
            payloads = _sse_body_payloads(body_text)
            if payloads is not None:
                body["parsed_body"] = payloads
                body.pop("string", None)
            continue
        response_data = _json_body(body)
        if response_data is not None:
            body["parsed_body"] = response_data
            body.pop("string", None)
    return cassette


def cassette_with_rehydrated_bodies(cassette: dict[str, Any]) -> dict[str, Any]:
    """Convert readable cassette bodies back to raw strings for VCR replay."""
    cassette = deepcopy(cassette)
    for interaction in cassette.get("interactions") or []:
        request = interaction.get("request", {})
        if "parsed_body" in request:
            request["body"] = _json_body_text(request["parsed_body"])
            request.pop("parsed_body", None)

        response = interaction.get("response", {})
        body = response.get("body")
        if not isinstance(body, dict) or "parsed_body" not in body:
            continue
        parsed_body = body["parsed_body"]
        if _is_sse_response(response) and isinstance(parsed_body, list):
            body["string"] = _sse_payloads_body(parsed_body)
        else:
            body["string"] = _json_body_text(parsed_body)
        body.pop("parsed_body", None)
    return cassette


@lru_cache(maxsize=None)
def _cached_cassette_interactions(cassette_path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(cassette_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    return data.get("interactions") or []


def _cassette_interactions(cassette_path: Path) -> list[dict[str, Any]]:
    return deepcopy(_cached_cassette_interactions(cassette_path))


def cassette_request_jsons(cassette_path: Path) -> list[dict[str, Any]]:
    """Decode every request body in the cassette as JSON; skip non-JSON bodies."""
    bodies = []
    for interaction in _cassette_interactions(cassette_path):
        payload = _request_json(interaction)
        if isinstance(payload, dict):
            bodies.append(payload)
    return bodies


def _normalize_usage(raw_usage: dict[str, Any] | None) -> dict[str, int | None]:
    if raw_usage is None:
        return {}
    input_tokens = raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))
    output_tokens = raw_usage.get("output_tokens", raw_usage.get("completion_tokens"))
    total_tokens = raw_usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    usage: dict[str, int | None] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    completion_details = raw_usage.get("completion_tokens_details") or {}
    cached_tokens = raw_usage.get("cached_tokens", prompt_details.get("cached_tokens"))
    reasoning_tokens = raw_usage.get("reasoning_tokens", completion_details.get("reasoning_tokens"))
    if cached_tokens is not None:
        usage["cached_tokens"] = cached_tokens
    if reasoning_tokens is not None:
        usage["reasoning_tokens"] = reasoning_tokens
    return usage


def _stream_payloads(text: str) -> list[dict[str, Any]]:
    payloads = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def stream_payloads_from_body(body: Any) -> list[dict[str, Any]]:
    """Return parsed streaming payloads from either readable or raw cassette bodies."""
    if isinstance(body, dict) and isinstance(body.get("parsed_body"), list):
        return [payload for payload in body["parsed_body"] if isinstance(payload, dict)]
    return _stream_payloads(decode_body_text(body))


def _non_streaming_chat_response(interaction: dict[str, Any]) -> RecordedChatResponse | None:
    body = interaction.get("response", {}).get("body")
    payload = _json_body(body)
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw_usage = payload.get("usage")
    return RecordedChatResponse(
        content=message.get("content") or "",
        usage=_normalize_usage(raw_usage),
        model=payload.get("model"),
        finish_reason=choice.get("finish_reason"),
        request_id=payload.get("id"),
        raw_usage=raw_usage,
    )


def _streaming_chat_response(interaction: dict[str, Any]) -> RecordedChatResponse:
    body = interaction.get("response", {}).get("body")
    payloads = stream_payloads_from_body(body)
    content_parts = []
    raw_usage = None
    model = None
    finish_reason = None
    request_id = None
    for payload in payloads:
        model = payload.get("model") or model
        request_id = payload.get("id") or request_id
        if payload.get("usage"):
            raw_usage = payload["usage"]
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            finish_reason = choice.get("finish_reason") or finish_reason
    return RecordedChatResponse(
        content="".join(content_parts),
        usage=_normalize_usage(raw_usage),
        model=model,
        finish_reason=finish_reason,
        request_id=request_id,
        raw_usage=raw_usage,
    )


def recorded_chat_response(
    cassette_path: Path,
    *,
    request_model: str | None = None,
    stream: bool = False,
) -> RecordedChatResponse:
    """Return the last recorded chat-completion in the cassette matching the filters.

    Selects interactions whose request payload has the given ``request_model`` (if
    set) and whose ``stream`` flag matches. Asserts at least one match exists.
    """
    matches = []
    for interaction in _cassette_interactions(cassette_path):
        request_payload = _request_json(interaction)
        if not isinstance(request_payload, dict):
            continue
        if request_model and request_payload.get("model") != request_model:
            continue
        if (request_payload.get("stream") is True) != stream:
            continue
        parser = _streaming_chat_response if stream else _non_streaming_chat_response
        response = parser(interaction)
        if response is not None:
            matches.append(response)
    assert matches, f"{cassette_path} does not contain a {'streaming' if stream else 'non-streaming'} chat response"
    return matches[-1]
