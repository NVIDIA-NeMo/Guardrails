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

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from vcr.request import Request

from tests.recorded.conftest import (
    ReadableYamlSerializer,
    _provider_key_fixture_name,
    before_record_request,
    before_record_response,
    build_vcr_config,
    recorded_body_matcher,
)
from tests.recorded.sanitization import FILTERED_HEADERS
from tests.recorded.utils import api_key_for_record_mode, set_api_key_for_record_mode

pytestmark = [pytest.mark.recorded]

RECORDED_DIR = Path(__file__).parent

FORBIDDEN_HEADER_NAMES = FILTERED_HEADERS

FORBIDDEN_PATTERNS = {
    "openai_api_key": r"\bsk-[A-Za-z0-9_-]{12,}\b",
    "nvidia_api_key": r"\bnvapi-[A-Za-z0-9_-]{12,}\b",
    "bearer_token": r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b",
    "openai_org": r"\borg-[A-Za-z0-9_-]{6,}\b",
    "openai_project": r"\bproj_[A-Za-z0-9_-]{6,}\b",
    "query_secret": r"[?&](api_key|key|token)=[^&\s]+",
    "xet_access_token": r'("accessToken"\s*:\s*"(?!\[REDACTED\])[^"]+"|\baccessToken\s*:\s*(?![\'"]?\[REDACTED\][\'"]?)[^\s]+)',
    "aws_presigned_url": r"[?&]X-Amz-(Credential|Security-Token|Signature)=",
    "unexpected_huggingface_host": r"https://(?:[^/\s]+\.)?huggingface\.co",
    "volatile_chat_response_id": r'"id"\s*:\s*"chat(?:cmpl)?-[^"]+"',
    "volatile_created_timestamp": r'"created"\s*:\s*[1-9]\d{8,}',
}

_COMBINED_FORBIDDEN = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in FORBIDDEN_PATTERNS.items()),
    re.IGNORECASE,
)


def _cassette_headers(data: Any) -> list[dict]:
    interactions = data.get("interactions", []) if isinstance(data, dict) else []
    headers = []
    for interaction in interactions:
        for side in ("request", "response"):
            section = interaction.get(side, {}) if isinstance(interaction, dict) else {}
            section_headers = section.get("headers") if isinstance(section, dict) else None
            if isinstance(section_headers, dict):
                headers.append(section_headers)
    return headers


def test_recorded_cassettes_are_sanitized():
    cassette_paths = sorted(RECORDED_DIR.rglob("cassettes/**/*.yaml"))
    if not cassette_paths:
        pytest.skip("No recorded cassettes committed in this branch")

    failures = []
    for path in cassette_paths:
        text = path.read_text(encoding="utf-8")
        for match in _COMBINED_FORBIDDEN.finditer(text):
            failures.append(f"{path}: matched forbidden pattern {match.lastgroup}")

        data = yaml.safe_load(text)
        for headers in _cassette_headers(data):
            for header in headers:
                if header.lower() in FORBIDDEN_HEADER_NAMES:
                    failures.append(f"{path}: contains forbidden header {header}")

    assert not failures, "\n".join(failures)


def test_recorded_cassette_serializer_keeps_json_bodies_readable():
    response = before_record_response(
        {
            "headers": {"Content-Length": ["100"], "Content-Type": ["application/json"]},
            "body": {"string": '{"id":"chatcmpl-123","created":1770000000,"answer":"ok"}'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"messages":[{"role":"user","content":"hello"}]}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.openai.com/v1/chat/completions",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    text = ReadableYamlSerializer.serialize(cassette)

    assert "Content-Length" not in text
    assert "parsed_body:" in text
    assert "id: '[RECORDED_RESPONSE_ID]'" in text
    assert "created: 0" in text
    loaded = yaml.safe_load(text)
    assert "string" not in loaded["interactions"][0]["response"]["body"]
    assert ReadableYamlSerializer.deserialize(text)["interactions"][0]["response"]["body"]["string"].startswith("{")


def test_recorded_cassette_serializer_handles_null_interactions():
    cassette = {"version": 1, "interactions": None}

    assert yaml.safe_load(ReadableYamlSerializer.serialize(cassette)) == cassette
    assert ReadableYamlSerializer.deserialize("version: 1\ninteractions:\n") == cassette


def test_recorded_cassette_serializer_redacts_access_tokens_from_parsed_bodies():
    request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body='{"accessToken":"request-access-token-1234567890","nested":{"xetAccessToken":"request-xet-token-1234567890"}}',
        headers={"Content-Type": "application/json"},
    )
    response = before_record_response(
        {
            "headers": {"Content-Type": ["application/json"]},
            "body": {
                "string": '{"access_token":"response-access-token-1234567890","nested":{"accessToken":"response-xet-token-1234567890"}}'
            },
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": before_record_request(request).body,
                    "headers": {"Content-Type": ["application/json"]},
                    "method": "POST",
                    "uri": "https://api.openai.com/v1/chat/completions",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    text = ReadableYamlSerializer.serialize(cassette)
    loaded = yaml.safe_load(text)

    assert "request-access-token-1234567890" not in text
    assert "request-xet-token-1234567890" not in text
    assert "response-access-token-1234567890" not in text
    assert "response-xet-token-1234567890" not in text
    assert loaded["interactions"][0]["request"]["parsed_body"]["accessToken"] == "[REDACTED]"
    assert loaded["interactions"][0]["request"]["parsed_body"]["nested"]["xetAccessToken"] == "[REDACTED]"
    assert loaded["interactions"][0]["response"]["body"]["parsed_body"]["access_token"] == "[REDACTED]"
    assert loaded["interactions"][0]["response"]["body"]["parsed_body"]["nested"]["accessToken"] == "[REDACTED]"


def test_recorded_cassette_serializer_preserves_smart_chars():
    request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body='{"prompt":"‘prompt’ “text” – — ‑ …"}',
        headers={"Content-Type": "application/json"},
    )
    response = before_record_response(
        {
            "headers": {"Content-Type": ["application/json"]},
            "body": {"string": '{"answer":"‘a’ “b” – — ‑ …"}'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": before_record_request(request).body,
                    "headers": {"Content-Type": ["application/json"]},
                    "method": "POST",
                    "uri": "https://api.openai.com/v1/chat/completions",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    text = ReadableYamlSerializer.serialize(cassette)
    loaded = yaml.safe_load(text)
    prompt = loaded["interactions"][0]["request"]["parsed_body"]["prompt"]
    answer = loaded["interactions"][0]["response"]["body"]["parsed_body"]["answer"]

    assert prompt == "‘prompt’ “text” – — ‑ …"
    assert answer == "‘a’ “b” – — ‑ …"


def test_recorded_body_matcher_normalizes_smart_chars_in_json_bodies():
    cassette_request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body=json.dumps({"messages": [{"role": "user", "content": 'It\'s a "smart" quote... - -- -'}]}),
        headers={"Content-Type": "application/json"},
    )
    replay_request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body=json.dumps({"messages": [{"role": "user", "content": "It’s a “smart” quote… – — ‑"}]}),
        headers={"Content-Type": "application/json"},
    )

    recorded_body_matcher(cassette_request, replay_request)


def test_recorded_cassette_serializer_filters_headers_by_prefix():
    response = before_record_response(
        {
            "headers": {
                "Content-Type": ["application/json"],
                "x-request-id": ["req-123"],
                "X-Content-Type-Options": ["nosniff"],
                "cf-cache-status": ["HIT"],
                "openai-version": ["2020-10-01"],
            },
            "body": {"string": '{"ok":true}'},
        }
    )
    headers = response["headers"]

    assert "Content-Type" in headers
    assert "x-request-id" not in headers
    assert "X-Content-Type-Options" not in headers
    assert "cf-cache-status" not in headers
    assert "openai-version" not in headers

    request = before_record_request(
        Request(
            method="POST",
            uri="https://api.openai.com/v1/chat/completions",
            body='{"prompt":"hi"}',
            headers={"Content-Type": "application/json", "x-stainless-os": "MacOS"},
        )
    )

    assert "x-stainless-os" not in request.headers
    assert "Content-Type" in request.headers


def test_recorded_cassette_serializer_keeps_sse_bodies_parseable():
    response = before_record_response(
        {
            "headers": {"Content-Type": ["text/event-stream"]},
            "body": {"string": 'data: {"id":"chatcmpl-123","created":1770000000,"choices":[]}\n\ndata: [DONE]\n\n'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"stream":true}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.openai.com/v1/chat/completions",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    text = ReadableYamlSerializer.serialize(cassette)
    loaded = yaml.safe_load(text)
    response_body = loaded["interactions"][0]["response"]["body"]

    assert "string" not in response_body
    assert response_body["parsed_body"][0]["id"] == "[RECORDED_RESPONSE_ID]"
    assert response_body["parsed_body"][0]["created"] == 0
    assert response_body["parsed_body"][-1] == "[DONE]"
    assert "data: [DONE]" in ReadableYamlSerializer.deserialize(text)["interactions"][0]["response"]["body"]["string"]


def test_recorded_cassette_serializer_preserves_non_strict_sse_bodies():
    response = before_record_response(
        {
            "headers": {"Content-Type": ["text/event-stream"]},
            "body": {"string": 'event: message\ndata: {"id":"chatcmpl-123","created":1770000000,"choices":[]}\n\n'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"stream":true}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.openai.com/v1/chat/completions",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    text = ReadableYamlSerializer.serialize(cassette)
    response_body = yaml.safe_load(text)["interactions"][0]["response"]["body"]

    assert "parsed_body" not in response_body
    assert response_body["string"] == (
        'event: message\ndata: {"id":"[RECORDED_RESPONSE_ID]","created":0,"choices":[]}\n\n'
    )


def test_recorded_response_metadata_normalization_preserves_nested_ids():
    response = before_record_response(
        {
            "headers": {"Content-Type": ["application/json"]},
            "body": {
                "string": (
                    '{"id":"chatcmpl-123","created":1770000000,'
                    '"choices":[{"message":{"tool_calls":[{"id":"call_123","type":"function"}]}}]}'
                )
            },
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"messages":[]}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.openai.com",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    loaded = yaml.safe_load(ReadableYamlSerializer.serialize(cassette))
    parsed_body = loaded["interactions"][0]["response"]["body"]["parsed_body"]

    assert parsed_body["id"] == "[RECORDED_RESPONSE_ID]"
    assert parsed_body["created"] == 0
    assert parsed_body["choices"][0]["message"]["tool_calls"][0]["id"] == "call_123"


def test_recorded_response_metadata_normalization_handles_top_level_arrays():
    response = before_record_response(
        {
            "headers": {"Content-Type": ["application/json"]},
            "body": {"string": '[{"id":"chatcmpl-123","created":1770000000,"choices":[]}]'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"messages":[]}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.openai.com",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    parsed_body = yaml.safe_load(ReadableYamlSerializer.serialize(cassette))["interactions"][0]["response"]["body"][
        "parsed_body"
    ]

    assert parsed_body == [{"id": "[RECORDED_RESPONSE_ID]", "created": 0, "choices": []}]


def test_recorded_jailbreak_score_normalization_allows_extra_fields():
    response = before_record_response(
        {
            "headers": {"Content-Type": ["application/json"]},
            "body": {"string": '{"jailbreak":true,"score":0.873,"model":"jailbreak-detect"}'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"messages":[]}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.nvidia.com",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    parsed_body = yaml.safe_load(ReadableYamlSerializer.serialize(cassette))["interactions"][0]["response"]["body"][
        "parsed_body"
    ]

    assert parsed_body == {"jailbreak": True, "score": 0.0, "model": "jailbreak-detect"}


def test_recorded_sse_jailbreak_score_normalization_allows_extra_fields():
    response = before_record_response(
        {
            "headers": {"Content-Type": ["text/event-stream"]},
            "body": {"string": 'data: {"jailbreak":true,"score":0.873,"model":"jailbreak-detect"}\n\n'},
        }
    )
    cassette = {
        "interactions": [
            {
                "request": {
                    "body": '{"stream":true}',
                    "headers": {},
                    "method": "POST",
                    "uri": "https://api.nvidia.com",
                },
                "response": response,
            }
        ],
        "version": 1,
    }

    parsed_body = yaml.safe_load(ReadableYamlSerializer.serialize(cassette))["interactions"][0]["response"]["body"][
        "parsed_body"
    ]

    assert parsed_body == [{"jailbreak": True, "score": 0.0, "model": "jailbreak-detect"}]


def test_recorded_request_sanitizer_strips_volatile_headers():
    request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body='{"messages":[{"role":"user","content":"hello"}]}',
        headers={"Content-Length": "100", "Content-Type": "application/json"},
    )

    sanitized = before_record_request(request)
    body = sanitized.body.decode("utf-8") if isinstance(sanitized.body, bytes) else sanitized.body

    assert "Content-Length" not in sanitized.headers
    assert "hello" in body


def test_recorded_request_sanitizer_redacts_non_object_json_body():
    request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body='[{"accessToken":"request-access-token-1234567890"}]',
        headers={"Content-Type": "application/json"},
    )

    sanitized = before_record_request(request)
    body = sanitized.body.decode("utf-8") if isinstance(sanitized.body, bytes) else sanitized.body

    assert json.loads(body) == [{"accessToken": "[REDACTED]"}]


def test_recorded_request_sanitizer_redacts_raw_non_json_body():
    request = Request(
        method="POST",
        uri="https://api.openai.com/v1/uploads",
        body="Bearer raw-token-1234567890",
        headers={"Content-Type": "text/plain"},
    )

    sanitized = before_record_request(request)
    body = sanitized.body.decode("utf-8") if isinstance(sanitized.body, bytes) else sanitized.body

    assert body == "Bearer [REDACTED]"


def test_recorded_response_sanitizer_redacts_raw_non_json_body():
    response = {
        "headers": {"Content-Type": ["text/plain"]},
        "body": {"string": "failed with sk-rawsecret1234567890"},
    }

    sanitized = before_record_response(response)

    assert sanitized["body"]["string"] == "failed with [OPENAI_API_KEY]"


def test_recorded_response_sanitizer_ignores_non_dict_body_container():
    response = {"headers": {"Content-Type": ["application/json"]}, "body": "not-a-vcr-body-dict"}

    assert before_record_response(response) == response


def test_recorded_vcr_config_matches_on_request_body():
    assert "recorded_body" in build_vcr_config()["match_on"]


def test_recorded_provider_key_lookup_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown recorded provider 'nvidia'; expected one of: nim, openai"):
        _provider_key_fixture_name("nvidia")


def test_recorded_refresh_uses_api_key_without_live_mode_gate(monkeypatch):
    monkeypatch.setenv("LIVE_TEST_MODE", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert api_key_for_record_mode("OPENAI_API_KEY", "dummy-key", "none") == "dummy-key"

    with pytest.raises(pytest.fail.Exception, match="OPENAI_API_KEY is required to refresh cassette"):
        api_key_for_record_mode("OPENAI_API_KEY", "dummy-key", "rewrite")

    monkeypatch.setenv("OPENAI_API_KEY", "real-key")
    assert api_key_for_record_mode("OPENAI_API_KEY", "dummy-key", "rewrite") == "real-key"


def test_recorded_refresh_fixture_returns_selected_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-key")

    assert set_api_key_for_record_mode(monkeypatch, "OPENAI_API_KEY", "dummy-key", "rewrite") == "real-key"
    assert set_api_key_for_record_mode(monkeypatch, "OPENAI_API_KEY", "dummy-key", "none") == "dummy-key"


def test_recorded_body_matcher_compares_sanitized_json_bodies():
    cassette_request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body={"model": "gpt-5.4-nano", "accessToken": "stored-token-1234567890"},
        headers={"Content-Type": "application/json"},
    )
    replay_request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body='{"model":"gpt-5.4-nano","accessToken":"live-token-1234567890"}',
        headers={"Content-Type": "application/json"},
    )
    stale_request = Request(
        method="POST",
        uri="https://api.openai.com/v1/chat/completions",
        body='{"model":"gpt-5.4-nano","messages":[{"role":"user","content":"changed"}]}',
        headers={"Content-Type": "application/json"},
    )

    recorded_body_matcher(cassette_request, replay_request)
    with pytest.raises(AssertionError, match="Recorded request JSON body"):
        recorded_body_matcher(cassette_request, stale_request)
