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

"""Gate: committed cassettes must be a fixed point of the recording sanitizers.

Every genuine cassette passed through before_record_request/response at record
time, so re-applying the sanitizers must change nothing (comparing bodies
JSON- and SSE-aware, since the encoders differ in formatting only). A
hand-authored cassette containing realistic volatile values (ids, timestamps,
cookies, provider headers) fails this check because the sanitizers would have
scrubbed them. This raises the bar against fabricated recordings; it cannot
prove provenance (a sentinel-perfect forgery passes), so unverifiable
cassettes from untrusted sources still need re-recording or an explicit
fake_cassette marking.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from tests.recorded.cassette import cassette_with_rehydrated_bodies
from tests.recorded.conftest import before_record_request, before_record_response

RECORDED_ROOT = Path(__file__).parent


def _canonical_body(value):
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    if "data:" in value:
        events = []
        for line in value.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                try:
                    events.append(json.loads(payload))
                except (json.JSONDecodeError, ValueError):
                    events.append(payload)
        if events:
            return events
    return value


def _idempotence_violations(cassette: dict, label: str) -> list:
    violations = []
    wire = cassette_with_rehydrated_bodies(deepcopy(cassette))
    for index, interaction in enumerate(wire.get("interactions") or []):
        request = interaction.get("request", {})
        sanitized_request = SimpleNamespace(headers=deepcopy(request.get("headers", {})), body=request.get("body"))
        before_record_request(sanitized_request)
        if sanitized_request.headers != request.get("headers", {}):
            violations.append(f"{label} interaction {index}: request headers change under re-sanitization")
        if _canonical_body(sanitized_request.body) != _canonical_body(request.get("body")):
            violations.append(f"{label} interaction {index}: request body changes under re-sanitization")

        response = interaction.get("response", {})
        sanitized_response = before_record_response(deepcopy(response))
        if sanitized_response.get("headers", {}) != response.get("headers", {}):
            violations.append(f"{label} interaction {index}: response headers change under re-sanitization")
        original_body = response.get("body", {}).get("string") if isinstance(response.get("body"), dict) else None
        sanitized_body = (
            sanitized_response.get("body", {}).get("string")
            if isinstance(sanitized_response.get("body"), dict)
            else None
        )
        if _canonical_body(sanitized_body) != _canonical_body(original_body):
            violations.append(f"{label} interaction {index}: response body changes under re-sanitization")
    return violations


def test_committed_cassettes_are_sanitizer_fixed_points():
    violations = []
    count = 0
    for path in sorted(RECORDED_ROOT.rglob("cassettes/**/*.yaml")):
        if "fake" in path.parts:
            continue
        count += 1
        cassette = yaml.safe_load(path.read_text(encoding="utf-8"))
        violations.extend(_idempotence_violations(cassette, str(path.relative_to(RECORDED_ROOT))))

    assert count > 0
    assert not violations, (
        "Cassettes that are not fixed points of the recording sanitizers (were these recorded through "
        "pytest-recording, or authored by hand?):\n" + "\n".join(violations)
    )


def test_fixed_point_check_detects_unsanitized_cassettes():
    fabricated = {
        "interactions": [
            {
                "request": {
                    "method": "POST",
                    "uri": "https://api.example.test/v1/scans",
                    "headers": {"Authorization": ["Bearer sk-real-looking-key"], "Content-Type": ["application/json"]},
                    "body": json.dumps({"input": "hello"}),
                },
                "response": {
                    "status": {"code": 200, "message": "OK"},
                    "headers": {"Set-Cookie": ["session=abc123"], "content-type": ["application/json"]},
                    "body": {
                        "string": json.dumps(
                            {"id": "chatcmpl-9xYz12AbCdEf", "created": 1720512345, "result": {"outcome": "cleared"}}
                        )
                    },
                },
            }
        ]
    }

    assert _idempotence_violations(fabricated, "fabricated")
