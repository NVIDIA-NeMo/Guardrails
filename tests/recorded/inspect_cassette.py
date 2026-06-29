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

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from tests.recorded.cassette import decode_body_json, decode_body_text, stream_payloads_from_body


def _request_payload(interaction: dict[str, Any]) -> dict[str, Any]:
    request = interaction.get("request", {})
    if isinstance(request.get("parsed_body"), dict):
        return request["parsed_body"]
    payload = decode_body_json(request.get("body"))
    return payload if isinstance(payload, dict) else {}


def _response_payload(interaction: dict[str, Any]) -> dict[str, Any]:
    body = interaction.get("response", {}).get("body")
    try:
        payload = decode_body_json(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _response_body_text(interaction: dict[str, Any]) -> str | None:
    body = interaction.get("response", {}).get("body")
    text = decode_body_text(body)
    return text or None


def cassette_summary(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = []
    interactions = data.get("interactions", []) if isinstance(data, dict) else []
    for index, interaction in enumerate(interactions):
        request = interaction.get("request", {})
        response = interaction.get("response", {})
        request_payload = _request_payload(interaction)
        response_payload = _response_payload(interaction)
        stream_payloads = stream_payloads_from_body(response.get("body"))
        rows.append(
            {
                "index": index,
                "method": request.get("method"),
                "uri": request.get("uri"),
                "status": response.get("status", {}).get("code"),
                "model": request_payload.get("model"),
                "stream": request_payload.get("stream", False),
                "response_model": response_payload.get("model") if response_payload else None,
                "raw_response": _response_body_text(interaction) if not response_payload else None,
                "stream_events": len(stream_payloads),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cassette", type=Path)
    args = parser.parse_args()
    print(json.dumps(cassette_summary(args.cassette), indent=2))


if __name__ == "__main__":
    main()
