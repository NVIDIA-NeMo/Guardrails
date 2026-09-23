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

import pytest

from tests.recorded.cassette import cassette_request_jsons, recorded_chat_response, stream_payloads_from_body
from tests.recorded.inspect_cassette import cassette_summary
from tests.recorded.normalization import normalize_stream_chunks

pytestmark = [pytest.mark.recorded]


def test_cassette_summary_reads_parsed_bodies(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text(
        """
version: 1
interactions:
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
      stream: true
  response:
    status:
      code: 200
      message: OK
    headers:
      Content-Type:
      - text/event-stream
    body:
      parsed_body:
      - id: '[RECORDED_RESPONSE_ID]'
        choices: []
      - '[DONE]'
""",
        encoding="utf-8",
    )

    assert cassette_summary(cassette) == [
        {
            "index": 0,
            "method": "POST",
            "uri": "https://api.openai.com/v1/chat/completions",
            "status": 200,
            "model": "gpt-5.4-nano",
            "stream": True,
            "response_model": None,
            "raw_response": None,
            "stream_events": 1,
        }
    ]


def test_cassette_summary_reads_raw_error_bodies(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text(
        """
version: 1
interactions:
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
  response:
    status:
      code: 503
      message: Service Unavailable
    headers:
      Content-Type:
      - text/plain
    body:
      string: upstream connect error
""",
        encoding="utf-8",
    )

    assert cassette_summary(cassette) == [
        {
            "index": 0,
            "method": "POST",
            "uri": "https://api.openai.com/v1/chat/completions",
            "status": 503,
            "model": "gpt-5.4-nano",
            "stream": False,
            "response_model": None,
            "raw_response": "upstream connect error",
            "stream_events": 0,
        }
    ]


def test_cassette_summary_handles_empty_files(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text("# empty\n", encoding="utf-8")

    assert cassette_summary(cassette) == []


def test_recorded_chat_response_normalizes_zero_and_nullable_usage(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text(
        """
version: 1
interactions:
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
  response:
    body:
      parsed_body:
        id: chatcmpl-zero
        choices:
        - message:
            content: ""
          finish_reason: stop
        usage:
          prompt_tokens: 0
          completion_tokens: 0
          total_tokens: 0
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano-null
  response:
    body:
      parsed_body:
        id: chatcmpl-null
        choices:
        - message:
            content: ""
          finish_reason: stop
        usage:
          prompt_tokens:
          completion_tokens:
          total_tokens:
""",
        encoding="utf-8",
    )

    zero_usage = recorded_chat_response(cassette, request_model="gpt-5.4-nano").usage
    nullable_usage = recorded_chat_response(cassette, request_model="gpt-5.4-nano-null").usage

    assert zero_usage["total_tokens"] == 0
    assert nullable_usage == {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def test_recorded_chat_response_skips_non_dict_response_payloads(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text(
        """
version: 1
interactions:
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
  response:
    body:
      parsed_body:
      - not-a-chat-response
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
  response:
    body:
      string: not-json
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
  response:
    body:
      parsed_body:
        id: chatcmpl-valid
        choices:
        - message:
            content: valid
          finish_reason: stop
""",
        encoding="utf-8",
    )

    assert recorded_chat_response(cassette, request_model="gpt-5.4-nano").content == "valid"


def test_cassette_request_jsons_returns_copy_of_cached_interactions(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text(
        """
version: 1
interactions:
- request:
    method: POST
    uri: https://api.openai.com/v1/chat/completions
    parsed_body:
      model: gpt-5.4-nano
  response:
    body:
      parsed_body:
        choices: []
""",
        encoding="utf-8",
    )

    cassette_request_jsons(cassette)[0]["model"] = "mutated"

    assert cassette_request_jsons(cassette)[0]["model"] == "gpt-5.4-nano"


def test_cassette_request_jsons_handles_null_interactions(tmp_path):
    cassette = tmp_path / "example.yaml"
    cassette.write_text(
        """
version: 1
interactions:
""",
        encoding="utf-8",
    )

    assert cassette_request_jsons(cassette) == []


def test_stream_payloads_from_body_skips_malformed_raw_sse_lines():
    body = {"string": 'data: not-json\n\ndata: {"choices":[]}\n\ndata: [DONE]\n\n'}

    assert stream_payloads_from_body(body) == [{"choices": []}]


def test_normalize_stream_chunks_ignores_non_string_content_fallback():
    result = normalize_stream_chunks([{"content": {"not": "text"}}, {"content": "ok"}])

    assert result["content"] == "ok"
