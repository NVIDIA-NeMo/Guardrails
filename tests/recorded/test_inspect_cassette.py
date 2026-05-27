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

from tests.recorded.inspect_cassette import cassette_summary


@pytest.mark.recorded
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


@pytest.mark.recorded
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
