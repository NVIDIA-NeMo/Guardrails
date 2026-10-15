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

import pytest

from nemoguardrails import LLMRails
from tests.recorded.assertions import assert_stream_contract
from tests.recorded.conftest import before_record_response
from tests.recorded.rails.public_api.configs import INPUT_RAILS_CONFIG
from tests.recorded.rails_config import load_config

pytestmark = [pytest.mark.recorded]


def test_load_config_returns_fresh_copy_when_llmrails_mutates_config():
    config = load_config(INPUT_RAILS_CONFIG)
    original_flow_count = len(config.flows)

    LLMRails(config, verbose=False)

    assert len(config.flows) > original_flow_count
    assert len(load_config(INPUT_RAILS_CONFIG).flows) == original_flow_count


def test_assert_stream_contract_accepts_metadata_text_chunks():
    assert assert_stream_contract([{"text": "Hello", "metadata": {"usage": {}}}], expect_multiple=False) == "Hello"


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_shared_sanitizer_preserves_responses_content_and_nested_identity(stream):
    """Content is evidence; only shared credential/metadata filtering applies."""
    payload = {
        "id": "response-id",
        "output": [
            {"id": "reasoning-id", "type": "reasoning", "summary": [], "encrypted_content": "opaque-state"},
            {"id": "message-id", "type": "message", "content": [{"type": "output_text", "text": "Hello."}]},
        ],
    }
    if stream:
        data = {"type": "response.completed", "response": payload, "sequence_number": 4}
        body = f"event: response.completed\ndata: {json.dumps(data)}\n\n"
        content_type = "text/event-stream"
    else:
        body = json.dumps(payload)
        content_type = "application/json"
    response = before_record_response(
        {"headers": {"content-type": [content_type], "authorization": ["Bearer test-secret"]}, "body": {"string": body}}
    )
    assert "authorization" not in response["headers"]
    if stream:
        encoded = next(line[6:] for line in response["body"]["string"].splitlines() if line.startswith("data: "))
        event = json.loads(encoded)
        assert event["sequence_number"] == 4
        assert event["response"] == payload
    else:
        actual = json.loads(response["body"]["string"])
        assert actual["output"] == payload["output"]
        assert actual["id"] == "[RECORDED_RESPONSE_ID]"
