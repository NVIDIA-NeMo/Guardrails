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

"""Exercise forward-compatible Responses stream metadata."""

import pytest

from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_integration import checker, post_stream
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT], ids=["native", "litellm"])
@pytest.mark.parametrize("unknown_fields", ["allow", "forbid"])
@pytest.mark.parametrize("kind", ["reasoning", "error"])
@pytest.mark.parametrize(
    "metadata", [None, {"id": "metadata-only"}, {"output": [{"type": "message", "id": "unrelated"}]}]
)
async def test_response_member_on_another_event_follows_unknown_field_policy(endpoint, unknown_fields, kind, metadata):
    if kind == "reasoning":
        extra = event_bytes("response.output_item.done", item={"type": "reasoning", "summary": []}, response=metadata)
        terminal = event_bytes("response.completed", response=response_payload())
    else:
        extra = event_bytes("error", message="provider diagnostic", code="provider_error", response=metadata)
        terminal = b""
    wire = (
        event_bytes("response.output_text.delta", delta="answer", item_id="msg_test", output_index=1) + extra + terminal
    )
    inspector = checker()
    response, source, dispatches = await post_stream(
        [wire], inspector, endpoint=endpoint, unknown_fields=unknown_fields
    )
    assert response.status_code == 200
    assert len(dispatches) == 1 and source.closed
    if unknown_fields == "allow":
        assert response.content == wire
        assert [(kind, check.output_content) for kind, check in inspector.calls if kind == "output"] == [
            ("output", "answer")
        ]
    else:
        assert b"unsupported_responses_response_shape" in response.content
        assert b"answer" not in response.content
        assert [kind for kind, _ in inspector.calls] == ["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, LITELLM_RESPONSES_ENDPOINT], ids=["native", "litellm"])
async def test_permissive_metadata_does_not_relax_lifecycle_identity_checks(endpoint):
    payload = response_payload()
    payload["output"][1]["id"] = "different-item"
    payload["future_metadata"] = {"counter": 1}
    wire = event_bytes("response.output_text.delta", delta="answer", item_id="msg_test", output_index=1) + event_bytes(
        "response.completed", response=payload
    )
    response, source, _ = await post_stream([wire], checker(), endpoint=endpoint, unknown_fields="allow")
    assert b"unsupported_responses_response_shape" in response.content
    assert b"answer" not in response.content
    assert source.closed
