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

"""Keep recording drift visible independently of permissive runtime tests."""

import json
from pathlib import Path

import pytest

from tests.recorded.server.experimental.openai.contract_audit import audit_exchange, recorded_endpoint
from tests.server.experimental.test_openai_responses_projections import response_payload
from tests.server.experimental.test_openai_responses_streaming_projections import event_bytes

RECORDING = Path("test_forward_compatibility/test_native_responses_forward_compatibility[buffered].yaml")
URI = "https://api.openai.com/v1/responses"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("location", ["request", "response", "event"])
async def test_strict_audit_rejects_unreviewed_fields_in_compatibility_recordings(streaming, location):
    request = {"input": "question", "store": False, "stream": streaming}
    if location == "request":
        request["future_option"] = True
    payload = response_payload()
    if location == "response" or (location == "event" and not streaming):
        payload["future_metadata"] = {"counter": 1}
    if streaming:
        additions = {"future_metadata": {"counter": 1}} if location == "event" else {}
        wire = event_bytes("response.output_text.delta", delta="answer", **additions) + event_bytes(
            "response.completed", response=payload
        )
    else:
        wire = json.dumps(payload).encode()
    with pytest.raises(AssertionError, match="unreviewed fields"):
        await audit_exchange(RECORDING, URI, json.dumps(request), {"status": {"code": 200}, "body": {"string": wire}})


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_audit_diagnostics_do_not_echo_provider_content(streaming):
    private_text = "private provider text"
    payload = response_payload(private_text)
    payload["future_metadata"] = {"counter": 1}
    wire = event_bytes("response.completed", response=payload) if streaming else json.dumps(payload).encode()
    with pytest.raises(AssertionError, match="unreviewed fields") as failure:
        await audit_exchange(
            RECORDING,
            URI,
            json.dumps({"input": "question", "store": False, "stream": streaming}),
            {"status": {"code": 200}, "body": {"string": wire}},
        )
    assert private_text not in str(failure.value)
    assert failure.value.__suppress_context__


@pytest.mark.asyncio
async def test_strict_audit_checks_snapshot_consistency():
    wire = event_bytes("response.output_text.delta", delta="answer") + event_bytes(
        "response.completed", response=response_payload("different")
    )
    with pytest.raises(ValueError, match="snapshot differs"):
        await audit_exchange(
            RECORDING,
            URI,
            json.dumps({"input": "question", "store": False, "stream": True}),
            {"status": {"code": 200}, "body": {"string": wire}},
        )


@pytest.mark.parametrize(
    "path,uri",
    [
        (Path("new_provider/capture.yaml"), URI),
        (RECORDING, "https://api.openai.com/v1/unknown"),
        (Path("test_chat_completions/capture.yaml"), URI),
    ],
)
def test_recording_audit_requires_an_explicit_binding(path, uri):
    with pytest.raises(ValueError, match="No reviewed proxy binding"):
        recorded_endpoint(path, uri)
