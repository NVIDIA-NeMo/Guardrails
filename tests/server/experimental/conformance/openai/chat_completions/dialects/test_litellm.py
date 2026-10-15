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

"""Test reviewed LiteLLM extension boundaries through the real Chat transport.

Cases follow the source inventory in providers/openai/dialects/litellm/README.md,
not generated guard schemas. They are synthetic examples, not recordings or a
claim that the OpenAI source fixture validates LiteLLM's extensions.
"""

import pytest

from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from tests.server.experimental.conformance.openai.chat_completions.http_support import (
    assert_output_error,
    checker_for,
    proxy,
)
from tests.server.experimental.conformance.openai.chat_completions.support import (
    chat_chunk,
    chat_request,
    chat_response,
    json_bytes,
    sse_bytes,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.block_network]
TEXT = "Cafe\u0301 — “quoted” ☃"
MESSAGE = ("choices", 0, "message")
CHOICE = ("choices", 0)
DELTA = ("choices", 0, "delta")
EXTENSION_LOCATIONS = [
    pytest.param(False, MESSAGE + ("provider_specific_fields",), id="message-bag"),
    pytest.param(False, CHOICE + ("provider_specific_fields",), id="choice-bag"),
    pytest.param(True, ("provider_specific_fields",), id="stream-root-bag"),
    pytest.param(True, DELTA + ("provider_specific_fields",), id="delta-bag"),
]
ARRAY_LOCATIONS = [
    pytest.param(streaming, location + (field,), id=f"{name}-{field}")
    for streaming, location, name, fields in [
        (False, MESSAGE, "message", ("images", "thinking_blocks", "reasoning_items")),
        (True, DELTA, "delta", ("images", "thinking_blocks", "reasoning_items", "annotations")),
    ]
    for field in fields
]


def extension_body(streaming, path, value):
    """Place one independently chosen extension beside the guarded text."""
    payload = chat_chunk(TEXT) if streaming else chat_response()
    if not streaming:
        payload["choices"][0]["message"]["content"] = TEXT
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    return sse_bytes(payload) if streaming else json_bytes(payload)


async def assert_extension_exchange(streaming, body, inspect, endpoint, accepted):
    checker = checker_for(inspect)
    request_body = json_bytes(chat_request(stream=streaming))
    async with proxy(checker, body, streaming=streaming, endpoint=endpoint) as (client, requests, source):
        response = await client.post(
            "/v1/chat/completions", content=request_body, headers={"content-type": "application/json"}
        )
    assert len(requests) == 1
    assert requests[0].content == request_body
    assert source.closed
    if not inspect or accepted:
        assert response.status_code == 200
        assert response.content == body
        assert checker.text_calls == ([("input", "question"), ("output", TEXT)] if inspect else [])
    else:
        assert_output_error(response, streaming)
        assert checker.text_calls == [("input", "question")]


@pytest.mark.parametrize(
    "endpoint", [CHAT_COMPLETIONS_ENDPOINT, LITELLM_CHAT_COMPLETIONS_ENDPOINT], ids=["openai", "litellm"]
)
@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize("streaming,path", EXTENSION_LOCATIONS)
@pytest.mark.parametrize("value", [None, {}], ids=["null", "empty"])
async def test_empty_extension_bags_require_explicit_dialect_selection(endpoint, inspect, streaming, path, value):
    await assert_extension_exchange(
        streaming,
        extension_body(streaming, path, value),
        inspect,
        endpoint,
        endpoint is LITELLM_CHAT_COMPLETIONS_ENDPOINT,
    )


@pytest.mark.parametrize(
    "endpoint", [CHAT_COMPLETIONS_ENDPOINT, LITELLM_CHAT_COMPLETIONS_ENDPOINT], ids=["openai", "litellm"]
)
@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize("streaming,path", ARRAY_LOCATIONS)
@pytest.mark.parametrize("value", [None, []], ids=["null", "empty"])
async def test_empty_content_extensions_do_not_add_inspection_targets(endpoint, inspect, streaming, path, value):
    await assert_extension_exchange(
        streaming,
        extension_body(streaming, path, value),
        inspect,
        endpoint,
        endpoint is LITELLM_CHAT_COMPLETIONS_ENDPOINT,
    )


@pytest.mark.parametrize(
    "endpoint", [CHAT_COMPLETIONS_ENDPOINT, LITELLM_CHAT_COMPLETIONS_ENDPOINT], ids=["openai", "litellm"]
)
@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize(
    "path,value",
    [
        pytest.param(
            MESSAGE + ("provider_specific_fields",), {"refusal": None, "thinking_blocks": []}, id="reviewed-message"
        ),
        pytest.param(
            CHOICE + ("provider_specific_fields",), {"native_finish_reason": "not guarded text"}, id="finish-metadata"
        ),
    ],
)
async def test_reviewed_buffered_members_preserve_bytes_without_changing_checked_text(endpoint, inspect, path, value):
    await assert_extension_exchange(
        False, extension_body(False, path, value), inspect, endpoint, endpoint is LITELLM_CHAT_COMPLETIONS_ENDPOINT
    )


@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize("streaming,path", ARRAY_LOCATIONS)
async def test_nonempty_content_extensions_are_withheld_when_inspected(inspect, streaming, path):
    await assert_extension_exchange(
        streaming,
        extension_body(streaming, path, ["unreviewed text"]),
        inspect,
        LITELLM_CHAT_COMPLETIONS_ENDPOINT,
        False,
    )


@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize("streaming,path", EXTENSION_LOCATIONS)
async def test_unreviewed_extension_members_fail_closed_at_every_container(inspect, streaming, path):
    await assert_extension_exchange(
        streaming,
        extension_body(streaming, path, {"future_content": "unreviewed text"}),
        inspect,
        LITELLM_CHAT_COMPLETIONS_ENDPOINT,
        False,
    )


@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize(
    "path,value",
    [
        pytest.param(DELTA + ("provider_specific_fields",), {"refusal": None}, id="buffered-refusal"),
        pytest.param(("provider_specific_fields",), {"native_finish_reason": "stop"}, id="buffered-finish-metadata"),
    ],
)
async def test_buffered_extension_permissions_do_not_leak_into_streams(inspect, path, value):
    await assert_extension_exchange(
        True, extension_body(True, path, value), inspect, LITELLM_CHAT_COMPLETIONS_ENDPOINT, False
    )


@pytest.mark.parametrize("inspect", [False, True], ids=["relay", "inspected"])
@pytest.mark.parametrize("extensions", [{}, {"future_content": "unreviewed text"}], ids=["reviewed", "unreviewed"])
async def test_usage_only_events_keep_metadata_role_without_bypassing_extension_validation(inspect, extensions):
    payload = chat_chunk(
        choices=[],
        usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        provider_specific_fields=extensions,
    )
    body = sse_bytes(payload)
    checker = checker_for(inspect)
    async with proxy(checker, body, streaming=True, endpoint=LITELLM_CHAT_COMPLETIONS_ENDPOINT) as (
        client,
        requests,
        source,
    ):
        response = await client.post("/v1/chat/completions", json=chat_request(stream=True))
    assert len(requests) == 1
    assert source.closed
    assert checker.text_calls == ([("input", "question")] if inspect else [])
    if inspect and extensions:
        assert_output_error(response, True)
    else:
        assert response.status_code == 200
        assert response.content == body
