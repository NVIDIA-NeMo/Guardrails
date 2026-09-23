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

import pytest

from nemoguardrails.http.composition import ensure_retrying_http_client
from nemoguardrails.http.instrumented import InstrumentedHTTPClient
from nemoguardrails.http.retry import RetryingHTTPClient, RetryPolicy
from nemoguardrails.http.types import HTTPResponse
from nemoguardrails.testing.http import RecordingHTTPClient


def test_retry_composition_wraps_transport():
    transport = RecordingHTTPClient()

    client = ensure_retrying_http_client(transport, RetryPolicy())

    assert isinstance(client, RetryingHTTPClient)
    assert client.wrapped_client is transport


def test_retry_composition_places_retry_inside_instrumentation():
    transport = RecordingHTTPClient()
    instrumented = InstrumentedHTTPClient(transport, None)

    client = ensure_retrying_http_client(instrumented, RetryPolicy())

    assert isinstance(client, InstrumentedHTTPClient)
    assert isinstance(client.wrapped_client, RetryingHTTPClient)
    assert client.wrapped_client.wrapped_client is transport


def test_retry_composition_preserves_canonical_client():
    retrying = RetryingHTTPClient(RecordingHTTPClient())
    instrumented = InstrumentedHTTPClient(retrying, None)

    client = ensure_retrying_http_client(instrumented, RetryPolicy())

    assert client is instrumented


@pytest.mark.asyncio
async def test_retry_composition_normalizes_reverse_order_and_preserves_policy():
    transport = RecordingHTTPClient(
        [
            HTTPResponse(status_code=503),
            HTTPResponse(status_code=503),
            HTTPResponse(status_code=200),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    retrying = RetryingHTTPClient(
        InstrumentedHTTPClient(transport, None),
        RetryPolicy(max_attempts=2, initial_delay=0.25, max_delay=0.25),
        sleep=sleep,
        random_value=lambda: 1.0,
    )

    client = ensure_retrying_http_client(
        retrying,
        RetryPolicy(max_attempts=3, initial_delay=1.0, max_delay=1.0),
    )
    response = await client.request("GET", "https://example.com/check")

    assert isinstance(client, InstrumentedHTTPClient)
    assert isinstance(client.wrapped_client, RetryingHTTPClient)
    assert client.wrapped_client.wrapped_client is transport
    assert response.status_code == 503
    assert len(transport.requests) == 2
    assert delays == [0.25]


def test_retry_composition_rejects_multiple_retry_layers():
    transport = RecordingHTTPClient()
    client = RetryingHTTPClient(RetryingHTTPClient(transport))

    with pytest.raises(ValueError, match="multiple retry layers"):
        ensure_retrying_http_client(client, RetryPolicy())
