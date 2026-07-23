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

from datetime import datetime, timezone

import pytest

from nemoguardrails.http.errors import HTTPConnectionError
from nemoguardrails.http.retry import RetryingHTTPClient, RetryPolicy
from nemoguardrails.http.testing import RecordingHTTPClient
from nemoguardrails.http.types import HTTPResponse


@pytest.mark.asyncio
async def test_retry_client_retries_status_and_preserves_request():
    transport = RecordingHTTPClient(
        [
            HTTPResponse(status_code=503),
            HTTPResponse(status_code=200, extensions={"request_id": "abc"}),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    client = RetryingHTTPClient(
        transport,
        RetryPolicy(retryable_methods=frozenset({"POST"})),
        sleep=sleep,
        random_value=lambda: 0.5,
    )

    response = await client.request(
        "POST",
        "https://example.com/check",
        headers={"x-key": "value"},
        json={"text": "hello"},
    )

    assert len(transport.requests) == 2
    assert transport.requests[0] == transport.requests[1]
    assert response.extensions == {"request_id": "abc", "retry_count": 1}
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_retry_client_honors_retry_after_case_insensitively():
    transport = RecordingHTTPClient(
        [
            HTTPResponse(status_code=429, headers={"Retry-After": "3"}),
            HTTPResponse(status_code=200),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    client = RetryingHTTPClient(transport, sleep=sleep)

    await client.request("GET", "https://example.com")

    assert delays == [3.0]


def test_retry_policy_parses_retry_after_date():
    policy = RetryPolicy()
    response = HTTPResponse(
        status_code=429,
        headers={"retry-after": "Thu, 01 Jan 2026 00:00:03 GMT"},
    )

    delay = policy.retry_after(response, now=datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert delay == 3.0


@pytest.mark.asyncio
async def test_retry_client_respects_retry_override():
    transport = RecordingHTTPClient([HTTPResponse(status_code=503, headers={"X-Should-Retry": "false"})])
    client = RetryingHTTPClient(
        transport,
        RetryPolicy(honor_retry_override_header=True),
    )

    response = await client.request("GET", "https://example.com")

    assert response.status_code == 503
    assert response.extensions["retry_count"] == 0
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_retry_client_ignores_nonstandard_override_by_default():
    transport = RecordingHTTPClient([HTTPResponse(status_code=200, headers={"X-Should-Retry": "true"})])
    client = RetryingHTTPClient(transport)

    response = await client.request("POST", "https://example.com")

    assert response.status_code == 200
    assert response.extensions["retry_count"] == 0
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_retry_client_does_not_retry_post_by_default():
    transport = RecordingHTTPClient(
        [
            HTTPResponse(status_code=503),
            HTTPResponse(status_code=200),
        ]
    )
    client = RetryingHTTPClient(transport)

    response = await client.request("POST", "https://example.com/jobs")

    assert response.status_code == 503
    assert response.extensions["retry_count"] == 0
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_retry_client_raises_transport_error_after_max_attempts():
    transport = RecordingHTTPClient(
        [
            HTTPConnectionError("unavailable"),
            HTTPConnectionError("unavailable"),
            HTTPConnectionError("unavailable"),
        ]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    client = RetryingHTTPClient(transport, sleep=sleep, random_value=lambda: 1.0)

    with pytest.raises(HTTPConnectionError) as exc_info:
        await client.request("GET", "https://example.com")

    assert exc_info.value.retry_count == 2
    assert delays == [0.5, 1.0]
    assert len(transport.requests) == 3


@pytest.mark.asyncio
async def test_transport_error_is_not_retried_when_disabled():
    transport = RecordingHTTPClient([HTTPConnectionError("offline")])
    client = RetryingHTTPClient(
        transport,
        RetryPolicy(
            max_attempts=3,
            retryable_methods=frozenset({"POST"}),
            retry_transport_errors=False,
        ),
    )

    with pytest.raises(HTTPConnectionError) as exc_info:
        await client.request("POST", "https://example.com/jobs")

    assert exc_info.value.retry_count == 0
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_retry_client_limits_retryable_methods():
    transport = RecordingHTTPClient([HTTPResponse(status_code=503), HTTPResponse(status_code=200)])
    client = RetryingHTTPClient(
        transport,
        RetryPolicy(retryable_methods=frozenset({"post"})),
        sleep=lambda delay: _completed_sleep(delay),
    )

    response = await client.request("GET", "https://example.com")

    assert response.status_code == 503
    assert response.extensions["retry_count"] == 0
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_retry_client_returns_final_retryable_status():
    transport = RecordingHTTPClient([HTTPResponse(status_code=503), HTTPResponse(status_code=503)])
    client = RetryingHTTPClient(
        transport,
        policy=RetryPolicy(max_attempts=2),
        sleep=lambda delay: _completed_sleep(delay),
    )

    response = await client.request("GET", "https://example.com")

    assert response.status_code == 503
    assert response.extensions["retry_count"] == 1


async def _completed_sleep(delay: float) -> None:
    return None


@pytest.mark.asyncio
async def test_retry_client_closes_wrapped_managed_client_once():
    transport = RecordingHTTPClient()
    client = RetryingHTTPClient(transport)

    await client.close()
    await client.close()

    assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_retry_client_context_manager_closes_wrapped_client():
    transport = RecordingHTTPClient()

    async with RetryingHTTPClient(transport) as client:
        assert isinstance(client, RetryingHTTPClient)

    assert transport.close_calls == 1


@pytest.mark.parametrize(
    "policy",
    [
        RetryPolicy(max_attempts=1),
        RetryPolicy(initial_delay=0, max_delay=0),
        RetryPolicy(max_retry_after=0),
    ],
)
def test_retry_policy_accepts_boundary_values(policy):
    assert isinstance(policy, RetryPolicy)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-1", 0.0),
        ("0", 0.0),
        ("3", 3.0),
        ("999", 5.0),
        ("Wed, 01 Jan 2020 00:00:00 GMT", 0.0),
    ],
)
def test_retry_policy_can_clamp_retry_after(value, expected):
    policy = RetryPolicy(max_retry_after=5, clamp_retry_after=True)
    response = HTTPResponse(status_code=429, headers={"Retry-After": value})

    assert policy.retry_after(response, now=datetime(2026, 1, 1, tzinfo=timezone.utc)) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_delay": -1},
        {"initial_delay": 2, "max_delay": 1},
        {"max_retry_after": -1},
    ],
)
def test_retry_policy_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)
