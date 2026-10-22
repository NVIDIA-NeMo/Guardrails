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

import asyncio
import base64
import gc
import json
import logging

import pytest

from nemoguardrails.http import HTTPConnectionError, HTTPResponse
from nemoguardrails.library.hiddenlayer import _token
from nemoguardrails.library.hiddenlayer._token import (
    HiddenLayerTransportError,
    _access_token,
    _reset_token_cache,
)
from nemoguardrails.library.hiddenlayer.actions import _with_connection_retry
from nemoguardrails.testing import RecordingHTTPClient

API_URL = "https://api.us.hiddenlayer.ai"
TOKEN_URL = f"{API_URL}/oauth2/token"
CLIENT_ID = "client-id-value"
CLIENT_SECRET = "client-secret-value"


def _token_response(token: str = "token-1", expires_in=3600, status: int = 200) -> HTTPResponse:
    body = {"access_token": token, "token_type": "Bearer"}
    if expires_in is not None:
        body["expires_in"] = expires_in
    return HTTPResponse(
        status_code=status, headers={"Content-Type": "application/json"}, content=json.dumps(body).encode()
    )


class _SlowTokenClient(RecordingHTTPClient):
    """Hands out token-<n> after a short await, so concurrent callers overlap on the fetch."""

    async def request(self, method, url, **kwargs):
        await asyncio.sleep(0.01)
        self.add_response(_token_response(f"token-{len(self.requests) + 1}"))
        return await super().request(method, url, **kwargs)


class _FailingTokenClient(RecordingHTTPClient):
    """Answers every token request with the given failure after a short await."""

    def __init__(self, failure):
        super().__init__()
        self._failure = failure

    async def request(self, method, url, **kwargs):
        await asyncio.sleep(0.01)
        self.add_response(self._failure)
        return await super().request(method, url, **kwargs)


class _GatedTokenClient(RecordingHTTPClient):
    """Holds each token request until the test releases it, then answers token-<n>."""

    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def request(self, method, url, **kwargs):
        self.add_response(_token_response(f"token-{len(self.requests) + 1}"))
        response = await super().request(method, url, **kwargs)
        await self.release.wait()
        return response


class _Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value


@pytest.fixture(autouse=True)
def reset_cache():
    _reset_token_cache()
    yield
    _reset_token_cache()


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(_token, "_now", clock)
    return clock


def _get(client, secret: str = CLIENT_SECRET, **kwargs) -> str:
    return asyncio.run(_access_token(client, API_URL, CLIENT_ID, secret, **kwargs))


def test_request_uses_basic_auth_and_form_body():
    client = RecordingHTTPClient([_token_response()])

    assert _get(client) == "token-1"

    (request,) = client.requests
    assert (request.method, request.url) == ("POST", TOKEN_URL)
    assert request.headers == {
        "Authorization": "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    assert request.content == "grant_type=client_credentials"
    assert request.json is None
    assert request.timeout == 30.0


def test_one_fetch_for_two_calls():
    client = RecordingHTTPClient([_token_response()])
    assert _get(client) == _get(client) == "token-1"
    assert len(client.requests) == 1


def test_concurrent_calls_on_one_loop_fetch_once():
    client = _SlowTokenClient()

    async def both():
        return await asyncio.gather(*(_access_token(client, API_URL, CLIENT_ID, CLIENT_SECRET) for _ in range(5)))

    assert asyncio.run(both()) == ["token-1"] * 5
    assert len(client.requests) == 1


class TestConcurrentFailures:
    def test_concurrent_calls_share_one_failed_fetch(self):
        client = _FailingTokenClient(HTTPResponse(status_code=503))

        async def many():
            return await asyncio.gather(
                *(_access_token(client, API_URL, CLIENT_ID, CLIENT_SECRET) for _ in range(8)), return_exceptions=True
            )

        results = asyncio.run(many())

        assert all(isinstance(result, HiddenLayerTransportError) for result in results)
        assert len(client.requests) == 1

    def test_connection_retry_makes_two_requests_for_concurrent_calls(self):
        client = _FailingTokenClient(HTTPConnectionError("connection refused"))

        async def check():
            return await _with_connection_retry(lambda: _access_token(client, API_URL, CLIENT_ID, CLIENT_SECRET))

        async def many():
            return await asyncio.gather(*(check() for _ in range(8)), return_exceptions=True)

        results = asyncio.run(many())

        assert all(isinstance(result, HTTPConnectionError) for result in results)
        assert len(client.requests) == 2

    def test_a_stalled_fetch_does_not_hold_up_other_credentials(self):
        stalled = _GatedTokenClient()
        healthy = RecordingHTTPClient([_token_response("other-token")])

        async def both():
            blocked = asyncio.ensure_future(_access_token(stalled, API_URL, CLIENT_ID, CLIENT_SECRET))
            await asyncio.sleep(0)
            other = await asyncio.wait_for(_access_token(healthy, "https://other.example", CLIENT_ID, CLIENT_SECRET), 1)
            stalled.release.set()
            return other, await blocked

        assert asyncio.run(both()) == ("other-token", "token-1")

    def test_a_cancelled_caller_does_not_cancel_the_shared_fetch(self):
        client = _GatedTokenClient()

        async def run():
            first = asyncio.ensure_future(_access_token(client, API_URL, CLIENT_ID, CLIENT_SECRET))
            second = asyncio.ensure_future(_access_token(client, API_URL, CLIENT_ID, CLIENT_SECRET))
            await asyncio.sleep(0.01)
            first.cancel()
            client.release.set()
            return first, await second

        first, token = asyncio.run(run())

        assert first.cancelled()
        assert token == "token-1"
        assert len(client.requests) == 1


class TestExpiry:
    def test_reused_until_ten_seconds_before_expiry(self, clock):
        client = RecordingHTTPClient([_token_response("token-1", expires_in=100), _token_response("token-2")])
        assert _get(client) == "token-1"

        clock.value += 89.9
        assert _get(client) == "token-1"

        clock.value += 0.1
        assert _get(client) == "token-2"
        assert len(client.requests) == 2

    def test_missing_expires_in_defaults_to_an_hour(self, clock):
        client = RecordingHTTPClient([_token_response("token-1", expires_in=None), _token_response("token-2")])
        assert _get(client) == "token-1"

        clock.value += 3589.9
        assert _get(client) == "token-1"
        clock.value += 0.1
        assert _get(client) == "token-2"


def test_secret_change_fetches_a_new_token():
    client = RecordingHTTPClient([_token_response("token-1"), _token_response("token-2")])
    assert _get(client) == "token-1"
    assert _get(client, secret="rotated-secret") == "token-2"
    assert len(client.requests) == 2


class TestRejected:
    def test_drops_the_matching_token(self):
        client = RecordingHTTPClient([_token_response("token-1"), _token_response("token-2")])
        assert _get(client) == "token-1"
        assert _get(client, rejected="token-1") == "token-2"
        assert len(client.requests) == 2

    def test_keeps_a_token_that_already_replaced_the_rejected_one(self):
        client = RecordingHTTPClient([_token_response("token-2")])
        assert _get(client) == "token-2"
        assert _get(client, rejected="token-1") == "token-2"
        assert len(client.requests) == 1


class TestFailures:
    @pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
    def test_non_200_raises_safe_transport_error(self, status):
        body = b'{"error": "invalid_client", "detail": "client-secret-value is wrong"}'
        response = HTTPResponse(status_code=status, headers={"Content-Type": "application/json"}, content=body)
        client = RecordingHTTPClient([response])

        with pytest.raises(HiddenLayerTransportError) as exc_info:
            _get(client)

        assert exc_info.value.status_code == status
        assert str(exc_info.value) == (
            f"HiddenLayer token request failed: status={status} content_type=application/json body_length={len(body)}"
        )

    @pytest.mark.parametrize(
        "content",
        [b"<html>not json</html>", b"[]", b'{"token_type": "Bearer"}', b'{"access_token": ""}', b'{"access_token": 7}'],
    )
    def test_unusable_body_raises_transport_error(self, content):
        client = RecordingHTTPClient([HTTPResponse(status_code=200, content=content)])
        with pytest.raises(HiddenLayerTransportError) as exc_info:
            _get(client)
        assert exc_info.value.status_code == 200
        assert "body_length" in str(exc_info.value)

    def test_connection_error_propagates(self):
        client = RecordingHTTPClient([HTTPConnectionError("connection refused")])
        with pytest.raises(HTTPConnectionError):
            _get(client)

    def test_failed_fetch_is_not_cached(self):
        client = RecordingHTTPClient([HTTPResponse(status_code=503), _token_response("token-1")])
        with pytest.raises(HiddenLayerTransportError):
            _get(client)
        assert _get(client) == "token-1"


class TestEventLoops:
    def test_two_asyncio_run_calls_do_not_raise(self):
        client = _SlowTokenClient()

        async def contended(secret: str):
            return await asyncio.gather(
                _access_token(client, API_URL, CLIENT_ID, secret), _access_token(client, API_URL, CLIENT_ID, secret)
            )

        assert asyncio.run(contended("secret-a")) == ["token-1", "token-1"]
        assert asyncio.run(contended("secret-b")) == ["token-2", "token-2"]

    def test_closed_loops_do_not_accumulate_in_the_fetch_registry(self):
        client = _SlowTokenClient()

        async def contended(secret: str):
            await asyncio.gather(
                _access_token(client, API_URL, CLIENT_ID, secret), _access_token(client, API_URL, CLIENT_ID, secret)
            )
            return asyncio.get_running_loop()

        for run in range(5):
            asyncio.run(contended(f"secret-{run}"))
        last_loop = asyncio.run(contended("secret-last"))
        gc.collect()

        assert _token._fetches == {}
        assert last_loop.is_closed()


def test_secret_and_token_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    client = RecordingHTTPClient([_token_response("token-value-1"), HTTPResponse(status_code=401, content=b"nope")])

    assert _get(client) == "token-value-1"
    with pytest.raises(HiddenLayerTransportError) as exc_info:
        _get(client, rejected="token-value-1")

    for text in (caplog.text, str(exc_info.value)):
        assert CLIENT_SECRET not in text
        assert "token-value-1" not in text
