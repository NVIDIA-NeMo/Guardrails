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

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nemoguardrails.library.hallucination.anchor.actions import check_anchor_drift


@pytest.mark.asyncio
async def test_check_anchor_drift_missing_key(monkeypatch):
    monkeypatch.delenv("ANCHOR_API_KEY", raising=False)
    with pytest.raises(ValueError):
        await check_anchor_drift(context={})


@pytest.mark.asyncio
async def test_check_anchor_drift_missing_context_vars(monkeypatch):
    monkeypatch.setenv("ANCHOR_API_KEY", "test_key")
    assert await check_anchor_drift(context={"last_bot_message": ""}) is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_check_anchor_drift_allow(mock_post, monkeypatch):
    monkeypatch.setenv("ANCHOR_API_KEY", "test_key")
    mock_response = httpx.Response(200, json={"allow": True}, request=httpx.Request("POST", "url"))
    mock_post.return_value = mock_response

    res = await check_anchor_drift(context={"last_bot_message": "hello", "relevant_chunks": "hello world"})
    assert res is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_check_anchor_drift_block(mock_post, monkeypatch):
    monkeypatch.setenv("ANCHOR_API_KEY", "test_key")
    mock_response = httpx.Response(200, json={"allow": False}, request=httpx.Request("POST", "url"))
    mock_post.return_value = mock_response

    res = await check_anchor_drift(context={"last_bot_message": "hello", "relevant_chunks": "hello world"})
    assert res is False


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_check_anchor_drift_non_200(mock_post, monkeypatch):
    monkeypatch.setenv("ANCHOR_API_KEY", "test_key")
    mock_response = httpx.Response(500, request=httpx.Request("POST", "url"))
    mock_post.return_value = mock_response

    res = await check_anchor_drift(context={"last_bot_message": "hello", "relevant_chunks": "hello world"})
    assert res is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_check_anchor_drift_exception(mock_post, monkeypatch):
    monkeypatch.setenv("ANCHOR_API_KEY", "test_key")
    mock_post.side_effect = httpx.RequestError("error")

    res = await check_anchor_drift(context={"last_bot_message": "hello", "relevant_chunks": "hello world"})
    assert res is True
