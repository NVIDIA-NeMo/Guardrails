# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for base_engine module."""

import asyncio

import pytest
from aiohttp_retry import RetryClient

from nemoguardrails.guardrails._http import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_TOTAL,
)
from nemoguardrails.guardrails.base_engine import BaseEngine


class TestBaseEngineDefaults:
    """Test BaseEngine initializes with correct defaults."""

    def test_default_timeout(self):
        engine = BaseEngine()
        assert engine._timeout.total == DEFAULT_TIMEOUT_TOTAL
        assert engine._timeout.connect == DEFAULT_TIMEOUT_CONNECT

    def test_default_retry_attempts(self):
        engine = BaseEngine()
        assert engine._retry_options.attempts == DEFAULT_MAX_ATTEMPTS

    def test_custom_parameters(self):
        engine = BaseEngine(timeout_total=60, timeout_connect=10, max_attempts=5)
        assert engine._timeout.total == 60
        assert engine._timeout.connect == 10
        assert engine._retry_options.attempts == 5

    def test_initial_state(self):
        engine = BaseEngine()
        assert engine._client is None
        assert engine._running is False


class TestBaseEngineStartStop:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_client(self):
        engine = BaseEngine()
        await engine.start()
        try:
            assert engine._running is True
            assert isinstance(engine._client, RetryClient)
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        engine = BaseEngine()
        await engine.start()
        try:
            client_after_first = engine._client
            await engine.start()
            assert engine._client is client_after_first
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_client(self):
        engine = BaseEngine()
        await engine.start()
        await engine.stop()
        assert engine._running is False
        assert engine._client is None

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        engine = BaseEngine()
        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_start_after_stop_creates_new_client(self):
        engine = BaseEngine()
        await engine.start()
        first_client = engine._client
        await engine.stop()

        await engine.start()
        try:
            assert engine._running is True
            assert engine._client is not first_client
        finally:
            await engine.stop()


class TestBaseEngineContextManager:
    """Test async context manager protocol."""

    @pytest.mark.asyncio
    async def test_aenter_starts_engine(self):
        engine = BaseEngine()
        async with engine as ctx:
            assert ctx is engine
            assert engine._running is True

    @pytest.mark.asyncio
    async def test_aexit_stops_engine(self):
        engine = BaseEngine()
        async with engine:
            pass
        assert engine._running is False
        assert engine._client is None


class TestBaseEngineConcurrency:
    """Test that the lock prevents races on start/stop."""

    @pytest.mark.asyncio
    async def test_concurrent_starts_create_single_client(self):
        engine = BaseEngine()
        try:
            await asyncio.gather(engine.start(), engine.start(), engine.start())
            assert engine._running is True
            assert isinstance(engine._client, RetryClient)
        finally:
            await engine.stop()

    @pytest.mark.asyncio
    async def test_concurrent_stops_close_cleanly(self):
        engine = BaseEngine()
        await engine.start()
        await asyncio.gather(engine.stop(), engine.stop(), engine.stop())
        assert engine._running is False
        assert engine._client is None
