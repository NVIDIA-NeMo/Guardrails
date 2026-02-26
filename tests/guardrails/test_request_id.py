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

"""Tests for per-request correlation ID propagation.

Verifies that generate_async() stamps a unique request ID via ContextVar
and that the same ID is visible at every layer (rails_manager, model_manager)
throughout the request lifecycle.
"""

import asyncio
import re
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.guardrails_types import RailResult, get_request_id, reset_request_id, set_new_request_id
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG, NEMOGUARDS_CONFIG

REQUEST_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")


class SingleUseBarrier:
    """Single-use asyncio barrier compatible with Python 3.10+, replacement for asyncio.Barrier"""

    def __init__(self, parties: int):
        self._parties = parties
        self._count = 0
        self._event = asyncio.Event()

    async def wait(self):
        self._count += 1
        if self._count >= self._parties:
            self._event.set()
        await self._event.wait()


class TestResetRequestId:
    """Direct unit tests for the reset_request_id() public API."""

    def test_reset_restores_default(self):
        """reset_request_id restores the ContextVar to its default value."""
        assert get_request_id() == "no-req-id"
        token = set_new_request_id()
        assert get_request_id() != "no-req-id"
        reset_request_id(token)
        assert get_request_id() == "no-req-id"

    def test_reset_restores_previous_value(self):
        """Nested set/reset restores the outer value, not the default."""
        outer_token = set_new_request_id()
        outer_id = get_request_id()

        inner_token = set_new_request_id()
        assert get_request_id() != outer_id

        reset_request_id(inner_token)
        assert get_request_id() == outer_id

        reset_request_id(outer_token)
        assert get_request_id() == "no-req-id"


@pytest.fixture
@patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
def iorails():
    config = RailsConfig.from_content(config=NEMOGUARDS_CONFIG)
    return IORails(config)


def _make_capturing_mock(captured_ids: list, key: str, return_value):
    """Create an AsyncMock whose side_effect records the current request ID."""

    async def _side_effect(*args, **kwargs):
        captured_ids.append((key, get_request_id()))
        return return_value

    return _side_effect


class TestSingleRequest:
    """A single generate_async call gets one request ID visible at every layer."""

    @pytest.mark.asyncio
    async def test_request_id_is_valid_hex(self, iorails):
        """The generated request ID is an 8-character hex string."""
        captured_ids = []

        iorails.rails_manager.is_input_safe = _make_capturing_mock(captured_ids, "input", RailResult(is_safe=True))
        iorails.model_manager.generate_async = _make_capturing_mock(captured_ids, "llm", "Hello")
        iorails.rails_manager.is_output_safe = _make_capturing_mock(captured_ids, "output", RailResult(is_safe=True))

        await iorails.generate_async([{"role": "user", "content": "hi"}])

        assert len(captured_ids) == 3
        for _, rid in captured_ids:
            assert REQUEST_ID_PATTERN.match(rid), f"Invalid request ID format: {rid}"

    @pytest.mark.asyncio
    async def test_same_id_across_all_layers(self, iorails):
        """Input rails, LLM call, and output rails all see the same request ID."""
        captured_ids = []

        iorails.rails_manager.is_input_safe = _make_capturing_mock(captured_ids, "input", RailResult(is_safe=True))
        iorails.model_manager.generate_async = _make_capturing_mock(captured_ids, "llm", "Hello")
        iorails.rails_manager.is_output_safe = _make_capturing_mock(captured_ids, "output", RailResult(is_safe=True))

        await iorails.generate_async([{"role": "user", "content": "hi"}])

        ids = [rid for _, rid in captured_ids]
        assert ids[0] == ids[1] == ids[2]

    @pytest.mark.asyncio
    async def test_request_id_reset_after_completion(self, iorails):
        """After generate_async returns, the ContextVar is reset to default."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value="Hello")
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        await iorails.generate_async([{"role": "user", "content": "hi"}])

        assert get_request_id() == "no-req-id"

    @pytest.mark.asyncio
    async def test_request_id_reset_after_input_blocked(self, iorails):
        """ContextVar is reset even when the request is blocked at input."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="blocked"))

        await iorails.generate_async([{"role": "user", "content": "bad"}])

        assert get_request_id() == "no-req-id"

    @pytest.mark.asyncio
    async def test_request_id_reset_after_output_blocked(self, iorails):
        """ContextVar is reset even when the request is blocked at output."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value="bad response")
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=False, reason="blocked"))

        await iorails.generate_async([{"role": "user", "content": "hi"}])

        assert get_request_id() == "no-req-id"

    @pytest.mark.asyncio
    async def test_request_id_reset_after_exception(self, iorails):
        """ContextVar is reset even when generate_async raises an exception."""
        iorails.rails_manager.is_input_safe = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await iorails.generate_async([{"role": "user", "content": "hi"}])

        assert get_request_id() == "no-req-id"


class TestMultipleSequentialRequests:
    """Sequential generate_async calls each get a unique request ID."""

    @pytest.mark.asyncio
    async def test_unique_ids_per_request(self, iorails):
        """Each sequential call produces a different request ID."""
        ids_per_request = []

        async def capture_input(*args, **kwargs):
            ids_per_request.append(get_request_id())
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = capture_input
        iorails.model_manager.generate_async = AsyncMock(return_value="Hello")
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        for _ in range(5):
            await iorails.generate_async([{"role": "user", "content": "hi"}])

        assert len(ids_per_request) == 5
        assert len(set(ids_per_request)) == 5, f"Expected 5 unique IDs, got: {ids_per_request}"

    @pytest.mark.asyncio
    async def test_id_consistency_within_each_request(self, iorails):
        """Within each request, all layers see the same ID; across requests, IDs differ."""
        request_snapshots = []

        async def capture_input(*args, **kwargs):
            request_snapshots.append(("input", get_request_id()))
            return RailResult(is_safe=True)

        async def capture_llm(*args, **kwargs):
            request_snapshots.append(("llm", get_request_id()))
            return "Hello"

        async def capture_output(*args, **kwargs):
            request_snapshots.append(("output", get_request_id()))
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = capture_input
        iorails.model_manager.generate_async = capture_llm
        iorails.rails_manager.is_output_safe = capture_output

        for _ in range(3):
            await iorails.generate_async([{"role": "user", "content": "hi"}])

        # 3 calls per request × 3 requests = 9 captures
        assert len(request_snapshots) == 9

        # Group by request (batches of 3)
        request_ids = []
        for i in range(0, 9, 3):
            batch = request_snapshots[i : i + 3]
            ids_in_batch = [rid for _, rid in batch]
            # All layers within a request share the same ID
            assert ids_in_batch[0] == ids_in_batch[1] == ids_in_batch[2]
            request_ids.append(ids_in_batch[0])

        # Each request has a different ID
        assert len(set(request_ids)) == 3


class TestMultipleConcurrentRequests:
    """Concurrent generate_async calls each get their own isolated request ID."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_have_unique_ids(self, iorails):
        """Multiple concurrent requests each get a distinct request ID."""
        captured = []
        barrier = SingleUseBarrier(3)

        async def capture_input(*args, **kwargs):
            rid = get_request_id()
            captured.append(("input", rid))
            # Synchronize so all requests overlap
            await barrier.wait()
            return RailResult(is_safe=True)

        async def capture_llm(*args, **kwargs):
            captured.append(("llm", get_request_id()))
            return "Hello"

        async def capture_output(*args, **kwargs):
            captured.append(("output", get_request_id()))
            return RailResult(is_safe=True)

        iorails.rails_manager.is_input_safe = capture_input
        iorails.model_manager.generate_async = capture_llm
        iorails.rails_manager.is_output_safe = capture_output

        messages = [{"role": "user", "content": "hi"}]
        results = await asyncio.gather(
            iorails.generate_async(messages),
            iorails.generate_async(messages),
            iorails.generate_async(messages),
        )

        # All 3 requests completed successfully
        assert len(results) == 3
        for r in results:
            assert r["role"] == "assistant"

        # Extract the unique request IDs seen across all captures
        all_ids = {rid for _, rid in captured}
        assert len(all_ids) == 3, f"Expected 3 unique IDs across concurrent requests, got: {all_ids}"

    @pytest.mark.asyncio
    async def test_concurrent_requests_maintain_isolation(self, iorails):
        """Each concurrent request sees a consistent ID across its own layers."""
        # Map from task → list of captured IDs
        task_ids: dict[int, list[str]] = {}
        barrier = SingleUseBarrier(3)

        async def make_iorails_call(task_num: int):
            captured = []

            async def capture_input(*args, **kwargs):
                captured.append(get_request_id())
                # Insert a barrier which waits for all three make_iorails_calls to complete
                await barrier.wait()
                return RailResult(is_safe=True)

            async def capture_llm(*args, **kwargs):
                captured.append(get_request_id())
                return "Hello"

            async def capture_output(*args, **kwargs):
                captured.append(get_request_id())
                return RailResult(is_safe=True)

            # Each concurrent call needs its own IORails with independent mocks
            config = iorails.config
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
                engine = IORails(config)
            engine.rails_manager.is_input_safe = capture_input
            engine.model_manager.generate_async = capture_llm
            engine.rails_manager.is_output_safe = capture_output

            await engine.generate_async([{"role": "user", "content": "hi"}])
            task_ids[task_num] = captured

        await asyncio.gather(
            make_iorails_call(0),
            make_iorails_call(1),
            make_iorails_call(2),
        )

        # Each task captured 3 IDs (input, llm, output)
        for task_num, ids in task_ids.items():
            assert len(ids) == 3, f"Task {task_num} captured {len(ids)} IDs"
            assert ids[0] == ids[1] == ids[2], f"Task {task_num} saw inconsistent IDs: {ids}"

        # All 3 tasks had different request IDs
        unique_ids = {ids[0] for ids in task_ids.values()}
        assert len(unique_ids) == 3, f"Expected 3 unique IDs, got: {unique_ids}"

    @pytest.mark.asyncio
    async def test_contextvar_reset_after_concurrent_requests(self, iorails):
        """ContextVar is back to default after all concurrent requests complete."""
        iorails.rails_manager.is_input_safe = AsyncMock(return_value=RailResult(is_safe=True))
        iorails.model_manager.generate_async = AsyncMock(return_value="Hello")
        iorails.rails_manager.is_output_safe = AsyncMock(return_value=RailResult(is_safe=True))

        messages = [{"role": "user", "content": "hi"}]
        await asyncio.gather(
            iorails.generate_async(messages),
            iorails.generate_async(messages),
        )

        assert get_request_id() == "no-req-id"


class TestEndToEndPropagation:
    """Request ID propagates through the full stack: IORails -> RailsManager -> ModelManager -> ModelEngine."""

    @pytest.fixture
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"})
    def iorails_content_safety(self):
        """IORails with content-safety-only config (input + output, no jailbreak API)."""
        config = RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)
        return IORails(config)

    @pytest.mark.asyncio
    async def test_same_id_from_iorails_through_model_engine(self, iorails_content_safety):
        """The request ID set in IORails.generate_async is visible in both RailsManager and ModelEngine."""
        engine = iorails_content_safety
        captured_ids: list[tuple[str, str]] = []

        # --- Wrap RailsManager entry points to capture the ID at that layer ---
        original_is_input_safe = engine.rails_manager.is_input_safe
        original_is_output_safe = engine.rails_manager.is_output_safe

        async def capturing_is_input_safe(*args, **kwargs):
            captured_ids.append(("rails_manager_input", get_request_id()))
            return await original_is_input_safe(*args, **kwargs)

        async def capturing_is_output_safe(*args, **kwargs):
            captured_ids.append(("rails_manager_output", get_request_id()))
            return await original_is_output_safe(*args, **kwargs)

        engine.rails_manager.is_input_safe = capturing_is_input_safe
        engine.rails_manager.is_output_safe = capturing_is_output_safe

        # --- Patch ModelEngine.call to capture the ID at the HTTP layer ---
        async def capturing_call(self_engine, messages, **kwargs):
            captured_ids.append((f"model_engine:{self_engine.model_name}", get_request_id()))
            safe_response = '{"User Safety": "safe", "Response Safety": "safe"}'
            return {"choices": [{"message": {"content": safe_response}}]}

        with patch.object(ModelEngine, "call", capturing_call):
            # Skip the "not started" check since we don't call engine.start()
            for model_engine in engine.model_manager._engines.values():
                model_engine._running = True

            await engine.generate_async([{"role": "user", "content": "hello"}])

        # We expect 5 captures:
        #   1. rails_manager_input
        #   2. model_engine:content_safety  (input check)
        #   3. model_engine:main            (LLM generation)
        #   4. rails_manager_output
        #   5. model_engine:content_safety  (output check)
        assert len(captured_ids) == 5, f"Expected 5 captures, got {len(captured_ids)}: {captured_ids}"

        ids = [rid for _, rid in captured_ids]
        # Every layer — RailsManager and ModelEngine — saw the same request ID
        assert len(set(ids)) == 1, f"Request IDs differ across layers: {captured_ids}"
        # And it's a valid hex ID
        assert REQUEST_ID_PATTERN.match(ids[0]), f"Invalid request ID format: {ids[0]}"
