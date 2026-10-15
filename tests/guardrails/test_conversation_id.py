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

"""The request-scoped conversation ID reaches every stage of an IORails request, and only that request."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from nemoguardrails.context import conversation_id, get_conversation_id
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailType
from nemoguardrails.types import LLMResponse, LLMResponseChunk
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

USER = [{"role": "user", "content": "hi"}]


class TestConversationIdHelper:
    def test_unset_by_default(self):
        assert get_conversation_id() is None

    def test_sets_the_id_inside_the_block_and_restores_it_after(self):
        with conversation_id("outer"):
            with conversation_id("inner"):
                assert get_conversation_id() == "inner"
            with conversation_id(None):
                assert get_conversation_id() is None
            assert get_conversation_id() == "outer"
        assert get_conversation_id() is None

    def test_restores_the_previous_id_when_the_block_raises(self):
        with pytest.raises(RuntimeError):
            with conversation_id("c-1"):
                raise RuntimeError("boom")
        assert get_conversation_id() is None


@pytest_asyncio.fixture
async def iorails():
    async with started_iorails(NEMOGUARDS_CONFIG) as engine:
        yield engine


def _recording(seen: list, stage: str, result):
    async def record(*args, **kwargs):
        seen.append((stage, get_conversation_id()))
        return result

    return record


def _wire(engine: IORails, seen: list) -> None:
    engine.rails_manager.is_input_safe = _recording(seen, "input", RailResult.allow())
    engine.engine_registry.model_call = _recording(seen, "llm", LLMResponse(content="Hello"))
    engine.rails_manager.is_output_safe = _recording(seen, "output", RailResult.allow())


class TestOnIORails:
    @pytest.mark.asyncio
    async def test_generate_sees_the_id_at_every_stage(self, iorails):
        seen = []
        _wire(iorails, seen)

        with conversation_id("c-1"):
            await iorails.generate_async(messages=USER)

        assert seen == [("input", "c-1"), ("llm", "c-1"), ("output", "c-1")]

    @pytest.mark.asyncio
    async def test_check_sees_the_id(self, iorails):
        seen = []
        _wire(iorails, seen)

        with conversation_id("c-1"):
            await iorails.check_async(USER, rail_types=[RailType.INPUT])

        assert seen == [("input", "c-1")]

    @pytest.mark.asyncio
    async def test_stream_sees_the_id(self):
        seen = []
        input_only = {**NEMOGUARDS_CONFIG, "rails": {**NEMOGUARDS_CONFIG["rails"], "output": {"flows": []}}}

        async def stream(*args, **kwargs):
            seen.append(("llm", get_conversation_id()))
            yield LLMResponseChunk(delta_content="Hello")

        async with started_iorails(input_only) as engine:
            _wire(engine, seen)
            engine.engine_registry.stream_model_call = stream
            with conversation_id("c-1"):
                [chunk async for chunk in engine.stream_async(messages=USER)]

        assert sorted(seen) == [("input", "c-1"), ("llm", "c-1")]

    @pytest.mark.asyncio
    async def test_the_first_requests_id_does_not_reach_later_requests(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            engine = IORails(RailsConfig.from_content(config=NEMOGUARDS_CONFIG))
        seen = []
        _wire(engine, seen)
        try:
            # The first request starts the engine's workers lazily, inside its own context.
            with conversation_id("first"):
                await engine.generate_async(messages=USER)
            await engine.generate_async(messages=USER)
            with conversation_id("third"):
                await engine.check_async(USER, rail_types=[RailType.INPUT])
        finally:
            await engine.stop()

        assert [conversation for stage, conversation in seen if stage == "input"] == ["first", None, "third"]

    @pytest.mark.asyncio
    async def test_concurrent_requests_each_see_their_own_id(self, iorails):
        seen = []
        _wire(iorails, seen)
        both_inside = asyncio.Event()
        inside = 0

        async def input_rail(*args, **kwargs):
            nonlocal inside
            inside += 1
            if inside == 2:
                both_inside.set()
            await both_inside.wait()
            seen.append(("input", get_conversation_id()))
            return RailResult.allow()

        iorails.rails_manager.is_input_safe = input_rail
        iorails.engine_registry.model_call = AsyncMock(return_value=LLMResponse(content="Hello"))

        async def request(conversation: str):
            with conversation_id(conversation):
                await iorails.generate_async(messages=USER)

        await asyncio.gather(request("a"), request("b"))

        assert sorted(conversation for stage, conversation in seen if stage == "input") == ["a", "b"]
        assert {conversation for stage, conversation in seen if stage == "output"} == {"a", "b"}
