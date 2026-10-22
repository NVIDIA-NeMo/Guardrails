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

"""Every model call has its own round-trip ID, and both of its legs report under it."""

import asyncio
import uuid

import pytest
import pytest_asyncio

from nemoguardrails import RailsConfig
from nemoguardrails.context import get_roundtrip_id, new_roundtrip_id, roundtrip_id
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.options import RailType
from nemoguardrails.types import LLMResponse, LLMResponseChunk
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG
from tests.utils import TestChat

USER = [{"role": "user", "content": "hi"}]
EXCHANGE = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

RECORDING_COLANG = """
define flow record input
  execute record_leg(leg="input")

define flow record output
  execute record_leg(leg="output")
"""

RECORDING_YAML = """
rails:
    input:
        flows:
            - record input
    output:
        flows:
            - record output
"""


class TestRoundtripIdHelper:
    def test_unset_by_default(self):
        assert get_roundtrip_id() is None

    def test_sets_the_id_inside_the_block_and_restores_it_after(self):
        with roundtrip_id("outer"):
            with roundtrip_id("inner"):
                assert get_roundtrip_id() == "inner"
            with roundtrip_id(None):
                assert get_roundtrip_id() is None
            assert get_roundtrip_id() == "outer"
        assert get_roundtrip_id() is None

    def test_restores_the_previous_id_when_the_block_raises(self):
        with pytest.raises(RuntimeError):
            with roundtrip_id("r-1"):
                raise RuntimeError("boom")
        assert get_roundtrip_id() is None

    def test_a_new_round_trip_mints_a_uuid(self):
        with new_roundtrip_id() as minted:
            assert get_roundtrip_id() == minted
            uuid.UUID(minted)
        assert get_roundtrip_id() is None

    def test_a_new_round_trip_is_a_different_id_every_time(self):
        with new_roundtrip_id() as first:
            pass
        with new_roundtrip_id() as second:
            pass
        assert first != second

    def test_a_new_round_trip_keeps_an_id_the_caller_set(self):
        with roundtrip_id("r-1"):
            with new_roundtrip_id() as current:
                assert current == "r-1"
            assert get_roundtrip_id() == "r-1"


@pytest_asyncio.fixture
async def iorails():
    async with started_iorails(NEMOGUARDS_CONFIG) as engine:
        yield engine


def _recording(seen: list, stage: str, result):
    async def record(*args, **kwargs):
        seen.append((stage, get_roundtrip_id()))
        return result

    return record


def _wire(engine: IORails, seen: list) -> None:
    engine.rails_manager.is_input_safe = _recording(seen, "input", RailResult.allow())
    engine.engine_registry.model_call = _recording(seen, "llm", LLMResponse(content="Hello"))
    engine.rails_manager.is_output_safe = _recording(seen, "output", RailResult.allow())


class TestOnIORails:
    @pytest.mark.asyncio
    async def test_every_stage_of_one_request_shares_the_id(self, iorails):
        """The rails run on the work queue, so this also covers the ID reaching a queued item."""
        seen = []
        _wire(iorails, seen)

        await iorails.generate_async(messages=USER)

        stages = [stage for stage, _ in seen]
        assert stages == ["input", "llm", "output"]
        ids = {request for _, request in seen}
        assert len(ids) == 1 and None not in ids

    @pytest.mark.asyncio
    async def test_each_request_gets_its_own_id(self, iorails):
        seen = []
        _wire(iorails, seen)

        await iorails.generate_async(messages=USER)
        await iorails.check_async(USER, rail_types=[RailType.INPUT])

        inputs = [request for stage, request in seen if stage == "input"]
        assert len(set(inputs)) == 2

    @pytest.mark.asyncio
    async def test_an_id_the_caller_set_is_not_overwritten(self, iorails):
        seen = []
        _wire(iorails, seen)

        with roundtrip_id("r-1"):
            await iorails.generate_async(messages=USER)
            await iorails.check_async(USER, rail_types=[RailType.INPUT])

        assert {request for _, request in seen} == {"r-1"}

    @pytest.mark.asyncio
    async def test_a_stream_shares_the_id_with_the_rails_that_opened_it(self):
        seen = []
        input_only = {**NEMOGUARDS_CONFIG, "rails": {**NEMOGUARDS_CONFIG["rails"], "output": {"flows": []}}}

        async def stream(*args, **kwargs):
            seen.append(("llm", get_roundtrip_id()))
            yield LLMResponseChunk(delta_content="Hello")

        async with started_iorails(input_only) as engine:
            _wire(engine, seen)
            engine.engine_registry.stream_model_call = stream
            [chunk async for chunk in engine.stream_async(messages=USER)]

        assert sorted(stage for stage, _ in seen) == ["input", "llm"]
        ids = {request for _, request in seen}
        assert len(ids) == 1 and None not in ids

    @staticmethod
    async def _streaming_engine(seen: list):
        """An engine whose only rail is an input rail, with a three-chunk model."""
        input_only = {**NEMOGUARDS_CONFIG, "rails": {**NEMOGUARDS_CONFIG["rails"], "output": {"flows": []}}}

        async def stream(*args, **kwargs):
            seen.append(("llm", get_roundtrip_id()))
            for token in ("a", "b", "c"):
                yield LLMResponseChunk(delta_content=token)

        engine = started_iorails(input_only)
        started = await engine.__aenter__()
        _wire(started, seen)
        started.engine_registry.stream_model_call = stream
        return engine, started

    @staticmethod
    async def _drain_in_another_task(chunks) -> str:
        """Pull the first chunk here and the rest from a child task, as the server does."""
        delivered = [await anext(chunks)]

        async def drain():
            async for chunk in chunks:
                delivered.append(chunk)

        await asyncio.create_task(drain())
        return "".join(delivered)

    @pytest.mark.asyncio
    async def test_a_stream_drained_by_another_task_keeps_the_id(self):
        seen = []
        engine, started = await self._streaming_engine(seen)
        try:
            delivered = await self._drain_in_another_task(started.stream_async(messages=USER))
        finally:
            await engine.__aexit__(None, None, None)

        assert delivered == "abc"
        ids = {request for _, request in seen}
        assert len(ids) == 1 and None not in ids
        assert get_roundtrip_id() is None

    @pytest.mark.asyncio
    async def test_abandoning_a_stream_leaves_the_caller_without_an_id(self):
        """A client that disconnects mid-stream closes the generator from its own context."""
        seen = []
        engine, started = await self._streaming_engine(seen)
        try:
            async for _ in started.stream_async(messages=USER):
                break
        finally:
            await engine.__aexit__(None, None, None)

        assert get_roundtrip_id() is None

    @pytest.mark.asyncio
    async def test_a_stream_does_not_lend_its_id_to_the_next_request(self):
        seen = []
        engine, started = await self._streaming_engine(seen)
        try:
            await self._drain_in_another_task(started.stream_async(messages=USER))
            await started.check_async(USER, rail_types=[RailType.INPUT])
        finally:
            await engine.__aexit__(None, None, None)

        assert get_roundtrip_id() is None
        inputs = [request for stage, request in seen if stage == "input"]
        assert len(inputs) == 2 and len(set(inputs)) == 2

    @pytest.mark.asyncio
    async def test_no_id_outside_a_request(self, iorails):
        seen = []
        _wire(iorails, seen)

        await iorails.generate_async(messages=USER)

        assert get_roundtrip_id() is None


class TestOnLLMRails:
    @staticmethod
    def _recording_chat(seen: list) -> TestChat:
        chat = TestChat(RailsConfig.from_content(RECORDING_COLANG, RECORDING_YAML), llm_completions=[])

        async def record_leg(leg: str):
            seen.append((leg, get_roundtrip_id()))

        chat.app.register_action(record_leg, "record_leg")
        return chat

    @pytest.mark.asyncio
    async def test_both_legs_of_one_model_call_share_the_id(self):
        seen = []
        chat = self._recording_chat(seen)

        await chat.app.check_async(EXCHANGE)

        assert [leg for leg, _ in seen] == ["input", "output"]
        ids = {request for _, request in seen}
        assert len(ids) == 1 and None not in ids

    @pytest.mark.asyncio
    async def test_each_model_call_gets_its_own_id(self):
        seen = []
        chat = self._recording_chat(seen)

        await chat.app.check_async(EXCHANGE)
        await chat.app.check_async(EXCHANGE)

        assert len({request for _, request in seen}) == 2

    @pytest.mark.asyncio
    async def test_an_id_the_caller_set_is_kept(self):
        seen = []
        chat = self._recording_chat(seen)

        with roundtrip_id("r-1"):
            await chat.app.check_async(EXCHANGE)
            await chat.app.check_async(EXCHANGE)

        assert {request for _, request in seen} == {"r-1"}
