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

"""State threading and explain/log isolation for the public ``generate_async`` API.

Both scenarios exercise logic the decomposition moved (``colang_turns``,
``serialization``, ``generation_context``) and are fully deterministic, so they use
``FakeLLMModel`` for the main generation and need no cassette: the state round-trip
and the per-call log isolation do not depend on a real provider response.
"""

from __future__ import annotations

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.normalization import normalize_generation_response, normalize_llm_calls
from tests.recorded.rails.public_api.configs import OPENAI_BASELINE_CONFIG, STATE_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


async def test_dialog_generate_async_state_round_trips_across_turns():
    """A Colang 1.0 ``state`` object round-trips and advances the flow.

    Turn 1 starts from an empty state (``{}``) and returns ``GenerationResponse.state``
    as ``{"events": [...]}``. Threading that state into turn 2 continues the multi-step
    greeting flow (``express greeting`` -> ``express greeting again``), which only
    happens if the prior turn's events were carried forward.
    """
    rails = LLMRails(
        load_config(STATE_CONFIG),
        llm=FakeLLMModel(responses=["  express greeting", "  express greeting"]),
        verbose=False,
    )

    turn1 = await rails.generate_async(
        messages=[{"role": "user", "content": "hi"}],
        state={},
        options={"log": {"activated_rails": True}},
    )

    assert isinstance(turn1, GenerationResponse)
    # The output state is the transcript-shaped {"events": [...]} object.
    assert isinstance(turn1.state, dict)
    assert set(turn1.state) == {"events"}
    assert isinstance(turn1.state["events"], list)
    assert turn1.state["events"]
    assert normalize_generation_response(turn1) == snapshot(
        {
            "response": [{"role": "assistant", "content": "Hey!"}],
            "activated_rails": [
                {
                    "type": "dialog",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {"type": "dialog", "name": "greeting", "decisions": ["express greeting"], "stop": False},
                {
                    "type": "generation",
                    "name": "generate bot message",
                    "decisions": ["execute retrieve_relevant_chunks", "execute generate_bot_message"],
                    "stop": False,
                },
            ],
            "llm_calls": [],
        }
    )

    turn2 = await rails.generate_async(
        messages=[{"role": "user", "content": "hi"}],
        state=turn1.state,
        options={"log": {"activated_rails": True}},
    )

    assert isinstance(turn2, GenerationResponse)
    assert isinstance(turn2.state, dict)
    assert set(turn2.state) == {"events"}
    assert turn2.state["events"]
    # Continuity: the flow advanced to "express greeting again" because turn 1's state
    # was threaded in, so the bot answers "Hey again!" instead of repeating "Hey!".
    assert normalize_generation_response(turn2) == snapshot(
        {
            "response": [{"role": "assistant", "content": "Hey again!"}],
            "activated_rails": [
                {
                    "type": "dialog",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {"type": "dialog", "name": "greeting", "decisions": ["express greeting again"], "stop": False},
                {
                    "type": "generation",
                    "name": "generate bot message",
                    "decisions": ["execute retrieve_relevant_chunks", "execute generate_bot_message"],
                    "stop": False,
                },
            ],
            "llm_calls": [],
        }
    )


async def test_baseline_generate_async_log_does_not_leak_across_sequential_calls():
    """Per-call ``log.llm_calls`` isolation across two calls on one ``LLMRails``.

    Invariant (holds both pre- and post-decomposition): a call's
    ``GenerationResponse.log.llm_calls`` contains only that call's LLM calls, never the
    previous call's. This is the substantive no-leak guarantee, and it is what is
    asserted here.

    Behavior change surfaced (flagged, deliberately NOT pinned as an invariant): the
    running ``rails.explain().llm_calls`` tally differs between the two code paths. Pre-
    decomposition it *accumulates* across sequential calls in the same async context
    (``["first reply", "second reply"]``); post-decomposition ``generation_context``
    resets it per call (``["second reply"]``). Both are consistent with the no-leak
    guarantee on ``log.llm_calls``; only the ``explain()`` tally changed (an apparent
    improvement worth confirming as intended). Because that count cannot hold on both
    sides, we assert only that ``explain()``'s most recent entry is the current call.
    """
    rails = LLMRails(
        load_config(OPENAI_BASELINE_CONFIG),
        llm=FakeLLMModel(responses=["first reply", "second reply"]),
        verbose=False,
    )
    options = {"log": {"llm_calls": True}}

    first = await rails.generate_async(messages=[{"role": "user", "content": "first turn"}], options=options)
    second = await rails.generate_async(messages=[{"role": "user", "content": "second turn"}], options=options)

    assert isinstance(first, GenerationResponse)
    assert isinstance(second, GenerationResponse)

    # log.llm_calls is isolated per call: the second response does not carry the first.
    assert normalize_llm_calls(first) == snapshot(
        [
            {
                "task": "general",
                "provider": "test",
                "model": "fake",
                "completion": "first reply",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        ]
    )
    assert normalize_llm_calls(second) == snapshot(
        [
            {
                "task": "general",
                "provider": "test",
                "model": "fake",
                "completion": "second reply",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        ]
    )
    assert [c["completion"] for c in normalize_llm_calls(second)] == ["second reply"]

    # Stable across both code paths: explain()'s most recent call reflects this call.
    assert rails.explain().llm_calls[-1].completion == "second reply"
