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

"""Event-level public API (generate_events_async) for Colang 1.0.

Deterministic (FakeLLMModel + rule-based input rail), no cassette. Exercises the
conversation_events / colang_turns assembly the decomposition moved. The Colang 2.x
``process_events_async`` event/state path (A5.2 / A5.3) lives in ``test_colang_v2.py``.
"""

from __future__ import annotations

import pytest

from nemoguardrails import LLMRails
from tests.recorded.rails.public_api.configs import INPUT_RAILS_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]

_USER_EVENT = {"type": "UtteranceUserActionFinished", "final_transcript": "allowed input"}


async def test_generate_events_async_returns_event_stream():
    """A5.1: events in -> the full internal event stream out.

    A ``UtteranceUserActionFinished`` is processed through the input rail and generation
    (fake main), producing the ordered internal event stream. Event ids/timestamps are
    volatile, so the deterministic event-*type* sequence is pinned.
    """
    rails = LLMRails(load_config(INPUT_RAILS_CONFIG), llm=FakeLLMModel(responses=["hi"]), verbose=False)

    out_events = await rails.generate_events_async([_USER_EVENT])

    assert isinstance(out_events, list)
    assert out_events
    event_types = [event.get("type") for event in out_events]
    assert "BotMessage" in event_types
    assert event_types == snapshot(
        [
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "StartInputRails",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "StartInputRail",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "InputRailFinished",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "InputRailsFinished",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "UserMessage",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "BotMessage",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "StartUtteranceBotAction",
            "Listen",
        ]
    )


async def test_process_events_async_raises_on_colang_1():
    """A5.x boundary: ``process_events_async`` is a Colang 2.x API; on a Colang 1.0 config the
    1.0 runtime does not implement it and raises ``NotImplementedError``. Pinned so the
    decomposition preserves the boundary (the Colang 2.x path is covered in test_colang_v2)."""
    rails = LLMRails(load_config(INPUT_RAILS_CONFIG), llm=FakeLLMModel(responses=["hi"]), verbose=False)

    with pytest.raises(NotImplementedError):
        await rails.process_events_async([_USER_EVENT])
