# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

LIVE_TEST_MODE = os.environ.get("LIVE_TEST") or os.environ.get("LIVE_TEST_MODE")


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
class TestProcessEventsAsyncV2x:
    """Integration tests for LLMRails.process_events_async with v2.x runtime.

    These tests would have caught issue #1505 where PR #1380 incorrectly
    assumed process_events_async returns dict and tried to convert with
    State(**output_state), causing TypeError.

    The bug was introduced because:
    1. The type annotation in llmrails.py was wrong (said it returns dict)
    2. PR #1380 "fixed" chat.py based on the wrong annotation
    3. No tests called LLMRails.process_events_async() with v2.x
       (all tests called runtime.process_events() directly)
    """

    @pytest.mark.asyncio
    async def test_process_events_async_returns_state_object(self):
        """Test that LLMRails.process_events_async returns State object for v2.x, not dict.

        This is the critical test that would have caught the bug immediately.
        """
        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.colang.v2_x.runtime.flows import State

        config = RailsConfig.from_content(
            """
            import core

            flow main
              user said "hi"
              bot say "Hello!"
            """,
            """
            colang_version: "2.x"
            models:
              - type: main
                engine: openai
                model: gpt-4o-mini
            """,
        )

        rails = LLMRails(config)

        events = [{"type": "UtteranceUserActionFinished", "final_transcript": "hi"}]

        output_events, output_state = await rails.process_events_async(
            events, state=None
        )

        assert isinstance(
            output_state, State
        ), f"Expected State object, got {type(output_state)}"
        assert isinstance(output_events, list)
        assert len(output_events) > 0

    @pytest.mark.asyncio
    async def test_process_events_async_accepts_state_object(self):
        """Test that LLMRails.process_events_async accepts State object as input.

        The bug in PR #1380 also incorrectly converted State to dict using asdict()
        before passing to process_events_async. This test verifies that passing
        State objects directly works correctly.
        """
        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.colang.v2_x.runtime.flows import State
        from nemoguardrails.utils import new_event_dict

        config = RailsConfig.from_content(
            """
            import core

            flow main
              user said "hi"
              bot say "Hello!"
              user said "bye"
              bot say "Goodbye!"
            """,
            """
            colang_version: "2.x"
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """,
        )

        rails = LLMRails(config)

        events = [{"type": "UtteranceUserActionFinished", "final_transcript": "hi"}]

        output_events_1, output_state_1 = await rails.process_events_async(
            events, state=None
        )

        assert isinstance(output_state_1, State)

        events_2 = []
        for event in output_events_1:
            if event["type"] == "StartUtteranceBotAction":
                events_2.append(
                    new_event_dict(
                        "UtteranceBotActionFinished",
                        action_uid=event["action_uid"],
                        is_success=True,
                        final_script=event["script"],
                    )
                )

        events_2.append(
            {"type": "UtteranceUserActionFinished", "final_transcript": "bye"}
        )

        output_events_2, output_state_2 = await rails.process_events_async(
            events_2, state=output_state_1
        )

        assert isinstance(
            output_state_2, State
        ), "Second call should also return State object when passing State as input"


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
class TestChatV2xE2E:
    """End-to-end tests for chat CLI with v2.x runtime.

    These tests exercise the actual chat.py code paths that were broken by PR #1380.
    """

    @pytest.mark.asyncio
    async def test_chat_v2x_with_real_llm(self):
        """E2E test of v2.x chat with real LLM.

        This requires LIVE_TEST_MODE=1 and OpenAI API key.
        """
        from unittest.mock import patch

        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.cli.chat import _run_chat_v2_x
        from nemoguardrails.colang.v2_x.runtime.flows import State

        config = RailsConfig.from_content(
            """
            import core

            flow main
              user said "hi"
              bot say "Hello from v2.x!"
            """,
            """
            colang_version: "2.x"
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """,
        )

        rails = LLMRails(config)

        simulated_input = ["hi", "exit"]
        input_iter = iter(simulated_input)

        def mock_input(*args, **kwargs):
            try:
                return next(input_iter)
            except StopIteration:
                raise KeyboardInterrupt()

        with patch("builtins.input", side_effect=mock_input):
            try:
                await _run_chat_v2_x(rails)
            except (KeyboardInterrupt, StopIteration):
                pass

    @pytest.mark.asyncio
    async def test_chat_v2x_process_events_flow(self):
        """Test the exact code path that was broken in chat.py.

        This simulates what _run_chat_v2_x does internally.
        """
        from dataclasses import dataclass, field
        from typing import List, Optional

        from nemoguardrails import LLMRails, RailsConfig
        from nemoguardrails.colang.v2_x.runtime.flows import State

        @dataclass
        class ChatState:
            state: Optional[State] = None
            input_events: List[dict] = field(default_factory=list)
            output_events: List[dict] = field(default_factory=list)
            output_state: Optional[State] = None

        config = RailsConfig.from_content(
            """
            import core

            flow main
              user said "hi"
              bot say "Hello!"
            """,
            """
            colang_version: "2.x"
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo
            """,
        )

        rails = LLMRails(config)
        chat_state = ChatState()

        chat_state.input_events = [
            {"type": "UtteranceUserActionFinished", "final_transcript": "hi"}
        ]

        input_events_copy = chat_state.input_events.copy()
        chat_state.input_events = []

        output_events, output_state = await rails.process_events_async(
            input_events_copy,
            chat_state.state,
        )
        chat_state.output_events = output_events

        if output_state:
            chat_state.output_state = output_state

        assert isinstance(chat_state.output_state, State)
        assert len(chat_state.output_events) > 0
