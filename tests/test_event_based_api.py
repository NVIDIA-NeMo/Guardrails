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
import json

from nemoguardrails import RailsConfig
from tests.utils import TestChat, any_event_conforms


def test_1():
    config = RailsConfig.from_content(
        """
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting
        """
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
        ],
    )

    new_events = chat.app.generate_events(
        events=[{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    )

    # We don't want to pin the exact number here in the test as the exact number of events
    # can vary as the implementation changes.
    assert len(new_events) > 10

    print(json.dumps(new_events, indent=True))

    # We check certain key events are present.
    assert any_event_conforms(
        {"intent": "express greeting", "type": "UserIntent"}, new_events
    )
    assert any_event_conforms(
        {"intent": "express greeting", "type": "BotIntent"}, new_events
    )
    assert any_event_conforms(
        {"script": "Hello!", "type": "StartUtteranceBotAction"}, new_events
    )
    assert any_event_conforms({"type": "Listen"}, new_events)


CONFIG_WITH_EVENT = """
    define user express greeting
      "hello"

    define flow
      user express greeting
      bot express greeting

    define flow
      event UserSilent
      bot ask if more time needed
"""


def test_2():
    """Test a flow that uses a custom event, i.e., `user silent`."""
    config = RailsConfig.from_content(CONFIG_WITH_EVENT)

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
            '  "Do you need more time?"',
        ],
    )

    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "Hello!"}]
    new_events = chat.app.generate_events(events)

    any_event_conforms(
        {"type": "StartUtteranceBotAction", "script": "Hello!"}, new_events
    )

    events.extend(new_events)
    events.append({"type": "UserSilent"})

    new_events = chat.app.generate_events(events)

    any_event_conforms(
        {
            "type": "StartUtteranceBotAction",
            "script": "Do you need more time?",
        },
        new_events,
    )


def test_3():
    """Test a flow that uses a custom event, i.e., `user silent` using the messages API."""
    config = RailsConfig.from_content(CONFIG_WITH_EVENT)

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hello!"',
            '  "Do you need more time?"',
        ],
    )

    messages = [{"role": "user", "content": "Hello!"}]

    new_message = chat.app.generate(messages=messages)

    assert new_message == {"role": "assistant", "content": "Hello!"}

    messages.append(new_message)
    messages.append({"role": "event", "event": {"type": "UserSilent"}})

    new_message = chat.app.generate(messages=messages)

    assert new_message == {"role": "assistant", "content": "Do you need more time?"}
