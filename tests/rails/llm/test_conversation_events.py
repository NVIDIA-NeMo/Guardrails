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

from types import SimpleNamespace

import pytest

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.conversation import conversation_events
from nemoguardrails.rails.llm.conversation.conversation_events import (
    events_for_messages,
    events_history_cache_prefix,
)
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.utils import get_history_cache_key


class FakeRails:
    def __init__(self, config: RailsConfig):
        self.config = config
        self.events_history_cache = {}


def make_rails(config: RailsConfig):
    return FakeRails(config=config)


def test_conversation_events_public_exports():
    assert conversation_events.__all__ == [
        "ConversationEventSurface",
        "events_for_messages",
        "events_history_cache_prefix",
    ]


def test_events_for_messages_v1_converts_supported_message_roles(monkeypatch):
    monkeypatch.setattr(conversation_events, "new_uuid", lambda: "action-1")
    tool_calls = [{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}]
    passthrough_event = {"type": "UserSilent"}
    rails = make_rails(RailsConfig(models=[], colang_version="1.0"))

    events = events_for_messages(
        rails,
        [
            {"role": "context", "content": {"user_name": "Ada"}},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "event", "event": passthrough_event},
        ],
        state=None,
    )

    assert events[0] == {"type": "ContextUpdate", "data": {"user_name": "Ada"}}
    assert events[1] == {"type": "SystemMessage", "content": "Be concise."}
    assert events[2] == {"type": "UtteranceUserActionFinished", "final_transcript": "Hi"}
    assert events[3] == {"type": "UserMessage", "text": "Hi"}
    assert events[4]["type"] == "StartUtteranceBotAction"
    assert events[4]["script"] == "Hello"
    assert events[4]["action_uid"] == "action-1"
    assert events[5]["type"] == "UtteranceBotActionFinished"
    assert events[5]["final_script"] == "Hello"
    assert events[5]["action_uid"] == "action-1"
    assert events[6] == {"type": "BotToolCalls", "tool_calls": tool_calls}
    assert events[7] is passthrough_event


def test_events_for_messages_v1_reuses_longest_history_cache_prefix():
    config = RailsConfig(models=[], colang_version="1.0")
    rails = make_rails(config)
    cached_prefix = [{"type": "CachedEvent"}]
    cached_messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    rails.events_history_cache[get_history_cache_key(cached_messages)] = cached_prefix

    events = events_for_messages(
        rails,
        cached_messages + [{"role": "user", "content": "Next"}],
        state=None,
    )

    assert events == [
        {"type": "CachedEvent"},
        {"type": "UtteranceUserActionFinished", "final_transcript": "Next"},
    ]
    assert events is not cached_prefix
    assert cached_prefix == [{"type": "CachedEvent"}]


def test_events_history_cache_prefix_returns_copy_of_longest_match():
    events_history_cache = {}
    short_messages = [{"role": "user", "content": "Hi"}]
    long_messages = short_messages + [{"role": "assistant", "content": "Hello"}]
    cached_events = [{"type": "LongCachedEvent"}]
    events_history_cache[get_history_cache_key(short_messages)] = [{"type": "ShortCachedEvent"}]
    events_history_cache[get_history_cache_key(long_messages)] = cached_events

    prefix_len, events = events_history_cache_prefix(
        events_history_cache,
        long_messages + [{"role": "user", "content": "Next"}],
    )

    assert prefix_len == len(long_messages)
    assert events == [{"type": "LongCachedEvent"}]
    assert events is not cached_events


def test_events_for_messages_v1_final_tool_message_creates_synthetic_user_message_without_tool_input_rails():
    rails = make_rails(RailsConfig(models=[], colang_version="1.0"))

    events = events_for_messages(
        rails,
        [
            {"role": "user", "content": "Use the tool"},
            {"role": "tool", "content": "tool result", "name": "lookup", "tool_call_id": "call-1"},
        ],
        state=None,
    )

    assert events[-1] == {"type": "UserMessage", "text": "Use the tool"}


def test_events_for_messages_v1_final_tool_message_groups_tool_messages_when_tool_input_rails_exist():
    config = RailsConfig(models=[], colang_version="1.0")
    config.rails.tool_input.flows = ["check tool input"]
    rails = make_rails(config)

    events = events_for_messages(
        rails,
        [
            {"role": "user", "content": "Use tools"},
            {"role": "tool", "content": "first", "name": "lookup", "tool_call_id": "call-1"},
            {"role": "tool", "content": "second", "tool_call_id": "call-2"},
        ],
        state=None,
    )

    assert events[-1] == {
        "type": "UserToolMessages",
        "tool_messages": [
            {"content": "first", "name": "lookup", "tool_call_id": "call-1"},
            {"content": "second", "name": "unknown", "tool_call_id": "call-2"},
        ],
    }


def test_events_for_messages_v2_converts_supported_messages():
    rails = make_rails(RailsConfig(models=[], colang_version="2.x"))
    passthrough_event = {"type": "UserSilent"}

    events = events_for_messages(
        rails,
        [
            {"role": "context", "content": {"generation_options": {}}},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "event", "event": passthrough_event},
        ],
        state=SimpleNamespace(actions={}),
    )

    assert events == [
        {"type": "ContextUpdate", "data": {"generation_options": {}}},
        {"type": "SystemMessage", "content": "Be concise."},
        {"type": "UtteranceUserActionFinished", "final_transcript": "Hi"},
        passthrough_event,
    ]


def test_events_for_messages_v2_rejects_assistant_messages():
    rails = make_rails(RailsConfig(models=[], colang_version="2.x"))

    with pytest.raises(ValueError, match="assistant.*not supported"):
        events_for_messages(
            rails,
            [{"role": "assistant", "content": "Hello"}],
            state=SimpleNamespace(actions={}),
        )


def test_events_for_messages_v2_converts_tool_message_to_action_finished_event():
    rails = make_rails(RailsConfig(models=[], colang_version="2.x"))
    state = SimpleNamespace(actions={"call-1": SimpleNamespace(name="LookupAction")})

    events = events_for_messages(
        rails,
        [{"role": "tool", "content": "tool result", "tool_call_id": "call-1"}],
        state=state,
    )

    assert len(events) == 1
    assert events[0]["type"] == "LookupActionFinished"
    assert events[0]["action_uid"] == "call-1"
    assert events[0]["action_name"] == "LookupAction"
    assert events[0]["status"] == "success"
    assert events[0]["is_success"] is True
    assert events[0]["return_value"] == "tool result"
    assert events[0]["events"] == []


def test_events_for_messages_accepts_llmrails_instance():
    rails = LLMRails(RailsConfig(models=[], colang_version="1.0"))

    events = events_for_messages(
        rails,
        [{"role": "user", "content": "Hi"}],
        state=None,
    )

    assert events == [{"type": "UtteranceUserActionFinished", "final_transcript": "Hi"}]
