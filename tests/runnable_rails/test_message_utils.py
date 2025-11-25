#!/usr/bin/env python3

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

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from nemoguardrails.integrations.langchain.message_utils import (
    all_base_messages,
    create_ai_message,
    create_ai_message_chunk,
    create_human_message,
    create_system_message,
    create_tool_message,
    dict_to_message,
    dicts_to_messages,
    get_message_class,
    get_message_role,
    is_ai_message,
    is_base_message,
    is_human_message,
    is_message_type,
    is_system_message,
    is_tool_message,
    message_to_dict,
    messages_to_dicts,
)


class TestMessageRoleAndClass:
    def test_get_message_role_ai(self):
        msg = AIMessage(content="test")
        assert get_message_role(msg) == "assistant"

    def test_get_message_role_human(self):
        msg = HumanMessage(content="test")
        assert get_message_role(msg) == "user"

    def test_get_message_role_system(self):
        msg = SystemMessage(content="test")
        assert get_message_role(msg) == "system"

    def test_get_message_role_tool(self):
        msg = ToolMessage(content="test", tool_call_id="123")
        assert get_message_role(msg) == "tool"

    def test_get_message_class_user(self):
        assert get_message_class("user") == HumanMessage

    def test_get_message_class_assistant(self):
        assert get_message_class("assistant") == AIMessage

    def test_get_message_class_bot(self):
        assert get_message_class("bot") == AIMessage

    def test_get_message_class_system(self):
        assert get_message_class("system") == SystemMessage

    def test_get_message_class_developer(self):
        assert get_message_class("developer") == SystemMessage

    def test_get_message_class_tool(self):
        assert get_message_class("tool") == ToolMessage

    def test_get_message_class_unknown(self):
        with pytest.raises(ValueError, match="Unknown message type"):
            get_message_class("unknown")


class TestMessageConversion:
    def test_ai_message_with_tool_calls(self):
        original = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "test"},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ]
            },
            response_metadata={"model": "gpt-4"},
            id="msg-001",
        )

        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, AIMessage)
        assert recreated.content == original.content
        assert recreated.tool_calls == original.tool_calls
        assert recreated.additional_kwargs == original.additional_kwargs
        assert recreated.response_metadata == original.response_metadata
        assert recreated.id == original.id

    def test_ai_message_with_invalid_tool_calls(self):
        original = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "malformed_tool",
                    "args": "invalid json string",
                    "id": "call_invalid",
                    "error": "Invalid JSON in arguments",
                }
            ],
            id="msg-invalid",
        )

        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, AIMessage)
        assert recreated.content == original.content
        assert recreated.invalid_tool_calls == original.invalid_tool_calls
        assert recreated.id == original.id

    def test_tool_message(self):
        original = ToolMessage(
            content="Result data",
            tool_call_id="call_123",
            name="search",
            additional_kwargs={"extra": "data"},
            id="tool-msg-001",
        )

        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, ToolMessage)
        assert recreated.content == original.content
        assert recreated.tool_call_id == original.tool_call_id
        assert recreated.name == original.name
        assert recreated.additional_kwargs == original.additional_kwargs
        assert recreated.id == original.id

    def test_human_message_basic(self):
        original = HumanMessage(content="Hello", id="human-1")
        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, HumanMessage)
        assert recreated.content == original.content
        assert recreated.id == original.id

    def test_system_message_basic(self):
        original = SystemMessage(content="System prompt", id="sys-1")
        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, SystemMessage)
        assert recreated.content == original.content
        assert recreated.id == original.id

    def test_developer_role_conversion(self):
        msg_dict = {"role": "developer", "content": "Developer instructions"}
        msg = dict_to_message(msg_dict)
        assert isinstance(msg, SystemMessage)
        assert msg.content == "Developer instructions"

    def test_empty_collections_now_included(self):
        msg = AIMessage(content="Test", additional_kwargs={}, tool_calls=[])
        msg_dict = message_to_dict(msg)

        assert "additional_kwargs" in msg_dict
        assert "tool_calls" in msg_dict
        assert msg_dict["additional_kwargs"] == {}
        assert msg_dict["tool_calls"] == []

    def test_message_to_dict_preserves_role(self):
        human_msg = HumanMessage(content="test")
        ai_msg = AIMessage(content="test")
        system_msg = SystemMessage(content="test")

        assert message_to_dict(human_msg)["role"] == "user"
        assert message_to_dict(ai_msg)["role"] == "assistant"
        assert message_to_dict(system_msg)["role"] == "system"


class TestBatchConversion:
    def test_messages_to_dicts(self):
        originals = [
            HumanMessage(content="Hello", id="human-1"),
            AIMessage(
                content="Hi there",
                tool_calls=[{"name": "tool", "args": {}, "id": "c1", "type": "tool_call"}],
                id="ai-1",
            ),
            ToolMessage(content="Tool result", tool_call_id="c1", name="tool"),
            SystemMessage(content="System prompt", id="sys-1"),
        ]

        dicts = messages_to_dicts(originals)

        assert len(dicts) == len(originals)
        assert dicts[0]["role"] == "user"
        assert dicts[1]["role"] == "assistant"
        assert dicts[2]["role"] == "tool"
        assert dicts[3]["role"] == "system"

    def test_dicts_to_messages(self):
        msg_dicts = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "Result", "tool_call_id": "123"},
            {"role": "system", "content": "System"},
        ]

        messages = dicts_to_messages(msg_dicts)

        assert len(messages) == len(msg_dicts)
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], ToolMessage)
        assert isinstance(messages[3], SystemMessage)

    def test_round_trip_conversion(self):
        originals = [
            HumanMessage(content="Test 1", id="h1", name="user1"),
            AIMessage(
                content="Test 2",
                id="a1",
                tool_calls=[{"name": "func", "args": {"x": 1}, "id": "tc1", "type": "tool_call"}],
            ),
            SystemMessage(content="Test 3", id="s1"),
            ToolMessage(content="Test 4", tool_call_id="tc1", name="func", id="t1"),
        ]

        dicts = messages_to_dicts(originals)
        recreated = dicts_to_messages(dicts)

        for orig, recr in zip(originals, recreated):
            assert type(orig) is type(recr)
            assert orig.content == recr.content
            if hasattr(orig, "id") and orig.id:
                assert orig.id == recr.id
            if hasattr(orig, "name") and orig.name:
                assert orig.name == recr.name


class TestTypeChecking:
    def test_is_message_type(self):
        ai_msg = AIMessage(content="test")
        human_msg = HumanMessage(content="test")

        assert is_message_type(ai_msg, AIMessage)
        assert not is_message_type(ai_msg, HumanMessage)
        assert is_message_type(human_msg, HumanMessage)
        assert not is_message_type(human_msg, AIMessage)

    def test_is_base_message(self):
        assert is_base_message(AIMessage(content="test"))
        assert is_base_message(HumanMessage(content="test"))
        assert is_base_message(SystemMessage(content="test"))
        assert is_base_message(ToolMessage(content="test", tool_call_id="123"))
        assert not is_base_message("not a message")
        assert not is_base_message({"role": "user", "content": "test"})

    def test_is_ai_message(self):
        ai_msg = AIMessage(content="test")
        assert is_ai_message(ai_msg)

        assert not is_ai_message(HumanMessage(content="test"))
        assert not is_ai_message(SystemMessage(content="test"))
        assert not is_ai_message(ToolMessage(content="test", tool_call_id="123"))
        assert not is_ai_message("not a message")

    def test_is_human_message(self):
        human_msg = HumanMessage(content="test")
        assert is_human_message(human_msg)

        assert not is_human_message(AIMessage(content="test"))
        assert not is_human_message(SystemMessage(content="test"))
        assert not is_human_message(ToolMessage(content="test", tool_call_id="123"))
        assert not is_human_message("not a message")

    def test_is_system_message(self):
        assert is_system_message(SystemMessage(content="test"))
        assert not is_system_message(AIMessage(content="test"))

    def test_is_tool_message(self):
        assert is_tool_message(ToolMessage(content="test", tool_call_id="123"))
        assert not is_tool_message(AIMessage(content="test"))

    def test_all_base_messages(self):
        messages = [
            AIMessage(content="1"),
            HumanMessage(content="2"),
            SystemMessage(content="3"),
        ]
        assert all_base_messages(messages)

        mixed = [AIMessage(content="1"), "not a message"]
        assert not all_base_messages(mixed)

        assert all_base_messages([])


class TestMessageCreation:
    def test_create_ai_message_basic(self):
        msg = create_ai_message("test content")
        assert msg.content == "test content"
        assert isinstance(msg, AIMessage)

    def test_create_ai_message_with_tool_calls(self):
        tool_calls = [{"name": "func", "args": {}, "id": "123", "type": "tool_call"}]
        usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 50,
            "total_tokens": 100,
        }
        msg = create_ai_message(
            "content",
            tool_calls=tool_calls,
            additional_kwargs={"key": "value"},
            response_metadata={"model": "gpt-4"},
            id="msg-1",
            usage_metadata=usage_metadata,
        )

        assert msg.content == "content"
        assert msg.tool_calls == tool_calls
        assert msg.additional_kwargs == {"key": "value"}
        assert msg.response_metadata == {"model": "gpt-4"}
        assert msg.id == "msg-1"
        assert msg.usage_metadata == usage_metadata

    def test_create_ai_message_chunk(self):
        chunk = create_ai_message_chunk("chunk content", id="chunk-1")
        assert chunk.content == "chunk content"
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.id == "chunk-1"

    def test_create_human_message(self):
        msg = create_human_message(
            "user input",
            additional_kwargs={"meta": "data"},
            response_metadata={"source": "user"},
            id="human-1",
            name="user1",
        )

        assert msg.content == "user input"
        assert msg.additional_kwargs == {"meta": "data"}
        assert msg.response_metadata == {"source": "user"}
        assert msg.id == "human-1"
        assert msg.name == "user1"

    def test_create_system_message(self):
        msg = create_system_message(
            "system prompt",
            additional_kwargs={"sys": "info"},
            response_metadata={"type": "system"},
            id="sys-1",
            name="system",
        )

        assert msg.content == "system prompt"
        assert msg.additional_kwargs == {"sys": "info"}
        assert msg.response_metadata == {"type": "system"}
        assert msg.id == "sys-1"
        assert msg.name == "system"

    def test_create_tool_message(self):
        msg = create_tool_message(
            "tool result",
            tool_call_id="call-123",
            name="calculator",
            additional_kwargs={"result": "success"},
            response_metadata={"tool": "calc"},
            id="tool-1",
            status="success",
        )

        assert msg.content == "tool result"
        assert msg.tool_call_id == "call-123"
        assert msg.name == "calculator"
        assert msg.additional_kwargs == {"result": "success"}
        assert msg.response_metadata == {"tool": "calc"}
        assert msg.id == "tool-1"
        assert msg.status == "success"


class TestEdgeCases:
    def test_falsey_values_preservation(self):
        original = AIMessage(
            content="Test",
            additional_kwargs={},
            tool_calls=[],
            name="",
            response_metadata={},
            id="test-id",
        )
        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert recreated.additional_kwargs == {}
        assert recreated.tool_calls == []
        assert recreated.name == ""
        assert recreated.response_metadata == {}
        assert recreated.id == "test-id"

    def test_human_message_with_empty_name(self):
        original = HumanMessage(content="Hello", name="")
        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, HumanMessage)
        assert recreated.name == ""

    def test_system_message_with_empty_additional_kwargs(self):
        original = SystemMessage(content="System prompt", additional_kwargs={})
        msg_dict = message_to_dict(original)
        recreated = dict_to_message(msg_dict)

        assert isinstance(recreated, SystemMessage)
        assert recreated.additional_kwargs == {}

    def test_dict_to_message_missing_role_and_type(self):
        with pytest.raises(ValueError, match="must have 'type' or 'role'"):
            dict_to_message({"content": "test"})

    def test_dict_to_message_with_type_field(self):
        msg = dict_to_message({"type": "user", "content": "test"})
        assert isinstance(msg, HumanMessage)

    def test_dict_to_message_with_role_field(self):
        msg = dict_to_message({"role": "user", "content": "test"})
        assert isinstance(msg, HumanMessage)

    def test_tool_message_without_tool_call_id(self):
        msg_dict = {"role": "tool", "content": "test"}
        msg = dict_to_message(msg_dict)
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == ""

    def test_message_with_none_values(self):
        original = AIMessage(
            content="test",
            additional_kwargs={"valid": "value"},
        )
        msg_dict = message_to_dict(original)

        assert msg_dict["content"] == "test"
        assert msg_dict["role"] == "assistant"
        assert "additional_kwargs" in msg_dict
        assert msg_dict["additional_kwargs"] == {"valid": "value"}

    def test_preserves_unknown_fields_in_dict(self):
        msg_dict = {
            "role": "assistant",
            "content": "test",
            "id": "123",
            "name": "bot",
        }
        msg = dict_to_message(msg_dict)

        assert isinstance(msg, AIMessage)
        assert msg.content == "test"
        assert msg.id == "123"
        assert msg.name == "bot"
