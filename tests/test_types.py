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

import pytest

from nemoguardrails.types import (
    ChatMessage,
    LLMFramework,
    LLMModel,
    LLMResponse,
    LLMResponseChunk,
    Role,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)


class TestRole:
    def test_string_equality(self):
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.SYSTEM == "system"
        assert Role.TOOL == "tool"

    def test_string_serialization(self):
        assert str(Role.USER) == "Role.USER"
        assert Role.USER.value == "user"
        assert f"{Role.USER.value}" == "user"

    def test_is_instance_of_str(self):
        assert isinstance(Role.USER, str)


class TestToolCallFunction:
    def test_creation(self):
        fn = ToolCallFunction(name="search", arguments={"query": "hello"})
        assert fn.name == "search"
        assert fn.arguments == {"query": "hello"}


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(
            id="tc_0",
            function=ToolCallFunction(name="search", arguments={"query": "hello"}),
        )
        assert tc.function.name == "search"
        assert tc.function.arguments == {"query": "hello"}
        assert tc.id == "tc_0"
        assert tc.type == "function"

    def test_default_function(self):
        tc = ToolCall(id="tc_1")
        assert tc.function.name == ""
        assert tc.function.arguments == {}
        assert tc.type == "function"

    def test_to_dict(self):
        tc = ToolCall(
            id="tc_0",
            function=ToolCallFunction(name="search", arguments={"q": "hello"}),
        )
        assert tc.to_dict() == {
            "id": "tc_0",
            "type": "function",
            "function": {"name": "search", "arguments": {"q": "hello"}},
        }

    def test_to_dict_default(self):
        tc = ToolCall(id="tc_0")
        assert tc.to_dict() == {
            "id": "tc_0",
            "type": "function",
            "function": {"name": "", "arguments": {}},
        }


class TestChatMessage:
    def test_init(self):
        msg = ChatMessage(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.provider_metadata == {}

    def test_from_user(self):
        msg = ChatMessage.from_user("hi")
        assert msg.role == Role.USER
        assert msg.content == "hi"

    def test_from_assistant(self):
        msg = ChatMessage.from_assistant("hello")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "hello"

    def test_from_system(self):
        msg = ChatMessage.from_system("you are a bot")
        assert msg.role == Role.SYSTEM
        assert msg.content == "you are a bot"

    def test_from_tool(self):
        msg = ChatMessage.from_tool("result", tool_call_id="tc_1")
        assert msg.role == Role.TOOL
        assert msg.content == "result"
        assert msg.tool_call_id == "tc_1"

    def test_content_can_be_none(self):
        msg = ChatMessage(role=Role.ASSISTANT)
        assert msg.content is None

    def test_content_can_be_list(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "http://example.com"}}]
        msg = ChatMessage(role=Role.USER, content=blocks)
        assert msg.content == blocks

    def test_to_dict_minimal(self):
        msg = ChatMessage.from_user("hi")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hi"}

    def test_to_dict_omits_none_content(self):
        msg = ChatMessage(role=Role.ASSISTANT)
        d = msg.to_dict()
        assert "content" not in d

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(
            id="tc_1",
            function=ToolCallFunction(name="search", arguments={"q": "x"}),
        )
        msg = ChatMessage(
            role=Role.ASSISTANT,
            content="found it",
            name="bot",
            tool_calls=[tc],
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "found it"
        assert d["name"] == "bot"
        assert d["tool_calls"] == [
            {"id": "tc_1", "type": "function", "function": {"name": "search", "arguments": {"q": "x"}}}
        ]
        assert "tool_call_id" not in d

    def test_to_dict_preserves_empty_string_fields(self):
        msg = ChatMessage(role=Role.ASSISTANT, content="", tool_call_id="", name="")
        d = msg.to_dict()
        assert d["content"] == ""
        assert d["tool_call_id"] == ""
        assert d["name"] == ""

    def test_to_dict_preserves_empty_tool_calls_list(self):
        msg = ChatMessage(role=Role.ASSISTANT, tool_calls=[])
        d = msg.to_dict()
        assert d["tool_calls"] == []

    def test_to_dict_keeps_arguments_as_dict(self):
        tc = ToolCall(
            id="tc_1",
            function=ToolCallFunction(name="search", arguments={"q": "x"}),
        )
        msg = ChatMessage(role=Role.ASSISTANT, tool_calls=[tc])
        d = msg.to_dict()
        assert d["tool_calls"][0]["function"]["arguments"] == {"q": "x"}

    def test_from_dict_basic(self):
        msg = ChatMessage.from_dict({"role": "user", "content": "hi"})
        assert msg.role == Role.USER
        assert msg.content == "hi"

    def test_from_dict_with_openai_nested_tool_calls(self):
        d = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": {"q": "x"}},
                }
            ],
        }
        msg = ChatMessage.from_dict(d)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "search"
        assert msg.tool_calls[0].function.arguments == {"q": "x"}
        assert msg.tool_calls[0].id == "tc_1"
        assert msg.tool_calls[0].type == "function"

    def test_from_dict_with_json_string_arguments(self):
        d = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "x"}'},
                }
            ],
        }
        msg = ChatMessage.from_dict(d)
        assert msg.tool_calls[0].function.arguments == {"q": "x"}

    def test_from_dict_with_malformed_json_arguments(self):
        d = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "not json"},
                }
            ],
        }
        with pytest.raises(ValueError, match="not valid JSON"):
            ChatMessage.from_dict(d)

    def test_from_dict_with_non_object_json_arguments(self):
        d = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "[]"},
                }
            ],
        }
        with pytest.raises(ValueError, match="must be a JSON object"):
            ChatMessage.from_dict(d)

    def test_from_dict_with_non_dict_raw_arguments(self):
        d = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": [1, 2, 3]},
                }
            ],
        }
        with pytest.raises(ValueError, match="must be a dict"):
            ChatMessage.from_dict(d)

    def test_from_dict_with_legacy_flat_tool_calls(self):
        d = {
            "role": "assistant",
            "tool_calls": [{"name": "search", "args": {"q": "x"}, "id": "tc_1"}],
        }
        msg = ChatMessage.from_dict(d)
        assert msg.tool_calls[0].function.name == "search"
        assert msg.tool_calls[0].function.arguments == {"q": "x"}
        assert msg.tool_calls[0].id == "tc_1"

    def test_from_dict_captures_provider_metadata(self):
        d = {"role": "user", "content": "hi", "provider_metadata": {"custom_field": "value", "model": "gpt-4"}}
        msg = ChatMessage.from_dict(d)
        assert msg.provider_metadata == {"custom_field": "value", "model": "gpt-4"}

    def test_from_dict_unknown_keys_captured_into_provider_metadata(self):
        d = {"role": "user", "content": "hi", "unexpected_key": "v"}
        msg = ChatMessage.from_dict(d)
        assert msg.provider_metadata["unexpected_key"] == "v"

    def test_from_dict_explicit_provider_metadata_overrides_extra_keys(self):
        d = {"role": "user", "content": "hi", "logprobs": 10, "provider_metadata": {"logprobs": 5}}
        msg = ChatMessage.from_dict(d)
        assert msg.provider_metadata["logprobs"] == 5

    def test_from_dict_missing_provider_metadata_defaults_to_empty(self):
        msg = ChatMessage.from_dict({"role": "user", "content": "hi"})
        assert msg.provider_metadata == {}

    def test_to_dict_includes_provider_metadata(self):
        msg = ChatMessage(
            role=Role.USER,
            content="hi",
            provider_metadata={"custom_field": "value", "model": "gpt-4"},
        )
        d = msg.to_dict()
        assert d["provider_metadata"] == {"custom_field": "value", "model": "gpt-4"}

    def test_to_dict_omits_empty_provider_metadata(self):
        msg = ChatMessage.from_user("hi")
        d = msg.to_dict()
        assert "provider_metadata" not in d

    def test_roundtrip(self):
        tc = ToolCall(
            id="tc_2",
            function=ToolCallFunction(name="fn", arguments={"a": 1}),
        )
        original = ChatMessage(
            role=Role.ASSISTANT,
            content="ok",
            name="bot",
            tool_calls=[tc],
            tool_call_id="tc_0",
            provider_metadata={"custom": "data"},
        )
        rebuilt = ChatMessage.from_dict(original.to_dict())
        assert rebuilt.role == original.role
        assert rebuilt.content == original.content
        assert rebuilt.name == original.name
        assert rebuilt.tool_call_id == original.tool_call_id
        assert rebuilt.provider_metadata == original.provider_metadata
        assert len(rebuilt.tool_calls) == 1
        assert rebuilt.tool_calls[0].function.name == "fn"
        assert rebuilt.tool_calls[0].function.arguments == {"a": 1}

    def test_from_dict_with_empty_function_dict(self):
        d = {
            "role": "assistant",
            "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {}},
            ],
        }
        msg = ChatMessage.from_dict(d)
        assert msg.tool_calls[0].function.name == ""
        assert msg.tool_calls[0].function.arguments == {}

    @pytest.mark.parametrize(
        "alias,expected_role",
        [
            ("bot", Role.ASSISTANT),
            ("assistant", Role.ASSISTANT),
            ("human", Role.USER),
            ("user", Role.USER),
            ("developer", Role.SYSTEM),
            ("system", Role.SYSTEM),
            ("tool", Role.TOOL),
        ],
    )
    def test_from_dict_role_aliases(self, alias, expected_role):
        msg = ChatMessage.from_dict({"role": alias, "content": "test"})
        assert msg.role == expected_role

    def test_from_dict_raises_when_role_missing(self):
        with pytest.raises(ValueError, match="Missing required key"):
            ChatMessage.from_dict({"content": "hi"})

    def test_from_dict_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown role"):
            ChatMessage.from_dict({"role": "alien", "content": "hi"})


class TestLLMResponse:
    def test_creation_minimal(self):
        resp = LLMResponse(content="hello")
        assert resp.content == "hello"
        assert resp.tool_calls is None
        assert resp.reasoning is None
        assert resp.model is None
        assert resp.finish_reason is None
        assert resp.usage is None
        assert resp.provider_metadata is None

    def test_creation_full(self):
        tc = ToolCall(id="tc_1", function=ToolCallFunction(name="fn", arguments={}))
        resp = LLMResponse(
            content="ok",
            reasoning="thinking...",
            tool_calls=[tc],
            model="gpt-4",
            finish_reason="stop",
            stop_sequence="\n",
            request_id="req-123",
            usage=UsageInfo(total_tokens=100, input_tokens=50, output_tokens=50),
            provider_metadata={"key": "val"},
        )
        assert resp.tool_calls[0].function.name == "fn"
        assert resp.usage.total_tokens == 100
        assert resp.reasoning == "thinking..."
        assert resp.model == "gpt-4"
        assert resp.stop_sequence == "\n"
        assert resp.request_id == "req-123"


class TestLLMResponseChunk:
    def test_creation_minimal(self):
        chunk = LLMResponseChunk(delta_content="tok")
        assert chunk.delta_content == "tok"
        assert chunk.usage is None
        assert chunk.provider_metadata is None

    def test_creation_full(self):
        chunk = LLMResponseChunk(
            delta_content="tok",
            finish_reason="stop",
            usage=UsageInfo(total_tokens=5, input_tokens=3, output_tokens=2),
            provider_metadata={"finish_reason": "stop"},
        )
        assert chunk.finish_reason == "stop"
        assert chunk.usage.total_tokens == 5


class TestLLMModelProtocol:
    def test_mock_satisfies_protocol(self):
        class MockLLM:
            async def generate_async(self, prompt, *, stop=None, **kwargs):
                return LLMResponse(content="response")

            async def stream_async(self, prompt, *, stop=None, **kwargs):
                yield LLMResponseChunk(delta_content="chunk")

            @property
            def model_name(self):
                return "mock"

            @property
            def provider_name(self):
                return None

            @property
            def provider_url(self):
                return None

        assert isinstance(MockLLM(), LLMModel)

    def test_incomplete_class_fails_protocol(self):
        class IncompleteLLM:
            async def generate_async(self, prompt, *, stop=None, **kwargs):
                return LLMResponse(content="response")

        assert not isinstance(IncompleteLLM(), LLMModel)


class TestUsageInfo:
    def test_defaults(self):
        usage = UsageInfo()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.reasoning_tokens is None
        assert usage.cached_tokens is None

    def test_creation(self):
        usage = UsageInfo(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            reasoning_tokens=5,
            cached_tokens=3,
        )
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.total_tokens == 30
        assert usage.reasoning_tokens == 5
        assert usage.cached_tokens == 3


class TestLLMFrameworkProtocol:
    def test_mock_satisfies_protocol(self):
        class MockFramework:
            def create_model(self, model_name, provider_name, model_kwargs=None):
                return None

            def register_provider(self, name, provider_cls):
                pass

            def get_provider_names(self):
                return []

            async def reset(self):
                return

        assert isinstance(MockFramework(), LLMFramework)

    def test_incomplete_class_fails_protocol(self):
        class IncompleteFramework:
            pass

        assert not isinstance(IncompleteFramework(), LLMFramework)
