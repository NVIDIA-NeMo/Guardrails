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

"""Shared tool normalization preserves the existing Chat Completions contracts."""

from typing import Any

import pytest

from nemoguardrails.guardrails.tool_parsing import (
    extract_latest_tool_calls,
    extract_tool_exchanges,
    parse_tools,
)
from nemoguardrails.guardrails.tool_schema import ToolCallExtractionError, Toolset
from nemoguardrails.rails.llm.options import ToolViolationType
from nemoguardrails.types import ToolCall
from tests.guardrails.tool_helpers import (
    assistant_tool_calls,
    make_tool_conversation,
    malformed_prior_tool_call_messages,
    multi_turn_reused_call_id_messages,
    wire_tool_call,
)

_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        },
    },
    {"type": "function", "function": {"name": "noargs"}},
]


class TestParseTools:
    def test_parses_openai_chat_completions_tools(self):
        toolset = parse_tools({"tools": _OPENAI_TOOLS})

        assert isinstance(toolset, Toolset)
        assert sorted(t.key for t in toolset.tools) == ["get_weather", "noargs"]

        weather = toolset.get("get_weather")
        assert weather is not None
        assert weather.name == "get_weather"
        assert weather.type == "function"
        assert weather.description == "Get the weather for a city."
        assert weather.arguments_schema == _OPENAI_TOOLS[0]["function"]["parameters"]
        assert weather.strict is True

    def test_tool_without_parameters_has_no_arguments_schema(self):
        toolset = parse_tools({"tools": _OPENAI_TOOLS})
        noargs = toolset.get("noargs")
        assert noargs is not None
        assert noargs.arguments_schema is None

    def test_no_tools_returns_empty_toolset(self):
        assert parse_tools({}).tools == ()
        assert parse_tools(None).tools == ()
        assert parse_tools({"tools": []}).tools == ()

    def test_malformed_entries_are_skipped(self):
        """Non-dict / function-less entries are dropped so a malformed tool fails closed."""
        tools = [
            "garbage",
            {"type": "function"},
            {"type": "function", "function": {"name": "ok"}},
        ]
        toolset = parse_tools({"tools": tools})
        assert [t.key for t in toolset.tools] == ["ok"]

    def test_function_entries_without_name_are_skipped(self):
        """Name-less function entries are dropped, so duplicate empty keys never crash parse_tools."""
        tools = [
            {"type": "function", "function": {"description": "no name"}},
            {"type": "function", "function": {"name": ""}},
            {"type": "function", "function": {"name": "ok"}},
        ]
        toolset = parse_tools({"tools": tools})
        assert [t.key for t in toolset.tools] == ["ok"]


_TOOL_MESSAGES = make_tool_conversation()


class TestExtractToolExchanges:
    @staticmethod
    def _ids(exchanges):
        """Reduce exchanges to ``[(call_ids, result_call_ids), ...]`` for terse assertions."""
        return [([c.id for c in calls], [r.call_id for r in results]) for calls, results in exchanges]

    def test_groups_single_turn(self):
        exchanges = extract_tool_exchanges(_TOOL_MESSAGES)

        assert len(exchanges) == 1
        calls, results = exchanges[0]
        assert calls[0].id == "call_1"
        assert calls[0].function.name == "get_weather"
        # JSON-string arguments on the wire are normalized to a dict.
        assert calls[0].function.arguments == {"city": "Paris"}
        [result] = results
        assert (result.call_id, result.name, result.content, result.is_error) == ("call_1", "get_weather", "18C", False)

    def test_each_turn_is_its_own_exchange(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "a", "content": "r1"},
            {"role": "assistant", "content": "interim text"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c2", "name": "b", "content": "r2"},
        ]
        assert self._ids(extract_tool_exchanges(messages)) == [(["c1"], ["c1"]), (["c2"], ["c2"])]

    def test_recycled_ids_stay_in_separate_exchanges(self):
        # The core of #13: the same call_id reused across turns is NOT collapsed; each
        # turn is its own exchange so linkage stays turn-local.
        assert self._ids(extract_tool_exchanges(multi_turn_reused_call_id_messages())) == [
            (["call_0"], ["call_0"]),
            (["call_0"], ["call_0"]),
        ]

    def test_parallel_calls_in_one_turn_share_an_exchange(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "a", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "name": "b", "content": "r2"},
        ]
        assert self._ids(extract_tool_exchanges(messages)) == [(["c1", "c2"], ["c1", "c2"])]

    def test_orphan_result_has_no_calls(self):
        # A tool result with no preceding assistant tool-call turn becomes an exchange
        # with no calls, so the rail still flags it as an orphan.
        exchanges = extract_tool_exchanges([{"role": "tool", "tool_call_id": "x", "content": "y"}])
        assert self._ids(exchanges) == [([], ["x"])]

    def test_no_tool_data_returns_empty(self):
        assert extract_tool_exchanges([]) == []
        assert extract_tool_exchanges([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]) == []

    def test_skips_non_dict_messages(self):
        assert self._ids(extract_tool_exchanges(["garbage", *_TOOL_MESSAGES])) == [(["call_1"], ["call_1"])]

    def test_records_each_results_message_index(self):
        """Each result records its message's position in the conversation, counting non-dict and context messages."""
        messages = ["garbage", {"role": "context", "content": {"user_id": "u1"}}, *multi_turn_reused_call_id_messages()]
        exchanges = extract_tool_exchanges(messages)
        assert [result.message_index for _calls, results in exchanges for result in results] == [4, 8]

    def test_malformed_arguments_degrade_to_empty_for_linkage(self):
        """A historical call with malformed argument JSON degrades to empty arguments
        (id/name preserved) instead of aborting extraction."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "not json"}}],
            },
        ]
        exchanges = extract_tool_exchanges(messages)
        assert len(exchanges) == 1
        calls, _results = exchanges[0]
        assert calls[0].id == "c1"
        assert calls[0].function.name == "a"
        assert calls[0].function.arguments == {}


_SECRET_VALUE = "SECRET-VALUE"


def _with_second_call_arguments(arguments: Any) -> dict:
    """An assistant turn whose second call, ``call_2``, carries *arguments*."""
    return assistant_tool_calls(wire_tool_call(), wire_tool_call(call_id="call_2", arguments=arguments))


_LEGACY_CALL = {"name": "get_weather", "arguments": "{}"}
_FLAT_CALL = {"id": "call_1", "type": "function", "name": "run_sql", "arguments": '{"q": "SECRET-VALUE"}'}

# Case id -> (a last assistant turn with one call the extractor refuses, (violation type, index, tool_call_id, tool_name)).
_UNPARSABLE_CALLS = {
    "invalid_json_arguments": (
        _with_second_call_arguments('{"city": "SECRET-VALUE"'),
        ("malformed_arguments", 1, "call_2", "get_weather"),
    ),
    "invalid_json_before_a_valid_call": (
        assistant_tool_calls(wire_tool_call(arguments='{"city": "SECRET-VALUE"'), wire_tool_call(call_id="call_2")),
        ("malformed_arguments", 0, "call_1", "get_weather"),
    ),
    "json_array_arguments": (
        _with_second_call_arguments('["SECRET-VALUE"]'),
        ("malformed_arguments", 1, "call_2", "get_weather"),
    ),
    "non_dict_arguments": (
        _with_second_call_arguments([_SECRET_VALUE]),
        ("malformed_arguments", 1, "call_2", "get_weather"),
    ),
    # A completed turn carries "{}" for a call without arguments; "" is not JSON, as in non-streaming generation.
    "empty_string_arguments": (_with_second_call_arguments(""), ("malformed_arguments", 1, "call_2", "get_weather")),
    "non_dict_entry": (assistant_tool_calls(wire_tool_call(), _SECRET_VALUE), ("malformed_tool_call", 1, None, None)),
    "non_dict_function": (
        assistant_tool_calls({**wire_tool_call(), "function": _SECRET_VALUE}),
        ("malformed_tool_call", 0, "call_1", None),
    ),
    "no_function_object": (assistant_tool_calls(_FLAT_CALL), ("malformed_tool_call", 0, "call_1", None)),
    "non_string_function_name": (
        assistant_tool_calls({**wire_tool_call(), "function": {"name": {"x": 1}, "arguments": "{}"}}),
        ("malformed_tool_call", 0, "call_1", None),
    ),
    "empty_function_name": (
        assistant_tool_calls({**wire_tool_call(), "function": {"name": "", "arguments": "{}"}}),
        ("malformed_tool_call", 0, "call_1", None),
    ),
    "non_string_type": (
        assistant_tool_calls({**wire_tool_call(), "type": {"x": 1}}),
        ("malformed_tool_call", 0, "call_1", "get_weather"),
    ),
    "non_string_id": (
        assistant_tool_calls({**wire_tool_call(), "id": {"x": 1}}),
        ("malformed_tool_call", 0, None, "get_weather"),
    ),
}

# Case id -> (a last assistant turn carrying a legacy function_call, the tool name its error reports).
_LEGACY_FUNCTION_CALL_TURNS = {
    "legacy_function_call": ({"role": "assistant", "function_call": _LEGACY_CALL}, "get_weather"),
    "legacy_function_call_beside_tool_calls": (
        {**assistant_tool_calls(wire_tool_call()), "function_call": {**_LEGACY_CALL, "name": "delete_files"}},
        "delete_files",
    ),
}


class TestExtractLatestToolCalls:
    def test_parses_the_last_assistant_messages_tool_calls(self):
        """The last assistant message's tool calls are returned with their JSON arguments parsed into a dict."""
        calls = extract_latest_tool_calls(make_tool_conversation())
        assert [(c.id, c.type, c.function.name, c.function.arguments) for c in calls] == [
            ("call_1", "function", "get_weather", {"city": "Paris"})
        ]

    def test_only_the_last_assistant_message_is_parsed(self):
        """Tool calls on an earlier assistant message are neither returned nor parsed, even when malformed."""
        calls = extract_latest_tool_calls(malformed_prior_tool_call_messages())
        assert [(c.id, c.function.arguments) for c in calls] == [("call_1", {"city": "London"})]

    def test_context_message_is_ignored(self):
        """A server ``context`` message, whose role ChatMessage rejects, does not break extraction."""
        messages = [{"role": "context", "content": {"user_id": "u1"}}, *make_tool_conversation()]
        assert [c.id for c in extract_latest_tool_calls(messages)] == ["call_1"]

    def test_last_assistant_message_without_tool_calls_returns_empty(self):
        """A final assistant text turn has no calls to check, even when an earlier turn made some."""
        messages = [*make_tool_conversation(), {"role": "assistant", "content": "It's 18C in Paris."}]
        assert extract_latest_tool_calls(messages) == []

    def test_no_assistant_message_returns_empty(self):
        """A conversation with no assistant message has no tool calls."""
        assert extract_latest_tool_calls([]) == []
        assert extract_latest_tool_calls([{"role": "user", "content": "hi"}]) == []

    def test_missing_arguments_become_an_empty_object(self):
        """A call with no ``arguments`` key parses as ``{}``."""
        messages = [assistant_tool_calls({"id": "call_1", "type": "function", "function": {"name": "list_files"}})]
        assert [c.function.arguments for c in extract_latest_tool_calls(messages)] == [{}]

    def test_call_without_an_id_is_named_by_its_position(self):
        """A malformed call with no id is named in its error by its position in ``tool_calls``."""
        call = {"type": "function", "function": {"name": "get_weather", "arguments": "not json"}}

        [error] = extract_latest_tool_calls([assistant_tool_calls(call)])

        assert str(error) == "tool call at index 0 has malformed arguments"

    def test_long_call_id_is_capped_in_the_message_but_kept_whole(self):
        """A call id, which the model chose, is cut to 64 characters in the message but kept whole as identity."""
        call_id = "i" * 200

        [error] = extract_latest_tool_calls([assistant_tool_calls(wire_tool_call("f", "not json", call_id))])

        assert (str(error), error.tool_call_id) == (f"tool call '{'i' * 64}...' has malformed arguments", call_id)

    @pytest.mark.parametrize(("message", "expected"), list(_UNPARSABLE_CALLS.values()), ids=list(_UNPARSABLE_CALLS))
    def test_unparsable_call_is_returned_as_its_error_in_place(self, message, expected):
        """An unparsable call's error takes its place, naming it without quoting its text; every other call parses."""
        violation_type, index, tool_call_id, tool_name = expected

        extracted = extract_latest_tool_calls([message])

        error = extracted[index]
        assert isinstance(error, ToolCallExtractionError)
        assert (error.violation_type, error.index, error.tool_call_id, error.tool_name) == (
            ToolViolationType(violation_type),
            index,
            tool_call_id,
            tool_name,
        )
        assert _SECRET_VALUE not in str(error)
        assert all(isinstance(entry, ToolCall) for position, entry in enumerate(extracted) if position != index)

    @pytest.mark.parametrize(
        ("message", "tool_name"), list(_LEGACY_FUNCTION_CALL_TURNS.values()), ids=list(_LEGACY_FUNCTION_CALL_TURNS)
    )
    def test_legacy_function_call_raises(self, message, tool_name):
        """A legacy ``function_call`` makes the whole turn unvalidatable, so extraction raises."""
        with pytest.raises(ToolCallExtractionError) as excinfo:
            extract_latest_tool_calls([message])

        error = excinfo.value
        assert (error.violation_type, error.index, error.tool_name) == (
            ToolViolationType.LEGACY_FUNCTION_CALL,
            None,
            tool_name,
        )
