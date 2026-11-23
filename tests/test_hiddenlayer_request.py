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

from nemoguardrails.guardrails.tool_schema import Tool, ToolResult
from nemoguardrails.library.hiddenlayer.request import (
    _REQUEST_PATH,
    _RESPONSE_PATH,
    _SURFACE_PATH,
    _headers,
    _input_check,
    _output_check,
    _tool_call_check,
    _tool_result_check,
)
from nemoguardrails.types import ToolCall, ToolCallFunction


def _tool_call(arguments, call_id="call_abc", name="run_sql") -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))


class TestRoutes:
    """The route carries the direction of the evaluation, so each surface is pinned to one by name."""

    @pytest.mark.parametrize(
        ("surface", "path"),
        [
            ("input", _REQUEST_PATH),
            ("tool_result", _REQUEST_PATH),
            ("output", _RESPONSE_PATH),
            ("tool_call", _RESPONSE_PATH),
        ],
    )
    def test_surface_route(self, surface, path):
        assert _SURFACE_PATH[surface] == path

    def test_the_two_routes_are_the_contract_paths(self):
        assert _REQUEST_PATH == "/detection/v2/nemo-guardrails/request-evaluations"
        assert _RESPONSE_PATH == "/detection/v2/nemo-guardrails/response-evaluations"

    def test_every_surface_has_a_route(self):
        assert set(_SURFACE_PATH) == {"input", "output", "tool_call", "tool_result"}


class TestBodies:
    def test_input_is_the_user_message(self):
        assert _input_check("gpt-5-nano", "Hello") == {
            "check": "input",
            "model": "gpt-5-nano",
            "user_message": "Hello",
        }

    def test_output_is_the_bot_message(self):
        assert _output_check("gpt-5-nano", "Hi there") == {
            "check": "output",
            "model": "gpt-5-nano",
            "bot_message": "Hi there",
        }

    def test_tool_call_carries_the_call_as_nemo_holds_it(self):
        tool_call = _tool_call({"query": "SELECT 1", "limit": 5})
        assert _tool_call_check("gpt-5-nano", tool_call, None) == {
            "check": "tool_call",
            "model": "gpt-5-nano",
            "tool_call": {"id": "call_abc", "name": "run_sql", "arguments": {"query": "SELECT 1", "limit": 5}},
        }

    def test_tool_result_carries_the_result_and_the_call_it_answers(self):
        tool_call = _tool_call({"query": "SELECT 1"})
        tool_result = ToolResult(call_id="call_abc", name="run_sql", content="1 row")
        assert _tool_result_check("gpt-5-nano", tool_call, tool_result, None) == {
            "check": "tool_result",
            "model": "gpt-5-nano",
            "tool_call": {"id": "call_abc", "name": "run_sql", "arguments": {"query": "SELECT 1"}},
            "tool_result": {"content": "1 row", "is_error": False, "call_id": "call_abc", "name": "run_sql"},
        }

    def test_a_failed_tool_result_is_flagged(self):
        tool_result = ToolResult(call_id="call_abc", content="boom", is_error=True)
        body = _tool_result_check("m", _tool_call({}), tool_result, None)
        assert body["tool_result"]["is_error"] is True

    @pytest.mark.parametrize("model", [None, ""])
    def test_no_model_is_sent_without_one(self, model):
        assert "model" not in _input_check(model, "Hello")


class TestToolCallFields:
    @pytest.mark.parametrize(
        "arguments",
        [
            {"nested": {"n": 9007199254740993, "s": "x"}, "list": [1, "y"], "none": None},
            '{"query": "SELECT 1"}',
            "not json",
            {},
        ],
        ids=["dict", "json-string", "unparsed-string", "empty"],
    )
    def test_arguments_are_sent_as_nemo_holds_them(self, arguments):
        body = _tool_call_check("m", _tool_call(arguments), None)
        assert body["tool_call"]["arguments"] == arguments

    @pytest.mark.parametrize(("call_id", "name"), [("", ""), ("call_a", "run_sql")])
    def test_id_and_name_are_sent_as_held_so_the_server_owns_the_fallback(self, call_id, name):
        body = _tool_call_check("m", _tool_call({}, call_id=call_id, name=name), None)
        assert body["tool_call"]["id"] == call_id
        assert body["tool_call"]["name"] == name


class TestToolResultContent:
    @pytest.mark.parametrize(
        "content",
        [
            "plain text",
            None,
            "",
            [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
            [{"type": "image", "source": {"data": "iVBORw0KGgo="}}],
            [],
        ],
        ids=["text", "none", "empty", "parts", "no-text-parts", "empty-list"],
    )
    def test_content_is_sent_as_nemo_holds_it(self, content):
        body = _tool_result_check("m", _tool_call({}), ToolResult(call_id="c", content=content), None)
        assert body["tool_result"]["content"] == content

    @pytest.mark.parametrize("field", ["call_id", "name"])
    def test_an_unset_field_is_left_out_rather_than_sent_as_null(self, field):
        body = _tool_result_check("m", _tool_call({}), ToolResult(content="ok"), None)
        assert field not in body["tool_result"]


class TestToolDefinition:
    def test_the_declared_tool_is_sent_with_either_tool_check(self):
        tool = Tool(name="run_sql", description="Run a query", arguments_schema={"type": "object"})
        expected = {"name": "run_sql", "description": "Run a query", "parameters": {"type": "object"}}

        assert _tool_call_check("m", _tool_call({}), tool)["tool_definition"] == expected
        result_body = _tool_result_check("m", _tool_call({}), ToolResult(content="ok"), tool)
        assert result_body["tool_definition"] == expected

    def test_a_hosted_tool_is_named_by_its_type(self):
        tool = Tool(type="web_search")
        assert _tool_call_check("m", _tool_call({}), tool)["tool_definition"] == {"name": "web_search"}

    def test_omitted_without_a_declaration(self):
        assert "tool_definition" not in _tool_call_check("m", _tool_call({}), None)
        assert "tool_definition" not in _tool_result_check("m", _tool_call({}), ToolResult(content="ok"), None)


class TestHeaders:
    def test_header_set(self):
        assert _headers("tok", "evaluation-id", "hiddenlayer-nemo", "openai", project_id="my-project") == {
            "Authorization": "Bearer tok",
            "Content-Type": "application/json",
            "HL-Roundtrip-Id": "evaluation-id",
            "HL-Requester-Id": "hiddenlayer-nemo",
            "HL-Provider-Id": "openai",
            "HL-Project-Id": "my-project",
        }

    @pytest.mark.parametrize("value", [None, ""])
    def test_optional_headers_omitted_when_unset(self, value):
        headers = _headers("tok", "evaluation-id", "r", value, project_id=value)
        assert "HL-Provider-Id" not in headers
        assert "HL-Project-Id" not in headers

    def test_never_sends_tenant_or_correlation_headers(self):
        names = {name.lower() for name in _headers("tok", "evaluation-id", "r", "p", project_id="p")}
        assert names.isdisjoint({"x-correlation-id", "x-tenant-id", "hl-runtime-edge-provider"})
