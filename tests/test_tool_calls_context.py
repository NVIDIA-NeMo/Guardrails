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


from nemoguardrails.context import tool_calls_var


def test_tool_calls_var_default():
    """Test that tool_calls_var defaults to None."""
    assert tool_calls_var.get() is None


def test_tool_calls_var_set_and_get():
    """Test setting and getting tool calls from context."""
    test_tool_calls = [
        {
            "name": "get_weather",
            "args": {"location": "New York"},
            "id": "call_123",
            "type": "tool_call",
        },
        {
            "name": "calculate",
            "args": {"expression": "2+2"},
            "id": "call_456",
            "type": "tool_call",
        },
    ]

    tool_calls_var.set(test_tool_calls)

    result = tool_calls_var.get()
    assert result == test_tool_calls
    assert len(result) == 2
    assert result[0]["id"] == "call_123"
    assert result[1]["name"] == "calculate"


def test_tool_calls_var_clear():
    """Test clearing tool calls from context."""
    test_tool_calls = [{"name": "test", "args": {}, "id": "call_test", "type": "tool_call"}]

    tool_calls_var.set(test_tool_calls)
    assert tool_calls_var.get() == test_tool_calls

    tool_calls_var.set(None)
    assert tool_calls_var.get() is None


def test_tool_calls_var_empty_list():
    """Test setting empty list of tool calls."""
    tool_calls_var.set([])

    result = tool_calls_var.get()
    assert result == []
    assert len(result) == 0
