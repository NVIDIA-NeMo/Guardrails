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

"""Shared fixtures and assertions for the IORails tool-rail tests.

Lives alongside ``async_helpers.py`` / ``metric_helpers.py`` and is not collected
by pytest (no ``test_`` prefix). Holds the small tool shapes and the
blocked-result assertion reused across the tool-rail test modules. Assertions
carry explicit messages since this module is not assertion-rewritten by pytest.
"""

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def assert_blocked(result, *substrings: str) -> None:
    """Assert *result* blocked the request and its reason contains each substring."""
    assert result.is_safe is False, f"expected blocked, got {result!r}"
    assert result.reason is not None, f"expected a block reason, got {result!r}"
    for substring in substrings:
        assert substring in result.reason, f"{substring!r} not in reason {result.reason!r}"


def make_tool_conversation(result_call_id: str = "call_1") -> list:
    """A user turn, an assistant ``get_weather`` tool call (id ``call_1``), then a tool result.

    ``result_call_id`` sets the tool result's ``tool_call_id`` so callers can test
    linked (``call_1``) and unlinked (anything else) results.
    """
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": result_call_id, "name": "get_weather", "content": "18C"},
    ]
