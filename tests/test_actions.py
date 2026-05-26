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


import asyncio

from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.core import create_event


def test_action_decorator():
    @action(is_system_action=True, name="test_action", execute_async=True)
    def sample_action():
        return "test"

    assert hasattr(sample_action, "action_meta")
    assert sample_action.action_meta["name"] == "test_action"
    assert sample_action.action_meta["is_system_action"] is True
    assert sample_action.action_meta["execute_async"] is True


def test_action_decorator_with_output_mapping():
    def sample_output_mapping(result):
        return result == "blocked"

    @action(output_mapping=sample_output_mapping)
    def sample_action():
        return "blocked"

    assert hasattr(sample_action, "action_meta")
    assert sample_action.action_meta["output_mapping"] is not None
    assert sample_action.action_meta["output_mapping"]("blocked") is True
    assert sample_action.action_meta["output_mapping"]("not_blocked") is False


def test_create_event_accepts_empty_string_values():
    # #1700: the `$variable` reference check used `v[0] == "$"`, which raised
    # IndexError on empty string values. Using `startswith("$")` handles them.
    result = asyncio.run(
        create_event(event={"_type": "SomeEvent", "param": ""})
    )
    assert len(result.events) == 1
    assert result.events[0]["type"] == "SomeEvent"
    assert result.events[0]["param"] == ""


def test_create_event_still_resolves_dollar_prefixed_references():
    result = asyncio.run(
        create_event(
            event={"_type": "SomeEvent", "param": "$user_name"},
            context={"user_name": "John"},
        )
    )
    assert len(result.events) == 1
    assert result.events[0]["type"] == "SomeEvent"
    assert result.events[0]["param"] == "John"


def test_action_result():
    result = ActionResult(
        return_value="test_value",
        events=[{"event": "test_event"}],
        context_updates={"key": "value"},
    )

    assert result.return_value == "test_value"
    assert result.events == [{"event": "test_event"}]
    assert result.context_updates == {"key": "value"}
