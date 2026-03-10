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

import pytest

from nemoguardrails.actions.core import create_event


@pytest.mark.asyncio
async def test_create_event_empty_string_value():
    """Empty string values must not raise an IndexError (issue #1700)."""
    result = await create_event(event={"_type": "SomeEvent", "param": ""})
    assert len(result.events) == 1
    assert result.events[0]["param"] == ""


@pytest.mark.asyncio
async def test_create_event_variable_reference():
    """A '$'-prefixed value should be resolved from context."""
    result = await create_event(
        event={"_type": "UserMessage", "text": "$user_message"},
        context={"user_message": "Hello!"},
    )
    assert result.events[0]["text"] == "Hello!"


@pytest.mark.asyncio
async def test_create_event_variable_reference_missing_context():
    """A '$'-prefixed value with no matching context key resolves to None."""
    result = await create_event(
        event={"_type": "UserMessage", "text": "$missing_key"},
        context={"other_key": "value"},
    )
    assert result.events[0]["text"] is None


@pytest.mark.asyncio
async def test_create_event_variable_reference_no_context():
    """A '$'-prefixed value with context=None resolves to None."""
    result = await create_event(
        event={"_type": "UserMessage", "text": "$user_message"},
    )
    assert result.events[0]["text"] is None


@pytest.mark.asyncio
async def test_create_event_plain_string_value():
    """Regular strings without '$' prefix should pass through unchanged."""
    result = await create_event(event={"_type": "SomeEvent", "param": "hello world"})
    assert result.events[0]["param"] == "hello world"


@pytest.mark.asyncio
async def test_create_event_non_string_values():
    """Non-string values (int, list, dict) should pass through unchanged."""
    result = await create_event(
        event={"_type": "SomeEvent", "count": 42, "tags": ["a", "b"], "meta": {"k": "v"}},
    )
    assert result.events[0]["count"] == 42
    assert result.events[0]["tags"] == ["a", "b"]
    assert result.events[0]["meta"] == {"k": "v"}
