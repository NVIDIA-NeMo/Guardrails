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
    """Test that create_event handles empty string values without raising IndexError.

    Regression test for https://github.com/NVIDIA-NeMo/Guardrails/issues/1700
    """
    result = await create_event(event={"_type": "SomeEvent", "param": ""})
    assert len(result.events) == 1
    assert result.events[0]["param"] == ""


@pytest.mark.asyncio
async def test_create_event_dollar_prefix_resolved():
    """Test that $-prefixed string values are resolved from context."""
    result = await create_event(
        event={"_type": "SomeEvent", "param": "$user_name"},
        context={"user_name": "Alice"},
    )
    assert result.events[0]["param"] == "Alice"


@pytest.mark.asyncio
async def test_create_event_dollar_prefix_no_context():
    """Test that $-prefixed values resolve to None when no context is provided."""
    result = await create_event(
        event={"_type": "SomeEvent", "param": "$missing"},
    )
    assert result.events[0]["param"] is None


@pytest.mark.asyncio
async def test_create_event_regular_string():
    """Test that regular string values pass through unchanged."""
    result = await create_event(event={"_type": "SomeEvent", "param": "hello"})
    assert result.events[0]["param"] == "hello"
