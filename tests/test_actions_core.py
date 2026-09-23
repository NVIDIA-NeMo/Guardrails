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
async def test_empty_string_value_does_not_crash():
    """Empty string values must not raise IndexError."""
    event = {"_type": "SomeEvent", "param": ""}
    result = await create_event(event=event)
    assert result.events[0]["param"] == ""


@pytest.mark.asyncio
async def test_dollar_prefix_resolves_from_context():
    """Values starting with $ should be resolved from context."""
    event = {"_type": "SomeEvent", "param": "$my_var"}
    result = await create_event(event=event, context={"my_var": "hello"})
    assert result.events[0]["param"] == "hello"


@pytest.mark.asyncio
async def test_dollar_prefix_without_context_resolves_to_none():
    """Values starting with $ resolve to None when context is absent."""
    event = {"_type": "SomeEvent", "param": "$my_var"}
    result = await create_event(event=event, context=None)
    assert result.events[0]["param"] is None


@pytest.mark.asyncio
async def test_plain_string_value_is_unchanged():
    """Non-$ strings should pass through unchanged."""
    event = {"_type": "SomeEvent", "param": "just text"}
    result = await create_event(event=event)
    assert result.events[0]["param"] == "just text"
