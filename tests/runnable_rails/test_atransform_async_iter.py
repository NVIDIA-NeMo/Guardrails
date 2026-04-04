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

"""Tests that RunnableRails.atransform returns an AsyncIterator (fixes #1692)."""

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest
from langchain_core.runnables import RunnableLambda

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


@pytest.fixture
def passthrough_rails():
    """Create a minimal passthrough RunnableRails instance."""
    config = RailsConfig.from_content(yaml_content="models: []")
    return RunnableRails(config, passthrough=True)


def test_transform_returns_iterator(passthrough_rails):
    """transform() should yield results, not return a single value."""
    result = passthrough_rails.transform(iter(["hello"]))
    assert isinstance(result, Iterator)
    items = list(result)
    assert len(items) == 1


@pytest.mark.asyncio
async def test_atransform_returns_async_iterator(passthrough_rails):
    """atransform() should yield results as an async iterator."""

    async def _aiter():
        yield "hello"

    result = passthrough_rails.atransform(_aiter())
    assert isinstance(result, AsyncIterator)
    items = []
    async for item in result:
        items.append(item)
    assert len(items) == 1


@pytest.mark.asyncio
async def test_astream_through_pipeline(passthrough_rails):
    """RunnableRails should work in a pipeline with astream (reproduces #1692)."""
    pipeline = RunnableLambda(lambda x: x) | passthrough_rails

    items = []
    async for chunk in pipeline.astream("hello"):
        items.append(chunk)

    # Should get at least one result without TypeError
    assert len(items) >= 1
