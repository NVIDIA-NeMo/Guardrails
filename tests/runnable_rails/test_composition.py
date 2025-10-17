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

"""Tests for RunnableRails composition methods."""

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


@pytest.fixture
def rails():
    """Create a RunnableRails instance for testing."""
    config = RailsConfig.from_content(config={"models": []})
    llm = FakeLLM(responses=["test response"])
    return RunnableRails(config, llm=llm)


def test_with_config_exists(rails):
    """Test that with_config method exists and returns a new runnable."""
    config = RunnableConfig(tags=["test"])

    assert hasattr(rails, "with_config")

    new_rails = rails.with_config(config)
    assert new_rails is not rails
    from langchain_core.runnables.base import RunnableBinding

    assert isinstance(new_rails, RunnableBinding)


def test_with_config_preserves_functionality(rails):
    """Test that with_config preserves functionality."""
    config = RunnableConfig(tags=["test"])
    new_rails = rails.with_config(config)

    result = new_rails.invoke("Hello world")
    assert result == "test response"


def test_pipe_method_exists(rails):
    """Test that pipe method exists as alternative to | operator."""
    mock_runnable = RunnableLambda(lambda x: x.upper())

    assert hasattr(rails, "pipe")

    chained = rails.pipe(mock_runnable)
    assert chained is not rails

    result = chained.invoke("hello")
    assert result == "TEST RESPONSE"


def test_with_retry_exists(rails):
    """Test that with_retry method exists."""
    assert hasattr(rails, "with_retry")

    retry_rails = rails.with_retry()
    assert retry_rails is not rails


def test_with_fallbacks_exists(rails):
    """Test that with_fallbacks method exists."""
    fallback = RunnableLambda(lambda x: "fallback response")

    assert hasattr(rails, "with_fallbacks")

    fallback_rails = rails.with_fallbacks([fallback])
    assert fallback_rails is not rails


def test_assign_exists(rails):
    """Test that assign method exists for dict operations."""
    assert hasattr(rails, "assign")

    assigned_rails = rails.assign(extra_key=RunnableLambda(lambda x: "extra"))
    assert assigned_rails is not None


def test_pick_exists(rails):
    """Test that pick method exists for dict operations."""
    assert hasattr(rails, "pick")

    picked_rails = rails.pick("input")
    assert picked_rails is not None
