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

"""Pytest fixtures exposing the public testing surface.

To opt in, add the following to your project's ``conftest.py``::

    pytest_plugins = ["nemoguardrails.testing.fixtures"]

The fixtures below provide reasonable defaults that cover the most common
testing patterns: a scriptable :class:`FakeLLMModel` and a :class:`TestChat`
factory that wires the fake model into an :class:`LLMRails` app.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.testing.chat_harness import TestChat
from nemoguardrails.testing.fake_model import FakeLLMModel


@pytest.fixture
def fake_llm() -> FakeLLMModel:
    """Return a :class:`FakeLLMModel` with a single ``"Hello!"`` response.

    Override the responses by overriding this fixture in your own conftest, or
    instantiate :class:`FakeLLMModel` directly when you need custom behaviour::

        from nemoguardrails.testing import FakeLLMModel

        llm = FakeLLMModel(responses=["My scripted answer"])
    """

    return FakeLLMModel(responses=["Hello!"])


@pytest.fixture
def make_fake_llm() -> Callable[..., FakeLLMModel]:
    """Return a factory for building :class:`FakeLLMModel` instances.

    Useful when a single test needs more than one fake model, or when the set
    of responses is computed dynamically::

        def test_two_calls(make_fake_llm):
            llm_one = make_fake_llm(responses=["one"])
            llm_two = make_fake_llm(responses=["two"])
    """

    def _factory(responses: Optional[List[str]] = None, **kwargs) -> FakeLLMModel:
        return FakeLLMModel(responses=responses, **kwargs)

    return _factory


@pytest.fixture
def make_test_chat() -> Callable[..., TestChat]:
    """Return a factory that builds :class:`TestChat` instances bound to a
    fake LLM.

    Example::

        def test_greeting(make_test_chat):
            config = RailsConfig.from_path("./examples/bots/abc")
            chat = make_test_chat(config, llm_completions=["Hi there!"])
            chat.user("hi")
            chat.bot("Hi there!")
    """

    def _factory(
        config: RailsConfig,
        llm_completions: Optional[List[str]] = None,
        **kwargs,
    ) -> TestChat:
        return TestChat(config=config, llm_completions=llm_completions, **kwargs)

    return _factory
