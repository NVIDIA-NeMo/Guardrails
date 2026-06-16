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

from nemoguardrails import RailsConfig
from nemoguardrails.testing import FakeLLMModel, TestChat
from nemoguardrails.types import LLMResponse


def _minimal_config() -> RailsConfig:
    return RailsConfig.from_content(
        config={
            "models": [],
            "instructions": [
                {
                    "type": "general",
                    "content": "This is a conversation between a user and a bot.",
                }
            ],
        }
    )


def test_public_imports_are_re_exported():
    from nemoguardrails import testing as testing_pkg

    assert testing_pkg.FakeLLMModel is FakeLLMModel
    assert testing_pkg.TestChat is TestChat
    assert set(testing_pkg.__all__) == {"FakeLLMModel", "TestChat"}


def test_shim_re_exports_remain_compatible():
    from tests.utils import FakeLLMModel as ShimFakeLLMModel
    from tests.utils import TestChat as ShimTestChat

    assert ShimFakeLLMModel is FakeLLMModel
    assert ShimTestChat is TestChat


@pytest.mark.asyncio
async def test_fake_llm_model_smoke():
    llm = FakeLLMModel(responses=["hello"])

    result = await llm.generate_async(prompt="anything")

    assert isinstance(result, LLMResponse)
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_fake_llm_model_consumes_responses_in_order():
    llm = FakeLLMModel(responses=["first", "second"])

    first = await llm.generate_async(prompt="q1")
    second = await llm.generate_async(prompt="q2")

    assert first.content == "first"
    assert second.content == "second"

    with pytest.raises(RuntimeError, match="No responses available"):
        await llm.generate_async(prompt="q3")


@pytest.mark.asyncio
async def test_fake_llm_model_streaming_yields_chunks():
    llm = FakeLLMModel(responses=["one two three"])

    chunks = [chunk async for chunk in llm.stream_async(prompt="anything")]

    assembled = "".join(chunk.delta_content for chunk in chunks)
    assert assembled == "one two three"


def test_test_chat_user_bot_round_trip():
    chat = TestChat(_minimal_config(), llm_completions=["  Hello there!"])

    chat.user("hello!")
    chat.bot("Hello there!")


def test_test_chat_llm_exception_without_completions():
    chat = TestChat(_minimal_config(), llm_exception=RuntimeError("upstream is down"))

    chat.user("hi")
    with pytest.raises(Exception, match="upstream is down"):
        chat.bot("anything")


def test_fake_llm_fixture(fake_llm):
    assert isinstance(fake_llm, FakeLLMModel)
    assert fake_llm.responses == ["Hello!"]


def test_make_fake_llm_factory(make_fake_llm):
    llm = make_fake_llm(responses=["custom"])

    assert isinstance(llm, FakeLLMModel)
    assert llm.responses == ["custom"]


def test_make_test_chat_factory(make_test_chat):
    chat = make_test_chat(_minimal_config(), llm_completions=["Howdy!"])

    chat.user("hi")
    chat.bot("Howdy!")
