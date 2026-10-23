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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.testing.fake_model import FakeLLMModel


def assistant_message(response):
    if isinstance(response, GenerationResponse):
        assert isinstance(response.response, list)
        return response.response[0]
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [None, {"llm_params": {"temperature": 0}}])
async def test_cached_history_preserves_updated_context_and_accepts_new_context(options):
    config = RailsConfig.from_content(
        yaml_content="""
models: []
rails:
  input:
    flows: [count turns]
  output:
    flows: [return count]
""",
        colang_content="""
define flow count turns
  $counter = $counter + 1

define flow return count
  $bot_message = execute format_counter(counter=$counter)
""",
    )
    rails = LLMRails(config, llm=FakeLLMModel(responses=["Answer"] * 3))

    async def format_counter(counter):
        return str(counter)

    rails.register_action(format_counter, name="format_counter")
    messages = [{"role": "context", "content": {"counter": 0}}, {"role": "user", "content": "Hello"}]
    first = assistant_message(await rails.generate_async(messages=messages, options=options))
    assert first == {"role": "assistant", "content": "1"}
    messages.extend([first, {"role": "user", "content": "Hello again"}])
    second = assistant_message(await rails.generate_async(messages=messages, options=options))
    assert second == {"role": "assistant", "content": "2"}
    messages.extend([second, {"role": "context", "content": {"counter": 10}}, {"role": "user", "content": "Reset"}])
    third = assistant_message(await rails.generate_async(messages=messages, options=options))
    assert third == {"role": "assistant", "content": "11"}
    assert messages[0]["content"] == {"counter": 0}
