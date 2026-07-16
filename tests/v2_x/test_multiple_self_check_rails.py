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
from nemoguardrails.testing.fake_model import FakeLLMModel


@pytest.mark.asyncio
async def test_multiple_self_check_input_tasks():
    config = RailsConfig.from_content(
        colang_content="""
        import core
        import guardrails
        import nemoguardrails.library.self_check.input_check

        flow input rails $input_text
            self check input $task="check_harmful"
            self check input $task="check_off_topic"

        flow main
            await user said $text
            bot say "ordinary response"
        """,
        yaml_content="""
        colang_version: "2.x"
        models: []
        enable_rails_exceptions: true
        prompts:
          - task: self_check_input $task=check_harmful
            content: User input {{ user_input }}
            output_parser: is_content_safe
          - task: self_check_input $task=check_off_topic
            content: User input {{ user_input }}
            output_parser: is_content_safe
        """,
    )
    llm = FakeLLMModel(responses=["safe", "unsafe"])
    rails = LLMRails(config, llm=llm)

    result = await rails.generate_async(messages=[{"role": "user", "content": "blocked_off_topic"}])

    assert [call.task for call in rails.explain().llm_calls] == [
        "self_check_input $task=check_harmful",
        "self_check_input $task=check_off_topic",
    ]
    assert [event["type"] for event in result["events"]] == ["InputRailException"]
    assert llm.inference_count == 2


@pytest.mark.asyncio
async def test_multiple_self_check_output_tasks():
    config = RailsConfig.from_content(
        colang_content="""
        import core
        import guardrails
        import nemoguardrails.library.self_check.output_check

        flow output rails $output_text
            self check output $task="check_inappropriate"
            self check output $task="check_data_leakage"

        flow main
            await user said $text
            bot say "response containing blocked_data_leakage"
        """,
        yaml_content="""
        colang_version: "2.x"
        models: []
        enable_rails_exceptions: true
        prompts:
          - task: self_check_output $task=check_inappropriate
            content: Bot response {{ bot_response }}
            output_parser: is_content_safe
          - task: self_check_output $task=check_data_leakage
            content: Bot response {{ bot_response }}
            output_parser: is_content_safe
        """,
    )
    llm = FakeLLMModel(responses=["safe", "unsafe"])
    rails = LLMRails(config, llm=llm)

    result = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])

    assert [call.task for call in rails.explain().llm_calls] == [
        "self_check_output $task=check_inappropriate",
        "self_check_output $task=check_data_leakage",
    ]
    assert [event["type"] for event in result["events"]] == ["OutputRailException"]
    assert llm.inference_count == 2
