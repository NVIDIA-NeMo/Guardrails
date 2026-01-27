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

import os

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.rails.llm.options import GenerationResponse

has_langchain_openai = check_optional_dependency("langchain_openai")

has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

skip_if_no_openai = pytest.mark.skipif(
    not (has_langchain_openai and has_openai_key),
    reason="Requires langchain_openai and OPENAI_API_KEY environment variable",
)


@skip_if_no_openai
def test_task_specific_model_for_generate_user_intent_and_generate_next_steps():
    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

              - type: generate_user_intent
                engine: openai
                model: gpt-4o-mini

              - type: generate_next_steps
                engine: openai
                model: gpt-4o-mini
        """,
    )

    rails = LLMRails(config)

    res = rails.generate(
        messages=[{"role": "user", "content": "what can you do?"}],
        options={"log": {"llm_calls": True}},
    )

    assert isinstance(res, GenerationResponse)
    assert res.log is not None
    assert res.log.llm_calls is not None
    assert len(res.log.llm_calls) > 0

    task_specific_tasks = ["generate_user_intent", "generate_next_steps"]

    generate_user_intent_calls = [call for call in res.log.llm_calls if call.task == "generate_user_intent"]
    assert len(generate_user_intent_calls) > 0
    for call in generate_user_intent_calls:
        assert call.llm_model_name == "gpt-4o-mini"
        assert call.llm_provider_name == "openai"

    generate_next_steps_calls = [call for call in res.log.llm_calls if call.task == "generate_next_steps"]
    assert len(generate_next_steps_calls) > 0
    for call in generate_next_steps_calls:
        assert call.llm_model_name == "gpt-4o-mini"
        assert call.llm_provider_name == "openai"

    other_calls = [call for call in res.log.llm_calls if call.task not in task_specific_tasks]
    for call in other_calls:
        assert call.llm_model_name == "gpt-3.5-turbo-instruct"
