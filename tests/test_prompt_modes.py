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

import os

from nemoguardrails import RailsConfig
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.types import Task

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs", "with_prompt_modes")
TEST_CASES = [
    (
        "task1_openai_compact",
        Task.GENERATE_USER_INTENT,
        "<<This custom prompt generates the user intent>>",
    ),
    (
        "task2_openai_standard",
        Task.GENERATE_USER_INTENT,
        "<<This is a placeholder for a custom prompt for generating the user intent using gpt-3.5-turbo>>",
    ),
    (
        "task3_nemo_compact",
        Task.SELF_CHECK_HALLUCINATION,
        "<<Check for hallucinations>>",
    ),
    (
        "task4_nemo_standard",
        Task.SELF_CHECK_HALLUCINATION,
        "<<This is a long placeholder prompt to check for hallucinations>>",
    ),
]


def colang_config():
    return """
    define user express greeting
        "hi"
        "hello"

    define flow
        user express greeting
        bot express greeting
    """


def yaml_config(task_name):
    if "openai" in task_name:
        engine = "openai"
        model = "gpt-3.5-turbo"
    else:
        engine = "nim"
        model = "meta"

    prompting_mode = "compact" if "compact" in task_name else "standard"

    return f"""
        models:
          - type: main
            engine: {engine}
            model: {model}
        prompting_mode: {prompting_mode}
    """


def test_prompting_modes():
    prompts_config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "prompts"))

    for test_case in TEST_CASES:
        task_name, task, expected_prompt = test_case

        task_config = RailsConfig.from_content(colang_config(), yaml_config(task_name))
        task_config.prompts = prompts_config.prompts

        prompt = get_prompt(task_config, task)

        assert prompt.content == expected_prompt
