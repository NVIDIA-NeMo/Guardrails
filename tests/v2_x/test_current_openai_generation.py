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

from pathlib import Path

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.testing.chat_harness import TestChat


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-5.6-terra"])
@pytest.mark.parametrize(
    "task",
    [
        "generate_user_intent_from_user_action",
        "generate_user_intent_and_bot_action_from_user_action",
        "generate_flow_continuation",
        "generate_value_from_instruction",
    ],
)
def test_current_models_use_explicit_colang_chat_prompts(model, task):
    config = RailsConfig.from_content(
        yaml_content=f'colang_version: "2.x"\nmodels:\n  - type: main\n    engine: openai\n    model: {model}'
    )
    prompt = get_prompt(config, task)
    assert prompt.messages
    assert not prompt.content
    rendered = LLMTaskManager(config).render_task_prompt(
        task,
        events=[],
        context={
            "examples": "user ask about benefits",
            "context": {"relevant_chunks": "Vacation policy: two weeks."},
            "user_action": 'user said "How much vacation?"',
            "var_name": "is_safe",
            "instructions": "Return True if safe, False otherwise.",
        },
    )
    text = "\n".join(message["content"] for message in rendered)
    if task in ("generate_flow_continuation", "generate_user_intent_and_bot_action_from_user_action"):
        assert "bot intent:" in text
        assert 'bot action: bot say "<response>"' in text
        assert "Vacation policy: two weeks." in text
    elif task == "generate_value_from_instruction":
        assert "Python literal" in text
        assert "Return True if safe, False otherwise." in text
    else:
        assert "user intent:" in text
        assert "user ask about benefits" in text


@pytest.mark.parametrize("example", ["bots/abc_v2", "v2_x/tutorial/guardrails_1"])
def test_shipped_example_with_curly_quoted_continuation(example):
    config = RailsConfig.from_path(str(Path(__file__).parents[2] / "examples" / example))
    config.models[0].model = "gpt-6-luna"
    completion = (
        "bot intent: bot answer question\n"
        "bot action: bot say “Employees get two weeks of paid vacation.”\n"
        'user action: user said "Thanks"\n'
        'bot intent: bot acknowledge\nbot action: bot say "You are welcome"'
    )
    responses = ["user intent: user ask question about benefits", completion]
    if example == "bots/abc_v2":
        responses = ["No", *responses, "No"]
    else:
        responses = ["True", *responses]
    chat = TestChat(config, llm_completions=responses)

    chat >> "How much paid vacation do employees get?"
    chat << "Employees get two weeks of paid vacation."
    assert chat.llm.inference_count == len(responses)
