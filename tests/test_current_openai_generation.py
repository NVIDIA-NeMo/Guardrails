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

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.testing.fake_model import FakeLLMModel


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-6-luna"])
@pytest.mark.parametrize(
    "task",
    [
        "generate_user_intent",
        "generate_next_steps",
        "generate_bot_message",
        "generate_intent_steps_message",
        "generate_value",
    ],
)
def test_current_models_use_colang1_chat_prompts(model, task):
    config = RailsConfig.from_content(config={"models": [{"type": "main", "engine": "openai", "model": model}]})
    prompt = get_prompt(config, task)
    assert prompt.messages
    assert not prompt.content
    rendered = LLMTaskManager(config).render_task_prompt(
        task,
        events=[{"type": "UserMessage", "text": "How much vacation?"}],
        context={
            "examples": 'user "How much vacation?"\n  ask about benefits\nbot explain benefits\n  "Two weeks."',
            "potential_user_intents": "ask about benefits",
            "relevant_chunks": "Vacation policy: two weeks.",
            "var_name": "is_safe",
            "instructions": "Return True if safe, False otherwise.",
        },
    )
    assert isinstance(rendered, list)
    assert rendered[0]["type"] == "system"
    text = "\n".join(message["content"] for message in rendered)
    assert "How much vacation?" in text
    if task in ("generate_bot_message", "generate_intent_steps_message"):
        assert "Vacation policy: two weeks." in text
    if task in ("generate_user_intent", "generate_intent_steps_message"):
        assert "User intent: <intent name>" in text
        assert "ask about benefits" in text
    if task == "generate_value":
        assert "Python literal" in text
        assert "Return True if safe, False otherwise." in text


@pytest.mark.asyncio
@pytest.mark.parametrize("single_call", [False, True])
async def test_sample_example_uses_greeting_flow(single_call):
    config = RailsConfig.from_path(str(Path(__file__).parents[1] / "examples/configs/sample"))
    config.models[0].model = "gpt-6-luna"
    config.rails.dialog.single_call.enabled = single_call
    answer = "Hello, hello! Glad you stopped by."
    responses = (
        [f'User intent: express greeting\nBot intent: express greeting\nBot message: "{answer}"']
        if single_call
        else ["User intent: express greeting", f'Bot message: "{answer}"']
    )
    model = FakeLLMModel(responses=responses)
    rails = LLMRails(config, llm=model)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "hello"}],
        options={"log": {"activated_rails": True, "internal_events": True}},
    )
    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    assert result.response[0]["content"] == answer + "!!"
    assert result.log is not None and result.log.internal_events is not None
    assert [
        (e["type"], e["intent"]) for e in result.log.internal_events if e["type"] in ("UserIntent", "BotIntent")
    ] == [("UserIntent", "express greeting"), ("BotIntent", "express greeting")]
    assert {rail.name for rail in result.log.activated_rails} >= {"dummy input rail", "dummy output rail"}
    assert model.inference_count == (1 if single_call else 2)
