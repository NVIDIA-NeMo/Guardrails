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

"""Behavioral proof that the Colang topic_safety input flow drives the SHARED
library action through its neutral RailOutcome return.

The library action now returns a RailOutcome instead of a {on_topic} dict, and
flows.v1.co reads the neutral decision ($response.is_blocked). A canned model
drives action -> RailOutcome -> Colang rendering end to end, with no network:
an off-topic input renders the refusal, an on-topic input passes through.
"""

import textwrap

from nemoguardrails import RailsConfig
from tests.utils import FakeLLMModel, TestChat

CONFIG = textwrap.dedent(
    """
    models:
      - type: main
        engine: openai
        model: gpt-4o-mini
      - type: topic_control
        engine: openai
        model: placeholder

    rails:
      input:
        flows:
          - topic safety check input $model=topic_control

    prompts:
      - task: topic_safety_check_input $model=topic_control
        content: |
          Stay on topic.
    """
)


def _chat_with_verdict(verdict):
    config = RailsConfig.from_content(yaml_content=CONFIG)
    config.models = [model for model in config.models if model.type == "main"]

    chat = TestChat(config, llm_completions=["Hello! How can I help you?"])
    chat.app.runtime.registered_action_params["llms"] = {"topic_control": FakeLLMModel(responses=[verdict])}
    return chat


def test_off_topic_input_renders_refusal_through_railoutcome():
    chat = _chat_with_verdict("off-topic")
    response = chat.app.generate(messages=[{"role": "user", "content": "tell me about something unrelated"}])
    assert response["content"] == "I'm sorry, I can't respond to that."


def test_on_topic_input_passes_through_railoutcome():
    chat = _chat_with_verdict("on-topic")
    response = chat.app.generate(messages=[{"role": "user", "content": "a relevant question"}])
    assert response["content"] == "Hello! How can I help you?"
