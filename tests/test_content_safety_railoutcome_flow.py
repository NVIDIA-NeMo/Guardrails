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

"""Behavioral proof that the Colang content_safety input flow drives the SHARED
library action through its neutral RailOutcome return.

The library action now returns a RailOutcome instead of a {allowed, ...} dict,
and flows.v1.co reads the neutral decision ($response.is_blocked) plus its
evidence ($response.metadata["policy_violations"]). A canned content_safety
model drives action -> RailOutcome -> Colang rendering end to end, with no
network: a blocked input renders the refusal, an allowed input passes through.
"""

import json
import textwrap

from nemoguardrails import RailsConfig
from tests.utils import FakeLLMModel, TestChat

SAFE_VERDICT = json.dumps({"User Safety": "safe"})
UNSAFE_VERDICT = json.dumps({"User Safety": "unsafe", "Safety Categories": "S1: Violence"})

CONFIG = textwrap.dedent(
    """
    models:
      - type: main
        engine: openai
        model: gpt-4o-mini
      - type: content_safety
        engine: openai
        model: placeholder

    rails:
      input:
        flows:
          - content safety check input $model=content_safety

    prompts:
      - task: content_safety_check_input $model=content_safety
        content: |
          Check: {{ user_input }}
        output_parser: nemoguard_parse_prompt_safety
        max_tokens: 50
    """
)


def _chat_with_verdict(verdict):
    config = RailsConfig.from_content(yaml_content=CONFIG)
    config.models = [model for model in config.models if model.type == "main"]

    chat = TestChat(config, llm_completions=["Hello! How can I help you?"])
    chat.app.runtime.registered_action_params["llms"] = {"content_safety": FakeLLMModel(responses=[verdict])}
    return chat


def test_unsafe_input_renders_refusal_through_railoutcome():
    chat = _chat_with_verdict(UNSAFE_VERDICT)
    response = chat.app.generate(messages=[{"role": "user", "content": "an unsafe request"}])
    assert response["content"] == "I'm sorry, I can't respond to that."


def test_safe_input_passes_through_railoutcome():
    chat = _chat_with_verdict(SAFE_VERDICT)
    response = chat.app.generate(messages=[{"role": "user", "content": "hello there"}])
    assert response["content"] == "Hello! How can I help you?"
