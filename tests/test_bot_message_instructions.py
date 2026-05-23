# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regression tests for the Colang 1.0 bot message instructions feature.

A `# comment` placed above a `bot <name>` line in a flow is documented as a
natural-language instruction to the LLM for that bot message. Before the fix
for issue #1322, when the bot message also had a predefined sample utterance
in `define bot <name>`, the runtime short-circuited with the sample utterance
and the comment never reached the model, so the documented behavior did not
work.

These tests exercise the comment-above-bot-say path end to end and assert
that the comment is propagated into the prompt sent to the LLM.
"""

from nemoguardrails import RailsConfig
from nemoguardrails.colang.v1_0.runtime.flows import (
    FlowConfig,
    compute_next_steps,
)
from tests.utils import TestChat

COLANG_WITH_COMMENT_INSTRUCTION = """
define user express greeting
    "Hello"
    "Hi"

define bot express greeting
    "Hello world! How are you?"

define flow
    user express greeting
    # Respond in a very formal way and introduce yourself.
    bot express greeting
"""


def test_comment_above_bot_say_becomes_bot_intent_instruction():
    """The runtime should attach the comment above `bot say` as instructions
    on the BotIntent event, so downstream consumers can use it."""
    config = RailsConfig.from_content(
        colang_content=COLANG_WITH_COMMENT_INSTRUCTION,
        yaml_content="models: []\n",
    )

    flow_configs = {flow["id"]: FlowConfig(id=flow["id"], elements=flow["elements"]) for flow in config.flows}

    history = [{"type": "UserIntent", "intent": "express greeting"}]
    next_steps = compute_next_steps(history, flow_configs, config, processing_log=[])

    bot_intent_events = [step for step in next_steps if step["type"] == "BotIntent"]
    assert bot_intent_events, "Expected a BotIntent next step"

    instructions = bot_intent_events[0].get("instructions")
    assert instructions == "Respond in a very formal way and introduce yourself.", (
        f"Expected the comment to be carried as BotIntent instructions, got: {instructions!r}"
    )


def test_comment_above_bot_say_reaches_llm_prompt():
    """End-to-end: when a comment is placed above `bot express greeting`, the
    LLM call used to render the bot message should contain that comment as an
    instruction, even when a predefined sample utterance is available.

    Before the fix, the predefined utterance short-circuited the LLM call and
    the comment was silently dropped.
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_WITH_COMMENT_INSTRUCTION,
        yaml_content="models:\n  - type: main\n    engine: openai\n    model: gpt-3.5-turbo-instruct\n",
    )

    chat = TestChat(
        config,
        llm_completions=[
            # Generate user intent.
            "  express greeting",
            # Generate bot message with the formal instruction applied.
            '  "Greetings. I am pleased to make your acquaintance; allow me to introduce myself."',
        ],
    )

    chat >> "Hi"
    chat << "Greetings. I am pleased to make your acquaintance; allow me to introduce myself."

    info = chat.app.explain()

    bot_message_prompts = [call.prompt for call in info.llm_calls if "generate_bot_message" in (call.task or "")]
    assert bot_message_prompts, "Expected a generate_bot_message LLM call to be made"
    assert any("Respond in a very formal way and introduce yourself." in prompt for prompt in bot_message_prompts), (
        "Expected the comment above `bot express greeting` to be included in the prompt sent to the LLM, "
        f"but it was missing from: {bot_message_prompts!r}"
    )


def test_predefined_bot_message_still_used_when_no_comment_instruction():
    """When there is no comment instruction above the bot statement, the
    runtime should still use the predefined sample utterance verbatim,
    preserving prior behavior."""
    colang = """
define user express greeting
    "Hello"
    "Hi"

define bot express greeting
    "Hello world! How are you?"

define flow
    user express greeting
    bot express greeting
"""

    config = RailsConfig.from_content(
        colang_content=colang,
        yaml_content="models:\n  - type: main\n    engine: openai\n    model: gpt-3.5-turbo-instruct\n",
    )

    chat = TestChat(
        config,
        llm_completions=[
            # Only the user-intent generation call is expected.
            "  express greeting",
        ],
    )

    chat >> "Hi"
    chat << "Hello world! How are you?"
