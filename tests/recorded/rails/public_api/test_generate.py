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

from __future__ import annotations

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.assertions import (
    assert_generated_message,
    assert_generated_text,
    assert_llm_call_usage,
    assert_runtime_model_matches,
)
from tests.recorded.cassette import recorded_chat_response
from tests.recorded.normalization import normalize_generation_response
from tests.recorded.rails.public_api.configs import (
    NEMOGUARDS_FULL_CONFIG,
    NIM_BASELINE_CONFIG,
    NIM_MODEL,
    OPENAI_BASELINE_CONFIG,
    OPENAI_INVALID_MODEL_CONFIG,
    OPENAI_MODEL,
    OUTPUT_RAILS_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded]


@pytest.mark.vcr
def test_openai_generate_sync_public_contract(openai_api_key):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    result = rails.generate(prompt="Say a short safe greeting.")

    assert_generated_text(result)
    assert result == snapshot("Hello! How can I help you today?")


@pytest.mark.vcr
def test_nim_generate_sync_public_contract(nvidia_api_key):
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)

    result = rails.generate(messages=[{"role": "user", "content": "Say hello in one short sentence."}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": """\
<think>Hmm, the user asked me to say hello in one short sentence. That seems straightforward--they want a simple, friendly greeting. \n\

I should keep it concise and warm, matching the "one short sentence" request. No extra fluff. \n\

The user might be testing if I can follow instructions precisely, or maybe they just need a quick, cheerful response to start a conversation. \n\

I'll go with "Hello! How can I assist you today?"--it's polite, open-ended, and fits perfectly in one sentence. \n\

No need to overthink it; they kept it simple, so I'll keep my reply simple too.
</think>
Hello! How can I assist you today?\
""",
        }
    )


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_generate_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    result = await rails.generate_async(prompt="Say a short safe greeting.")

    assert_generated_text(result)
    assert result == snapshot("Hello! How can I help you today?")


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_nim_generate_async_public_contract(nvidia_api_key):
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)

    result = await rails.generate_async(messages=[{"role": "user", "content": "Say hello in one short sentence."}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": """\
<think>Okay, user just said "Say hello in one short sentence." Hmm, seems like a simple greeting task but let's unpack it. \n\

First, they want a *short* sentence - so no fluff. "Hello" itself is already short but maybe too bare? Wait, they said "in one short sentence" so the response should be a greeting, not just the word "hello". \n\

Noticing they used "Say hello" which is a command, so they're probably expecting me to actually say something like "Hi!" or "Hello!" but... wait, the instruction says "in one short sentence" meaning the whole response must be one sentence. \n\

*checks own capabilities* \n\
Can't do "Hi." because that's just a word. Needs to be a full sentence. "Hello!" is technically a sentence fragment but... wait, in English we often say "Hello!" as a greeting. But the user said "one short sentence" so maybe they want something grammatically complete? \n\

*thinks of options* \n\
- "Hello!" (too short, not a full sentence) \n\
- "Hi there!" (better but still fragment) \n\
- "Hello! How are you?" (too long) \n\

Ah! The most natural short sentence is just "Hello!" but technically it's an interjection. Wait, no - in common usage, we say "Hello" as a greeting and it's considered a sentence. Like "Hello!" as a standalone utterance. \n\

*double-checks user request* \n\
"Say hello in one short sentence" - so the response should be a sentence that says hello. The simplest answer is literally "Hello!" but to make it a sentence... wait, maybe they just want the greeting word. \n\

*considers user's possible intent* \n\
They might be testing if I follow instructions precisely. Or maybe non-native speaker wanting to practice English greetings. \n\

*decides* \n\
The safest, most direct answer is "Hello!" - it's the standard greeting, exactly one short sentence (as a greeting phrase), and meets "one short sentence" requirement. No extra words. \n\

*verifies length* \n\
"Hello!" is 6 characters, one word, one sentence. Perfect. \n\

*confirms no other interpretation* \n\
They didn't ask for creative greeting or anything else. Just "say hello" as instruction. So no need to overthink. \n\

Final decision: Respond with "Hello!" as the greeting.
</think>
Hello!\
""",
        }
    )


@pytest.mark.asyncio
async def test_output_rails_generate_async_blocks_fake_main_output():
    rails = LLMRails(load_config(OUTPUT_RAILS_CONFIG), llm=FakeLLMModel(responses=["block output"]), verbose=False)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "Say something."}],
        options={"log": {"activated_rails": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {
                    "type": "output",
                    "name": "output rail",
                    "decisions": [
                        "refuse to respond",
                        "execute retrieve_relevant_chunks",
                        "execute generate_bot_message",
                        "stop",
                    ],
                    "stop": True,
                },
            ],
            "llm_calls": [],
        }
    )


@pytest.mark.asyncio
async def test_output_rails_generate_async_modifies_fake_main_output():
    rails = LLMRails(load_config(OUTPUT_RAILS_CONFIG), llm=FakeLLMModel(responses=["modify output"]), verbose=False)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "Say something."}],
        options={"log": {"activated_rails": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "modified output"}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {"type": "output", "name": "output rail", "decisions": [], "stop": False},
            ],
            "llm_calls": [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_generate_async_log_matches_recorded_chat_completion(
    openai_api_key, record_mode, recorded_cassette_path
):
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    result = await rails.generate_async(
        prompt="Say a short safe greeting.",
        options={"log": {"llm_calls": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert_generated_text(result.response)
    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) == 1

    llm_call = result.log.llm_calls[0]
    assert llm_call.llm_provider_name == "openai"

    if record_mode == "none":
        expected = recorded_chat_response(recorded_cassette_path, request_model=OPENAI_MODEL)
        assert expected.raw_usage is not None
        assert expected.finish_reason == "stop"
        assert expected.request_id
        assert result.response == expected.content
        assert llm_call.completion == expected.content
        assert_llm_call_usage(llm_call, expected)
        assert_runtime_model_matches(llm_call, configured_model=OPENAI_MODEL, recorded_model=expected.model)

    assert normalize_generation_response(result) == snapshot(
        {
            "response": "Hello! How are you today?",
            "activated_rails": [],
            "llm_calls": [
                {
                    "task": "general",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": "Hello! How are you today?",
                    "prompt_tokens": 12,
                    "completion_tokens": 10,
                    "total_tokens": 22,
                }
            ],
        }
    )


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_nim_generate_async_log_matches_recorded_usage(nvidia_api_key, record_mode, recorded_cassette_path):
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        options={"log": {"llm_calls": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert result.response
    assert isinstance(result.response, list)
    response = result.response
    assert_generated_message(response[-1])
    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) == 1

    llm_call = result.log.llm_calls[0]
    assert llm_call.llm_provider_name == "nim"

    if record_mode == "none":
        expected = recorded_chat_response(recorded_cassette_path, request_model=NIM_MODEL)
        assert expected.raw_usage is not None
        assert expected.finish_reason == "stop"
        assert expected.request_id
        assert response[-1]["content"] == expected.content
        assert llm_call.completion == expected.content
        assert_llm_call_usage(llm_call, expected)
        assert_runtime_model_matches(llm_call, configured_model=NIM_MODEL, recorded_model=expected.model)

    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "Hello."}],
            "activated_rails": [],
            "llm_calls": [
                {
                    "task": "general",
                    "provider": "nim",
                    "model": "nvidia/nemotron-3-nano-30b-a3b",
                    "completion": "Hello.",
                    "prompt_tokens": 23,
                    "completion_tokens": 57,
                    "total_tokens": 80,
                }
            ],
        }
    )


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_nemoguards_full_generate_async(nvidia_api_key):
    rails = LLMRails(load_config(NEMOGUARDS_FULL_CONFIG), verbose=False)

    result = await rails.generate_async(prompt="what can you do?")

    assert_generated_text(result)
    assert result == snapshot("""\
<think>We need to respond as the AI assistant, talkative, specific details. Explain capabilities.
</think>
I'm a versatile, conversational AI that can help you with a wide range of tasks. Here's a quick snapshot of what I can do:

### 1. Answer Questions & Explain Concepts  \n\
- **General knowledge:** From "What's the capital of Mongolia?" to "Why does the sky appear blue?" I can dig up facts, provide concise explanations, or dive deep into scientific, historical, or cultural topics.  \n\
- **Technical subjects:** Need a rundown of how a transformer-based language model works, or how to write a Python script that parses CSV files? I've got you covered.

### 2. Write & Edit Content  \n\
- **Creative writing:** Short stories, poetry, dialogue snippets, world-building lore--just let me know the tone or theme you're after.  \n\
- **Professional writing:** Reports, proposals, emails, resumes, cover letters, blog posts, marketing copy, SVG or Markdown files, etc. I can polish the language, tighten structure, or generate drafts from scratch.  \n\
- **Academic help:** Essays, literature reviews, problem-sets, math derivations, lab write-ups--always with proper citations and a clear logical flow.

### 3. Coding & Programming  \n\
- **Multiple languages:** Python, JavaScript, Java, C++, Rust, SQL, HTML/CSS, Bash, and more.  \n\
- **Debugging & troubleshooting:** Paste a traceback or describe an error, and I'll suggest fixes.  \n\
- **Algorithms & data structures:** Explain concepts, sketch implementations, or provide ready-to-run snippets.  \n\
- **Web development:** Build static pages, set up a simple Flask/Django app, or draft a React component.  \n\
- **Automation scripts:** Generating batch files, cron-job ideas, Selenium scripts, REST-API calls, etc.

### 4. Learning & Tutoring  \n\
- **Step-by-step guidance:** From "How do I factor a quadratic?" to "Explain the basics of Bayesian inference."  \n\
- **Practice problems:** Generate quizzes, flashcards, or worked examples tailored to your skill level.  \n\
- **Study strategies:** Time-management tips, note-taking methods, exam-prep plans--whatever helps you retain information better.

### 5. Brainstorming & Planning  \n\
- **Idea generation:** Product concepts, story outlines, research topics, party themes, travel itineraries, you name it.  \n\
- **Project planning:** Create timelines, break down tasks, suggest resources, or draft Gantt-style outlines.  \n\
- **Decision support:** Compare pros/cons, run simple cost-benefit analyses, or help you weigh options with hypothetical scenarios.

### 6. Simulations & Role-Play  \n\
- **Conversation practice:** Want to simulate a job interview, a negotiation, or a language-learning dialogue? I can adopt various personas and give feedback.  \n\
- **Scenario exploration:** "What if humanity discovered cheap fusion energy tomorrow?" - I'll flesh out plausible outcomes, societal impacts, and technical hurdles.  \n\
- **Role-play exercises:** For language learning, public-speaking practice, or team-building scenarios.

### 7. Personal Assistance (within my limits)  \n\
- **Time-management tips:** Prioritization frameworks, Pomodoro tricks, habit-forming advice.  \n\
- **Health & wellness pointers:** General nutrition info, stress-relief techniques, simple workout concepts--but always remind you to consult a professional for medical or fitness plans.  \n\
- **Creative hobbies:** Suggest pottery techniques, knitting patterns, photography settings, or DIY home-improvement projects.

### 8. Fun & Entertainment  \n\
- **Trivia & puzzles:** Riddles, logic puzzles, lateral-thinking challenges, or quick brain teasers.  \n\
- **Games:** Text-based adventure stories, word games, or simple multiplayer scenarios.  \n\
- **Jokes, memes, and riddles:** Fresh material on demand--just ask!

---

#### How to Get the Most Out of Me  \n\
1. **Be specific:** The more detail you give about what you need, the better I can tailor the response.  \n\
2. **Iterate:** If the first answer isn't perfect, tell me what to adjust--more depth, a different tone, extra examples, etc.  \n\
3. **Ask follow-ups:** I can drill down into sub-topics, clarify jargon, or expand on any point you find interesting.

---

If you have a particular project, question, or just want to explore something new, swing it my way--I'm ready to dive in! 🚀\
""")


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_generate_async_invalid_model_raises(openai_api_key):
    rails = LLMRails(load_config(OPENAI_INVALID_MODEL_CONFIG), verbose=False)

    with pytest.raises(LLMCallException) as exc_info:
        await rails.generate_async(prompt="Say a short safe greeting.")
    assert getattr(exc_info.value.inner_exception, "status_code", None) == 404


@pytest.mark.asyncio
async def test_generate_async_without_prompt_or_messages_raises():
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), llm=FakeLLMModel(responses=["unused"]), verbose=False)

    with pytest.raises(ValueError, match="Either prompt or messages must be provided"):
        await rails.generate_async()


@pytest.mark.asyncio
async def test_generate_async_with_prompt_and_messages_raises():
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), llm=FakeLLMModel(responses=["unused"]), verbose=False)

    with pytest.raises(ValueError, match="Only one of prompt or messages can be provided"):
        await rails.generate_async(prompt="hi", messages=[{"role": "user", "content": "hi"}])
