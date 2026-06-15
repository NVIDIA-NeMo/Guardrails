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

"""Knowledge-base retrieval and prompt injection through ``generate_async`` (B3.1).

The KB is embedded with the OpenAI embeddings provider (a recorded HTTP call), and
the retrieved chunk is both returned via ``output_vars=["relevant_chunks"]`` and
injected into the chat request body that produces the answer.
"""

from __future__ import annotations

import json

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.cassette import cassette_request_jsons
from tests.recorded.normalization import normalize_generation_response
from tests.recorded.rails.public_api.configs import OPENAI_KB_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]

# Distinctive phrase planted in the kb/ doc; used to prove the chunk reached the prompt.
KB_MARKER = "guardrails-kb-marker-7f3a"


async def test_openai_kb_generate_async_retrieves_and_injects_chunks(
    openai_api_key, record_mode, recorded_cassette_path
):
    rails = LLMRails(load_config(OPENAI_KB_CONFIG), verbose=False)

    result = await rails.generate_async(
        messages=[{"role": "user", "content": "What is NeMo Guardrails?"}],
        options={"output_vars": ["relevant_chunks"], "log": {"activated_rails": True, "llm_calls": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert result.output_data is not None
    relevant_chunks = result.output_data.get("relevant_chunks")
    assert isinstance(relevant_chunks, str)
    assert KB_MARKER in relevant_chunks

    if record_mode == "none":
        bodies = cassette_request_jsons(recorded_cassette_path)
        # The retrieved KB context is injected into a chat request (the behavioral
        # fingerprint of RAG): the marker phrase appears in a recorded chat body.
        chat_bodies = [payload for payload in bodies if "messages" in payload]
        assert chat_bodies
        assert any(KB_MARKER in json.dumps(body) for body in chat_bodies)
        # B3.3: the embedding provider request-body fingerprint (model + input) is recorded too.
        embedding_requests = [payload for payload in bodies if "input" in payload and "messages" not in payload]
        assert embedding_requests, "expected a recorded embeddings request"
        assert all(payload.get("model") == "text-embedding-3-small" for payload in embedding_requests)
        assert all(payload.get("input") for payload in embedding_requests)

    assert result.output_data == snapshot(
        {
            "relevant_chunks": """\
NeMo Guardrails is an open-source toolkit for adding programmable guardrails to
LLM-based conversational applications. The unique retrieval marker phrase for this
document is "guardrails-kb-marker-7f3a".

Guardrails (or "rails" for short) control the output of a large language model, such
as keeping the bot on topic, following a predefined dialog path, or refusing unsafe
requests.\
"""
        }
    )
    assert normalize_generation_response(result) == snapshot(
        {
            "response": [
                {
                    "role": "assistant",
                    "content": """\
NVIDIA **NeMo Guardrails** is an open-source toolkit you can use to add **programmable "guardrails"** to LLM-based chat or conversational applications. The goal is to make the assistant's behavior **more reliable and safer**, and to reduce unwanted outputs.
In practice, NeMo Guardrails can help you:
- **Keep the model on topic** (based on what's appropriate for the conversation)
- **Constrain dialog flows** (so the bot follows certain paths/steps when needed)
- **Refuse unsafe or disallowed requests** (according to rules you define)
- **Control output formatting or policies** in a structured way
Instead of relying only on prompting, you define rules--often expressed as a combination of configurations and logic--so the system can intercept or correct the model's responses when they don't meet your criteria.
If you tell me what kind of chatbot you're building (support bot, booking assistant, internal policy bot, etc.), I can suggest which guardrails you'd likely want to start with.\
""",
                }
            ],
            "activated_rails": [
                {
                    "type": "dialog",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                },
                {
                    "type": "dialog",
                    "name": "generate next step",
                    "decisions": ["execute generate_next_steps"],
                    "stop": False,
                },
                {
                    "type": "generation",
                    "name": "generate bot message",
                    "decisions": ["execute retrieve_relevant_chunks", "execute generate_bot_message"],
                    "stop": False,
                },
            ],
            "llm_calls": [
                {
                    "task": "generate_user_intent",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": """\
NeMo Guardrails (often written as **NVIDIA NeMo Guardrails**) is a framework for building **more reliable, safer conversational AI**--especially chatbots and assistants.

In practice, it helps you control an AI model's behavior by adding guardrails such as:

- **Safety and policy constraints:** Define rules to prevent disallowed content or risky instructions.
- **Intent- and topic-based flows:** Steer the conversation using scripted or structured logic.
- **Response validation:** Check or constrain what the model is allowed to say.
- **Fallback behavior:** If the model is unsure or a request violates rules, it can follow a safe fallback response.
- **Grounding and consistency checks (depending on setup):** Ensure responses align with expected formats or provided knowledge.

A common use case is preventing a general LLM from straying off-topic, revealing sensitive data, or giving instructions you don't want--by enforcing guardrails around the conversation.

If you tell me what you're building (customer support bot, internal assistant, coding helper, etc.), I can explain which kinds of guardrails you'd typically use.\
""",
                    "prompt_tokens": 583,
                    "completion_tokens": 228,
                    "total_tokens": 811,
                },
                {
                    "task": "generate_next_steps",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": """\
Yes--**NVIDIA NeMo Guardrails** (often written as **NeMo Guardrails**) is a framework you can use to make conversational AI **more reliable and safer**, particularly for **chatbots and assistants**.

In practice, it helps you add "guardrails" around an LLM so it follows rules such as:

- **Refusing unsafe requests** (e.g., harmful content)
- **Staying within policy boundaries** (industry or internal rules)
- **Enforcing conversation structure** (required steps, formats, or workflows)
- **Handling missing context** (asking clarifying questions instead of guessing)
- **Managing fallback behavior** when the model is uncertain or off-track

If you tell me what you want to achieve (e.g., compliance filtering, refusal rules, tool-use constraints, or keeping a specific tone/format), I can explain how NeMo Guardrails would fit and what a typical setup looks like.\
""",
                    "prompt_tokens": 259,
                    "completion_tokens": 193,
                    "total_tokens": 452,
                },
                {
                    "task": "generate_bot_message",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": """\
NVIDIA **NeMo Guardrails** is an open-source toolkit you can use to add **programmable "guardrails"** to LLM-based chat or conversational applications. The goal is to make the assistant's behavior **more reliable and safer**, and to reduce unwanted outputs.

In practice, NeMo Guardrails can help you:
- **Keep the model on topic** (based on what's appropriate for the conversation)
- **Constrain dialog flows** (so the bot follows certain paths/steps when needed)
- **Refuse unsafe or disallowed requests** (according to rules you define)
- **Control output formatting or policies** in a structured way

Instead of relying only on prompting, you define rules--often expressed as a combination of configurations and logic--so the system can intercept or correct the model's responses when they don't meet your criteria.

If you tell me what kind of chatbot you're building (support bot, booking assistant, internal policy bot, etc.), I can suggest which guardrails you'd likely want to start with.\
""",
                    "prompt_tokens": 810,
                    "completion_tokens": 215,
                    "total_tokens": 1025,
                },
            ],
        }
    )
