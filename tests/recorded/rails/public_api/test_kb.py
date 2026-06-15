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
NeMo Guardrails (often shortened to "Guardrails") is an open-source toolkit for adding programmable **guardrails** to **LLM-based conversational applications**.
In practice, it helps you control and shape the model's behavior so it can:
- **Stay on topic / follow a dialog flow** (conversational control)
- **Enforce safety or policy rules** (e.g., refuse certain requests, avoid unsafe outputs)
- **Apply consistent structured behavior** using predefined logic
- **React to the conversation context** (so the bot can behave differently depending on what's happening in the chat)
So instead of relying on the LLM alone to "behave," Guardrails gives you additional mechanisms to constrain and manage the responses--making the chatbot more reliable and safer in real-world use.
If you tell me what kind of chatbot you're building (support bot, booking assistant, internal QA, etc.), I can suggest a typical way to use Guardrails.\
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
NeMo Guardrails (often shortened to "Guardrails") is an open-source framework for adding safety, policy enforcement, and conversational control to AI chatbots--especially those built with large language models (LLMs).

In practice, it helps you define what the assistant **should** and **shouldn't** do, and it can:
- **Block or rewrite unsafe responses** (content moderation / safety constraints)
- **Enforce conversation rules** (e.g., "ask for missing info before giving instructions")
- **Handle edge cases** with predictable behaviors (fallbacks, refusals, escalation)
- Support **intent- and topic-based behaviors** using rails like flows, prompts, and policies

It's commonly used when you want more reliability than "prompting alone," because you can implement guardrails as structured rules rather than hoping the model follows instructions correctly every time.

If you tell me your use case (customer support bot, internal assistant, healthcare/finance, etc.), I can suggest how Guardrails would typically be set up.\
""",
                    "prompt_tokens": 583,
                    "completion_tokens": 213,
                    "total_tokens": 796,
                },
                {
                    "task": "generate_next_steps",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": """\
Hello! 👋

NeMo Guardrails (often shortened to "Guardrails") is an open-source framework for adding **safety**, **policy enforcement**, and **conversational control** to AI chatbots--especially those built with **large language models (LLMs)**.

In practice, it helps you define things like:
- **What the assistant is allowed to do** (and what it must refuse)
- **When to block or redirect** certain requests
- **How the conversation should flow** (e.g., required steps, fallback behavior)
- **How to handle disallowed content** with consistent responses
- **Integration with tools/actions** under controlled conditions

If you tell me what you're trying to accomplish (e.g., "block PII," "enforce a company policy," "add refusal templates," "control tool use"), I can explain how Guardrails typically sets that up.\
""",
                    "prompt_tokens": 266,
                    "completion_tokens": 187,
                    "total_tokens": 453,
                },
                {
                    "task": "generate_bot_message",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": """\
NeMo Guardrails (often shortened to "Guardrails") is an open-source toolkit for adding programmable **guardrails** to **LLM-based conversational applications**.

In practice, it helps you control and shape the model's behavior so it can:
- **Stay on topic / follow a dialog flow** (conversational control)
- **Enforce safety or policy rules** (e.g., refuse certain requests, avoid unsafe outputs)
- **Apply consistent structured behavior** using predefined logic
- **React to the conversation context** (so the bot can behave differently depending on what's happening in the chat)

So instead of relying on the LLM alone to "behave," Guardrails gives you additional mechanisms to constrain and manage the responses--making the chatbot more reliable and safer in real-world use.  \n\

If you tell me what kind of chatbot you're building (support bot, booking assistant, internal QA, etc.), I can suggest a typical way to use Guardrails.\
""",
                    "prompt_tokens": 817,
                    "completion_tokens": 200,
                    "total_tokens": 1017,
                },
            ],
        }
    )
