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
from nemoguardrails.rails.llm.options import RailStatus
from tests.recorded.assertions import assert_generated_message, assert_rails_result
from tests.recorded.normalization import normalize_rails_result
from tests.recorded.rails.public_api.configs import DIALOG_CONFIG, DIALOG_GENERATION_CONFIG, SINGLE_CALL_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


@pytest.mark.vcr
async def test_dialog_generate_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(DIALOG_CONFIG), verbose=False)

    result = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
        }
    )


@pytest.mark.vcr
async def test_dialog_generation_bot_message_public_contract(openai_api_key):
    """Bot-message GENERATION prompt carries few-shot examples + retrieved kb chunks.

    "tell me about your product" maps (via a flow) to the bot intent
    "describe the product", which has no predefined message, so
    ``generate_bot_message`` calls the LLM with the bot-message index examples and
    the retrieved kb chunks. VCR matches on ``recorded_body``, so this cassette
    pins all three dialog prompts (intent, next-step, bot-message) by their exact
    request bodies. It was verified to replay byte-identically against both the
    pre-refactor ``generation.py`` on ``develop`` and the refactored branch, so
    the prompts are unchanged pre/post -- a prompt-fidelity equivalence check the
    prompt-independent ``FakeLLMModel`` oracle cannot make.
    """
    rails = LLMRails(load_config(DIALOG_GENERATION_CONFIG), verbose=False)

    result = await rails.generate_async(messages=[{"role": "user", "content": "tell me about your product"}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": """\
Sure! If you mean the **guardrails toolkit for large language models** you mentioned in the context, here’s what it is and what it helps you do:
- **Programmable “rails” for LLMs:** Lets developers define rules and logic that run during the conversation.
- **Check user input:** You can validate or filter what users send (e.g., detect disallowed content, enforce formatting, constrain requests).
- **Steer the dialog:** You can route or modify the assistant’s behavior based on the input or conversation state (e.g., ask clarifying questions, refuse safely, switch modes).
- **Validate bot output:** Before the response is shown to the user, rails can verify it meets requirements (e.g., must include/avoid certain content, follow a policy, adhere to a schema).
- **Reduce unsafe or off-policy responses:** The overall goal is to make LLM behavior more controlled, consistent, and safer.
If you tell me what kind of use case you’re building (customer support, tutoring, legal/health Q&A, internal tools, etc.), I can suggest what kinds of rails you’d typically implement.\
""",
        }
    )


@pytest.mark.vcr
async def test_single_call_generate_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(SINGLE_CALL_CONFIG), verbose=False)

    result = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": """\
bot express greeting
"Hello again! 😊 How can I assist you today?"\
""",
        }
    )


async def test_single_call_check_async_without_io_rails():
    rails = LLMRails(load_config(SINGLE_CALL_CONFIG), verbose=False)

    result = await rails.check_async([{"role": "user", "content": "hello"}])

    assert_rails_result(result, status=RailStatus.PASSED, content="hello")
    assert normalize_rails_result(result) == snapshot({"status": "passed", "rail": None, "content": "hello"})
