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

"""Live provider smoke suite — contract-drift backstop (RECORDED_TEST_PLAN §5.5).

Cassettes are frozen, so a provider that changes its response schema keeps the recorded
suite green (false confidence). These ``@pytest.mark.live`` tests hit the **real**
providers and assert the *contract shape* a schema change would break (a non-empty
assistant message; embeddings retrieval works), not exact content (which drifts).

They are a **maintainer / periodic** job, NOT part of the default run:

    # default run — these auto-skip (no real keys) and never touch the network
    pytest tests/rails/llm/test_live_smoke.py -m "not live" --block-network

    # maintainer / scheduled CI — real keys, no network block
    OPENAI_API_KEY=sk-... NVIDIA_API_KEY=nvapi-... pytest tests/rails/llm/test_live_smoke.py -m live

Each test auto-skips when its real key is absent or is the replay dummy, so the default
suite (and CI without secrets) stays green without special handling.
"""

from __future__ import annotations

import os

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.rails.public_api.configs import (
    NIM_BASELINE_CONFIG,
    OPENAI_BASELINE_CONFIG,
    OPENAI_KB_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.recorded.utils import DUMMY_NVIDIA_API_KEY, DUMMY_OPENAI_API_KEY

pytestmark = [pytest.mark.live]

# Marker phrase planted in the KB doc (mirrors test_kb.py).
KB_MARKER = "guardrails-kb-marker-7f3a"


def _require_real_key(env_name: str, dummy_value: str) -> None:
    value = os.environ.get(env_name)
    if not value or value == dummy_value:
        pytest.skip(f"{env_name} (a real key) is required for live tests")


@pytest.fixture
def live_openai_key() -> None:
    _require_real_key("OPENAI_API_KEY", DUMMY_OPENAI_API_KEY)


@pytest.fixture
def live_nvidia_key() -> None:
    _require_real_key("NVIDIA_API_KEY", DUMMY_NVIDIA_API_KEY)


def _assert_nonempty_assistant_message(result: GenerationResponse) -> None:
    assert isinstance(result, GenerationResponse)
    assert result.response, "expected at least one response message"
    message = result.response[0]
    assert message.get("role") == "assistant"
    assert isinstance(message.get("content"), str) and message["content"].strip()


@pytest.mark.asyncio
async def test_live_openai_generate_async_smoke(live_openai_key):
    """OpenAI still returns a usable chat completion through the public API."""
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "Say hello in a few words."}],
        options={"log": {"llm_calls": True}},
    )
    _assert_nonempty_assistant_message(result)
    # contract: at least one LLM call was made and carries a completion.
    assert result.log and result.log.llm_calls
    assert any(call.completion for call in result.log.llm_calls)


@pytest.mark.asyncio
async def test_live_nim_generate_async_smoke(live_nvidia_key):
    """NVIDIA NIM still returns a usable chat completion through the public API."""
    rails = LLMRails(load_config(NIM_BASELINE_CONFIG), verbose=False)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "Say hello in a few words."}],
        options={"log": {"llm_calls": True}},
    )
    _assert_nonempty_assistant_message(result)


@pytest.mark.asyncio
async def test_live_openai_kb_smoke(live_openai_key):
    """OpenAI embeddings still index + retrieve the KB chunk (the RAG contract)."""
    rails = LLMRails(load_config(OPENAI_KB_CONFIG), verbose=False)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "What is NeMo Guardrails?"}],
        options={"output_vars": ["relevant_chunks"]},
    )
    assert isinstance(result, GenerationResponse)
    assert result.output_data is not None
    relevant_chunks = result.output_data.get("relevant_chunks")
    assert isinstance(relevant_chunks, str) and KB_MARKER in relevant_chunks
