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

"""Tracing is output- and request-neutral (B2.1).

Enabling ``tracing`` must change neither the assembled output nor the provider request
body. Response neutrality is proven deterministically with ``FakeLLMModel`` (the
tracing code path forces and then restores the log options post-generation). Request
neutrality is proven by recording the same prompt through the baseline and the traced
config in sibling cassettes and asserting the chat request bodies are identical.
"""

from __future__ import annotations

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.cassette import cassette_request_jsons
from tests.recorded.normalization import normalize_generation_response
from tests.recorded.rails.public_api.configs import OPENAI_BASELINE_CONFIG, OPENAI_TRACING_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]

_PROMPT = [{"role": "user", "content": "Say a short safe greeting."}]
_OPTIONS = {"log": {"activated_rails": True, "llm_calls": True}}


async def test_tracing_output_neutral_with_fake_main():
    """B2.1 (response): the same fake provider output assembles identically with and
    without tracing enabled."""
    baseline_rails = LLMRails(
        load_config(OPENAI_BASELINE_CONFIG), llm=FakeLLMModel(responses=["a neutral reply"]), verbose=False
    )
    traced_rails = LLMRails(
        load_config(OPENAI_TRACING_CONFIG), llm=FakeLLMModel(responses=["a neutral reply"]), verbose=False
    )

    baseline = await baseline_rails.generate_async(messages=[{"role": "user", "content": "hello"}], options=_OPTIONS)
    traced = await traced_rails.generate_async(messages=[{"role": "user", "content": "hello"}], options=_OPTIONS)

    baseline_norm = normalize_generation_response(baseline)
    traced_norm = normalize_generation_response(traced)

    assert traced_norm == baseline_norm
    assert traced_norm == snapshot(
        {
            "response": [{"role": "assistant", "content": "a neutral reply"}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                }
            ],
            "llm_calls": [
                {
                    "task": "general",
                    "provider": "test",
                    "model": "fake",
                    "completion": "a neutral reply",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ],
        }
    )


@pytest.mark.vcr
async def test_openai_generate_async_untraced_request(openai_api_key):
    """Baseline anchor: records the untraced request body for the comparison below."""
    rails = LLMRails(load_config(OPENAI_BASELINE_CONFIG), verbose=False)

    result = await rails.generate_async(messages=_PROMPT, options=_OPTIONS)

    assert isinstance(result, GenerationResponse)
    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "Hello! How are you today?"}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                }
            ],
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


@pytest.mark.vcr
async def test_openai_generate_async_traced_request_matches_untraced(
    openai_api_key, record_mode, recorded_cassette_path
):
    """B2.1 (request): the traced config sends the same chat request body as baseline."""
    rails = LLMRails(load_config(OPENAI_TRACING_CONFIG), verbose=False)

    result = await rails.generate_async(messages=_PROMPT, options=_OPTIONS)

    assert isinstance(result, GenerationResponse)

    if record_mode == "none":
        baseline_cassette = recorded_cassette_path.parent / "test_openai_generate_async_untraced_request.yaml"
        baseline_bodies = [body for body in cassette_request_jsons(baseline_cassette) if "messages" in body]
        traced_bodies = [body for body in cassette_request_jsons(recorded_cassette_path) if "messages" in body]
        assert baseline_bodies and traced_bodies
        assert traced_bodies[-1] == baseline_bodies[-1]

    assert normalize_generation_response(result) == snapshot(
        {
            "response": [{"role": "assistant", "content": "Hello! How can I help you today?"}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                }
            ],
            "llm_calls": [
                {
                    "task": "general",
                    "provider": "openai",
                    "model": "gpt-5.4-nano",
                    "completion": "Hello! How can I help you today?",
                    "prompt_tokens": 12,
                    "completion_tokens": 12,
                    "total_tokens": 24,
                }
            ],
        }
    )
