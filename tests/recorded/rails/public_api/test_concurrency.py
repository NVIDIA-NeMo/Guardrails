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

"""Concurrency & contextvar isolation under ``asyncio.gather``.

Cassette replay serializes requests, so it cannot prove that concurrent
``generate_async`` calls do not cross-talk through the per-request contextvars
(``llm_stats_var``, ``generation_options_var``, ``explain_info_var``,
``streaming_handler_var``, ``raw_llm_request``). The decomposition centralizes all of
those in ``generation/generation_context.py``; if any were a mis-scoped global instead
of a properly copied contextvar, concurrent tasks would bleed completions / activated
rails / stats into each other.

``asyncio.gather`` runs each coroutine in its own copied context, so the correct
behavior is that concurrency changes nothing: the per-request result is identical to
running the same request alone. These tests assert exactly that. They complement the
sequential, single-task slice, are fully deterministic (``FakeLLMModel``, no cassette),
and pass identically on both pre-refactor and the decomposed refactor.

Assertions stay on the stable ``log.llm_calls`` invariant (not ``explain()``, whose
accumulate-vs-reset semantics differ across the two code paths).
"""

from __future__ import annotations

import asyncio

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.normalization import normalize_generation_response
from tests.recorded.rails.public_api.configs import INPUT_RAILS_CONFIG
from tests.recorded.rails.public_api.test_parity import GENERATE_CELLS, ParityCell
from tests.recorded.rails_config import load_config
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]

# How many times to re-run the concurrent batch to shake out scheduling-dependent races.
CONCURRENT_ROUNDS = 2


async def _run_cell(cell: ParityCell) -> tuple[str, dict]:
    """Run one parity cell in its own LLMRails + FakeLLM, return (id, normalized result)."""
    rails = LLMRails(load_config(cell.config), llm=FakeLLMModel(responses=[cell.main_output]), verbose=False)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": cell.message}],
        options={"log": {"activated_rails": True, "llm_calls": True}},
    )
    assert isinstance(result, GenerationResponse)
    return cell.id, normalize_generation_response(result)


async def test_concurrent_generate_async_results_do_not_cross_talk():
    """The parity matrix run concurrently must equal the same matrix run sequentially.

    If any per-request contextvar leaked across tasks, the activated-rails ordering,
    decisions, or llm_calls of one cell would contaminate another and the concurrent
    dict would diverge from the sequential baseline.
    """
    sequential = dict([await _run_cell(cell) for cell in GENERATE_CELLS])

    for _ in range(CONCURRENT_ROUNDS):
        concurrent = dict(await asyncio.gather(*(_run_cell(cell) for cell in GENERATE_CELLS)))
        assert concurrent == sequential


async def test_concurrent_generations_keep_per_request_llm_calls():
    """Each concurrent request sees only its own LLM completion (no llm_stats_var bleed)."""

    async def run(i: int) -> tuple[int, str, GenerationResponse]:
        out = f"unique-completion-{i}"
        rails = LLMRails(load_config(INPUT_RAILS_CONFIG), llm=FakeLLMModel(responses=[out]), verbose=False)
        result = await rails.generate_async(
            messages=[{"role": "user", "content": "allowed input"}],
            options={"log": {"llm_calls": True}},
        )
        assert isinstance(result, GenerationResponse)
        return i, out, result

    results = await asyncio.gather(*(run(i) for i in range(16)))

    for i, expected, result in results:
        assert result.log is not None
        completions = [call.completion for call in (result.log.llm_calls or [])]
        assert completions == [expected], f"task {i} saw {completions!r}, expected [{expected!r}]"
        assert isinstance(result.response, list)
        assert result.response[0]["content"] == expected
