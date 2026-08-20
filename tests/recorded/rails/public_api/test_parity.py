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

"""Parity baseline: a broad regression net over rail-pipeline assembly.

This module is the ``generate_async`` leg of a three-legged deterministic parity
backstop. Each public entry point gets a matrix over the same rule-based rail configs
(input / output / input+output / parallel) across the pass / block / modify decision
space, run with a deterministic main (``FakeLLMModel`` or a fake stream generator) and
no cassette, so the full normalized result is snapshotted and any decomposition change
to how the rail pipeline is assembled (order, decisions, stop, response, llm_calls)
trips a cell. The three legs are:

* ``generate_async`` -> this module (``test_parity_generate_async_rail_assembly``).
* ``check_async`` -> ``test_check.py::test_check_async_public_contracts`` (+ its sync
  twin ``test_check_sync_public_contracts``) — the same configs across
  pass/block/modify/auto/explicit, snapshotted via ``normalize_rails_result``.
* ``stream_async`` -> ``test_stream.py``'s deterministic ``streaming_output_rails``
  matrix (allowed / blocked / allow-then-block-boundary / disabled, driven by a fake
  ``generator=``), snapshotted via ``normalize_stream_chunks``.

Keeping each leg in its method's file avoids duplicating those matrices here; together
they are the broad net behind the targeted scenario tests.

It deliberately uses ``FakeLLMModel`` rather than recording: the provider-output
contract is already pinned by the targeted provider tests (test_generate / test_stream
/ test_check / test_dialog / test_requests and the new options/state/tools/kb/tracing
tests), so re-recording those here would only double-record. This module instead nets
the cross-config assembly behavior deterministically, with no cassette.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse
from tests.recorded.normalization import normalize_generation_response
from tests.recorded.rails.public_api.configs import (
    INPUT_OUTPUT_RAILS_CONFIG,
    INPUT_RAILS_CONFIG,
    OUTPUT_RAILS_CONFIG,
    PARALLEL_RAILS_CONFIG,
)
from tests.recorded.rails_config import RailsConfigSource, load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


@dataclass(frozen=True)
class ParityCell:
    id: str
    config: RailsConfigSource
    message: str
    main_output: str


GENERATE_CELLS = [
    ParityCell("input-allowed", INPUT_RAILS_CONFIG, "allowed input", "fake main output"),
    ParityCell("input-blocked", INPUT_RAILS_CONFIG, "block input", "fake main output"),
    ParityCell("input-modified", INPUT_RAILS_CONFIG, "modify input", "fake main output"),
    ParityCell("output-allowed", OUTPUT_RAILS_CONFIG, "say something", "safe output"),
    ParityCell("output-blocked", OUTPUT_RAILS_CONFIG, "say something", "block output"),
    ParityCell("output-modified", OUTPUT_RAILS_CONFIG, "say something", "modify output"),
    ParityCell("io-both-pass", INPUT_OUTPUT_RAILS_CONFIG, "allowed input", "safe output"),
    ParityCell("io-input-block", INPUT_OUTPUT_RAILS_CONFIG, "block input", "safe output"),
    ParityCell("io-output-block", INPUT_OUTPUT_RAILS_CONFIG, "allowed input", "block output"),
    ParityCell("parallel-pass", PARALLEL_RAILS_CONFIG, "allowed parallel", "fake main output"),
    ParityCell("parallel-block", PARALLEL_RAILS_CONFIG, "block parallel", "fake main output"),
    ParityCell("parallel-modified", PARALLEL_RAILS_CONFIG, "modify parallel", "fake main output"),
]


async def test_parity_generate_async_rail_assembly():
    results = {}
    for cell in GENERATE_CELLS:
        rails = LLMRails(load_config(cell.config), llm=FakeLLMModel(responses=[cell.main_output]), verbose=False)
        result = await rails.generate_async(
            messages=[{"role": "user", "content": cell.message}],
            options={"log": {"activated_rails": True, "llm_calls": True}},
        )
        assert isinstance(result, GenerationResponse)
        results[cell.id] = normalize_generation_response(result)

    assert results == snapshot(
        {
            "input-allowed": {
                "response": [{"role": "assistant", "content": "fake main output"}],
                "activated_rails": [
                    {"type": "input", "name": "input rail", "decisions": [], "stop": False},
                    {
                        "type": "generation",
                        "name": "generate user intent",
                        "decisions": ["execute generate_user_intent"],
                        "stop": False,
                    },
                ],
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "fake main output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "input-blocked": {
                "response": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}],
                "activated_rails": [
                    {
                        "type": "input",
                        "name": "input rail",
                        "decisions": [
                            "refuse to respond",
                            "execute retrieve_relevant_chunks",
                            "execute generate_bot_message",
                            "stop",
                        ],
                        "stop": True,
                    }
                ],
                "llm_calls": [],
            },
            "input-modified": {
                "response": [{"role": "assistant", "content": "fake main output"}],
                "activated_rails": [
                    {"type": "input", "name": "input rail", "decisions": [], "stop": False},
                    {
                        "type": "generation",
                        "name": "generate user intent",
                        "decisions": ["execute generate_user_intent"],
                        "stop": False,
                    },
                ],
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "fake main output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "output-allowed": {
                "response": [{"role": "assistant", "content": "safe output"}],
                "activated_rails": [
                    {
                        "type": "generation",
                        "name": "generate user intent",
                        "decisions": ["execute generate_user_intent"],
                        "stop": False,
                    },
                    {"type": "output", "name": "output rail", "decisions": [], "stop": False},
                ],
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "safe output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "output-blocked": {
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
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "block output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "output-modified": {
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
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "modify output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "io-both-pass": {
                "response": [{"role": "assistant", "content": "safe output"}],
                "activated_rails": [
                    {"type": "input", "name": "input rail", "decisions": [], "stop": False},
                    {
                        "type": "generation",
                        "name": "generate user intent",
                        "decisions": ["execute generate_user_intent"],
                        "stop": False,
                    },
                    {"type": "output", "name": "output rail", "decisions": [], "stop": False},
                ],
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "safe output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "io-input-block": {
                "response": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}],
                "activated_rails": [
                    {
                        "type": "input",
                        "name": "input rail",
                        "decisions": [
                            "refuse to respond",
                            "execute retrieve_relevant_chunks",
                            "execute generate_bot_message",
                            "stop",
                        ],
                        "stop": True,
                    }
                ],
                "llm_calls": [],
            },
            "io-output-block": {
                "response": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}],
                "activated_rails": [
                    {"type": "input", "name": "input rail", "decisions": [], "stop": False},
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
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "block output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "parallel-pass": {
                "response": [{"role": "assistant", "content": "fake main output"}],
                "activated_rails": [
                    {
                        "type": "input",
                        "name": "parallel input rail",
                        "decisions": ["execute check_parallel_blocked"],
                        "stop": False,
                    },
                    {
                        "type": "input",
                        "name": "parallel input pass rail",
                        "decisions": ["execute check_parallel_modified"],
                        "stop": False,
                    },
                    {
                        "type": "generation",
                        "name": "generate user intent",
                        "decisions": ["execute generate_user_intent"],
                        "stop": False,
                    },
                ],
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "fake main output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
            "parallel-block": {
                "response": [{"role": "assistant", "content": "Parallel input blocked."}],
                "activated_rails": [
                    {
                        "type": "input",
                        "name": "parallel input pass rail",
                        "decisions": ["execute check_parallel_modified"],
                        "stop": False,
                    },
                    {"type": "input", "name": "parallel input rail", "decisions": ["stop"], "stop": True},
                ],
                "llm_calls": [],
            },
            "parallel-modified": {
                "response": [{"role": "assistant", "content": "fake main output"}],
                "activated_rails": [
                    {
                        "type": "input",
                        "name": "parallel input rail",
                        "decisions": ["execute check_parallel_blocked"],
                        "stop": False,
                    },
                    {
                        "type": "input",
                        "name": "parallel input pass rail",
                        "decisions": ["execute check_parallel_modified"],
                        "stop": False,
                    },
                    {
                        "type": "generation",
                        "name": "generate user intent",
                        "decisions": ["execute generate_user_intent"],
                        "stop": False,
                    },
                ],
                "llm_calls": [
                    {
                        "task": "general",
                        "provider": "test",
                        "model": "fake",
                        "completion": "fake main output",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                ],
            },
        }
    )
