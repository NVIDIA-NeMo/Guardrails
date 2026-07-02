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

"""GenerationOptions effects on the rail pipeline (rails.input toggle, log ordering).

The rails under test are deterministic Colang flows (no provider call), and the main
generation is driven by ``FakeLLMModel``, so these are no-cassette tests: only the
option handling in generation_context / rails_check is exercised, which does not
depend on a real provider response.
"""

from __future__ import annotations

import pytest

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from tests.recorded.normalization import normalize_generation_response
from tests.recorded.rails.public_api.configs import (
    INPUT_OUTPUT_RAILS_CONFIG,
    INPUT_RAILS_CONFIG,
    OUTPUT_RAILS_CONFIG,
    RETRIEVAL_RAILS_CONFIG,
    TWO_INPUT_RAILS_CONFIG,
)
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot
from tests.utils import FakeLLMModel

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


async def _generate(config, messages, options, *, main_output):
    rails = LLMRails(load_config(config), llm=FakeLLMModel(responses=[main_output]), verbose=False)
    result = await rails.generate_async(messages=messages, options=options)
    assert isinstance(result, GenerationResponse)
    return result


async def test_input_rails_disabled_by_options_skips_input_rail():
    """B1.1: ``options.rails.input=False`` removes the input rail from the pipeline.

    The same blocking input is run twice. With default options the input rail fires
    and stops generation (refusal); with ``rails.input=False`` the input rail never
    appears in ``log.activated_rails`` and the fake main output is returned instead.
    """
    message = [{"role": "user", "content": "block input"}]
    log_options = {"log": {"activated_rails": True, "llm_calls": True}}

    default_run = await _generate(INPUT_RAILS_CONFIG, message, log_options, main_output="fake main output")
    disabled_run = await _generate(
        INPUT_RAILS_CONFIG,
        message,
        {"rails": {"input": False}, **log_options},
        main_output="fake main output",
    )

    default_norm = normalize_generation_response(default_run)
    disabled_norm = normalize_generation_response(disabled_run)

    default_rail_names = {rail["name"] for rail in default_norm["activated_rails"]}
    disabled_rail_names = {rail["name"] for rail in disabled_norm["activated_rails"]}
    assert "input rail" in default_rail_names
    assert "input rail" not in disabled_rail_names

    assert default_norm == snapshot(
        {
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
        }
    )
    assert disabled_norm == snapshot(
        {
            "response": [{"role": "assistant", "content": "fake main output"}],
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
                    "completion": "fake main output",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            ],
        }
    )


async def test_activated_rails_ordering_and_decisions():
    """B1.10: ``log.activated_rails`` preserves pipeline order and per-rail decisions.

    Input passes, dialog/generation runs, then the output rail blocks. The ordered
    list (input -> generation -> output) plus each rail's decisions and ``stop`` flag
    is the contract the decomposition must preserve.
    """
    result = await _generate(
        INPUT_OUTPUT_RAILS_CONFIG,
        [{"role": "user", "content": "allowed input"}],
        {"log": {"activated_rails": True, "llm_calls": True}},
        main_output="block output",
    )

    assert normalize_generation_response(result) == snapshot(
        {
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
        }
    )


async def test_generation_options_object_matches_dict_equivalent():
    """A1.4: a ``GenerationOptions`` object and the equivalent dict produce the same result."""
    message = [{"role": "user", "content": "hi"}]
    dict_result = await _generate(OUTPUT_RAILS_CONFIG, message, {"log": {"activated_rails": True}}, main_output="safe")
    obj_result = await _generate(
        OUTPUT_RAILS_CONFIG, message, GenerationOptions(log={"activated_rails": True}), main_output="safe"
    )

    dict_norm = normalize_generation_response(dict_result)
    assert dict_norm == normalize_generation_response(obj_result)
    assert dict_norm == snapshot(
        {
            "response": [{"role": "assistant", "content": "safe"}],
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


async def test_output_rails_disabled_by_options_skips_output_rail():
    """B1.2: ``options.rails.output=False`` removes the output rail; bot output passes through."""
    message = [{"role": "user", "content": "hi"}]
    log_options = {"log": {"activated_rails": True}}

    default = await _generate(OUTPUT_RAILS_CONFIG, message, log_options, main_output="block output")
    disabled = await _generate(
        OUTPUT_RAILS_CONFIG, message, {"rails": {"output": False}, **log_options}, main_output="block output"
    )

    default_norm = normalize_generation_response(default)
    disabled_norm = normalize_generation_response(disabled)
    assert "output rail" in {r["name"] for r in default_norm["activated_rails"]}
    assert "output rail" not in {r["name"] for r in disabled_norm["activated_rails"]}
    assert default_norm == snapshot(
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
    assert disabled_norm == snapshot(
        {
            "response": [{"role": "assistant", "content": "block output"}],
            "activated_rails": [
                {
                    "type": "generation",
                    "name": "generate user intent",
                    "decisions": ["execute generate_user_intent"],
                    "stop": False,
                }
            ],
            "llm_calls": [],
        }
    )


async def test_dialog_disabled_by_options_skips_generation():
    """B1.4: ``options.rails.dialog=False`` skips dialog/generation; only input rails run."""
    result = await _generate(
        INPUT_OUTPUT_RAILS_CONFIG,
        [{"role": "user", "content": "allowed input"}],
        {"rails": {"dialog": False}, "log": {"activated_rails": True}},
        main_output="safe",
    )

    norm = normalize_generation_response(result)
    assert "generation" not in {r["type"] for r in norm["activated_rails"]}
    assert norm == snapshot(
        {
            "response": [{"role": "assistant", "content": "allowed input"}],
            "activated_rails": [{"type": "input", "name": "input rail", "decisions": [], "stop": False}],
            "llm_calls": [],
        }
    )


async def test_output_vars_true_returns_full_context():
    """B1.7: ``options.output_vars=True`` returns the whole context in ``output_data``."""
    result = await _generate(
        INPUT_RAILS_CONFIG, [{"role": "user", "content": "modify input"}], {"output_vars": True}, main_output="x"
    )

    assert result.output_data is not None
    assert result.output_data["user_message"] == "modified input"
    assert sorted(result.output_data.keys()) == snapshot(
        [
            "bot_message",
            "event",
            "generation_options",
            "i",
            "input_flows",
            "last_bot_message",
            "last_user_message",
            "triggered_input_rail",
            "user_message",
        ]
    )


async def test_output_vars_list_returns_subset():
    """B1.8: ``options.output_vars=[names]`` returns only those context keys."""
    result = await _generate(
        INPUT_RAILS_CONFIG,
        [{"role": "user", "content": "modify input"}],
        {"output_vars": ["user_message", "last_user_message"]},
        main_output="x",
    )

    assert result.output_data == snapshot({"user_message": "modified input", "last_user_message": "modified input"})


async def test_log_internal_events_populated():
    """B1.12: ``options.log.internal_events=True`` attaches the internal event stream."""
    result = await _generate(
        OUTPUT_RAILS_CONFIG, [{"role": "user", "content": "hi"}], {"log": {"internal_events": True}}, main_output="safe"
    )

    assert result.log is not None
    assert result.log.internal_events is not None
    event_types = [event.get("type") for event in result.log.internal_events]
    assert event_types == snapshot(
        [
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "UserMessage",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "BotMessage",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "StartOutputRails",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "StartOutputRail",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "OutputRailFinished",
            "ContextUpdate",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "OutputRailsFinished",
            "StartInternalSystemAction",
            "InternalSystemActionFinished",
            "StartUtteranceBotAction",
            "Listen",
        ]
    )


async def test_retrieval_rail_disabled_by_options_skips_retrieval_rail():
    """B1.5: ``options.rails.retrieval=False`` skips configured retrieval rails."""
    options = {
        "output_vars": ["relevant_chunks"],
        "log": {"activated_rails": True},
    }
    default = await _generate(
        RETRIEVAL_RAILS_CONFIG,
        [{"role": "user", "content": "hi"}],
        options,
        main_output="  ask retrieval",
    )
    disabled = await _generate(
        RETRIEVAL_RAILS_CONFIG,
        [{"role": "user", "content": "hi"}],
        {**options, "rails": {"retrieval": False}},
        main_output="  ask retrieval",
    )

    assert default.output_data == {"relevant_chunks": "retrieval rail ran"}
    assert disabled.output_data == {"relevant_chunks": "\n"}


async def test_colang_history_log_matches_explain_and_llm_summary(capsys):
    """A6.3 / B1.13: Colang history and the LLM-call summary are exposed consistently."""
    rails = LLMRails(load_config(OUTPUT_RAILS_CONFIG), llm=FakeLLMModel(responses=["safe"]), verbose=False)
    result = await rails.generate_async(
        messages=[{"role": "user", "content": "hi"}],
        options={"log": {"colang_history": True}},
    )

    assert isinstance(result, GenerationResponse)
    assert result.log is not None
    assert result.log.colang_history
    explain = rails.explain()
    assert result.log.colang_history == explain.colang_history
    assert len(explain.llm_calls) == 1

    explain.print_llm_calls_summary()
    summary = capsys.readouterr().out
    assert "Summary: 1 LLM call(s)" in summary
    assert "Task `general`" in summary


async def test_input_rails_name_list_behaves_like_true_in_colang_1():
    """B1.3: a name-list for ``options.rails.input`` does NOT select a subset in the Colang 1.0
    ``generate_async`` path — it behaves like ``True`` (all configured input rails run).

    FINDING (pre-existing, not a refactor regression): ``llm_flows.co`` uses
    ``$generation_options.rails.input`` only as a truthiness gate and then runs all of
    ``$config.rails.input.flows``. The ``GenerationRailsOptions`` docstring's "only the
    specified rails will be applied" subset semantics is not wired for this path. Pinned so the
    decomposition preserves it; the doc/impl gap is flagged separately.
    """
    message = [{"role": "user", "content": "block second"}]
    log_options = {"log": {"activated_rails": True}}

    # Select only "first input rail" — yet "second input rail" still blocks "block second".
    subset = await _generate(
        TWO_INPUT_RAILS_CONFIG,
        message,
        {"rails": {"input": ["first input rail"]}, **log_options},
        main_output="fake out",
    )
    run_all = await _generate(TWO_INPUT_RAILS_CONFIG, message, log_options, main_output="fake out")

    assert normalize_generation_response(subset) == normalize_generation_response(run_all)
    assert normalize_generation_response(subset) == snapshot(
        {
            "response": [{"role": "assistant", "content": "I'm sorry, I can't respond to that."}],
            "activated_rails": [
                {"type": "input", "name": "first input rail", "decisions": [], "stop": False},
                {
                    "type": "input",
                    "name": "second input rail",
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
