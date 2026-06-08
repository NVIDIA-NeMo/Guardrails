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

"""Characterization of the parallel output-rail RailOutcome decision.

The parallel streaming output-rail path in
``nemoguardrails/colang/v1_0/runtime/runtime.py`` (``_run_output_rails_in_parallel_streaming``)
decides whether a chunk is blocked by reading ``RailOutcome.is_blocked`` from
the action result. These tests run fully offline using ``TestChat``.
"""

import asyncio
import json
from json.decoder import JSONDecodeError

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from tests.utils import TestChat


@action(is_system_action=True)
def rail_outcome_output_check(context=None, **params):
    bot_message_chunk = (context or {}).get("bot_message", "")
    if "HIGHSCORE" in bot_message_chunk:
        return RailOutcome.block()
    return RailOutcome.allow()


def _build_parallel_rail_outcome_config() -> RailsConfig:
    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": ["rail outcome output check"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": False,
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke

        define subflow rail outcome output check
          execute rail_outcome_output_check
        """,
    )


async def _stream_chunks_with_action(config, action_func, llm_completions):
    chat = TestChat(
        config,
        llm_completions=llm_completions,
        streaming=True,
    )
    chat.app.register_action(action_func)

    chunks = []
    async for chunk in chat.app.stream_async(messages=[{"role": "user", "content": "Hi!"}]):
        chunks.append(chunk)
    return chunks


def _error_chunks(chunks):
    errors = []
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "error" in parsed:
            errors.append(parsed)
    return errors


@pytest.mark.asyncio
async def test_parallel_output_rails_block_rail_outcome():
    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This response has a HIGHSCORE and must be blocked."',
    ]

    chunks = await _stream_chunks_with_action(
        _build_parallel_rail_outcome_config(),
        rail_outcome_output_check,
        llm_completions,
    )
    errors = _error_chunks(chunks)

    assert len(errors) == 1, f"Expected exactly one block error, got: {chunks}"
    error = errors[0]
    assert error["error"]["type"] == "guardrails_violation"
    assert error["error"]["code"] == "content_blocked"
    assert error["error"]["param"] == "rail outcome output check"

    response = "".join(chunks)
    assert "must be blocked" not in response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_output_rails_allow_rail_outcome():
    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This response stays well under the threshold and is allowed."',
    ]

    chunks = await _stream_chunks_with_action(
        _build_parallel_rail_outcome_config(),
        rail_outcome_output_check,
        llm_completions,
    )
    errors = _error_chunks(chunks)

    assert errors == [], f"Expected no block error, got: {errors}"

    response = "".join(chunks)
    assert "under the threshold and is allowed" in response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
