# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Colang runtime event helpers."""

import asyncio
import json
import logging
import time
from typing import Any, List, Optional, Tuple, Union, cast

from nemoguardrails.actions.llm.utils import get_colang_history
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.context import llm_stats_var, streaming_handler_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.rails.llm.types import ColangTurnSurface
from nemoguardrails.streaming import END_OF_STREAM
from nemoguardrails.utils import extract_error_json

log = logging.getLogger(__name__)

process_events_semaphore = asyncio.Semaphore(1)

__all__ = [
    "ColangTurnSurface",
    "generate_colang_events",
    "process_colang_events",
    "process_events_semaphore",
    "run_colang_turn",
]


async def run_colang_turn(
    rails: ColangTurnSurface,
    events: List[dict],
    state: Any,
    processing_log: List[dict],
) -> List[dict]:
    """Run one Colang turn for events already converted from messages."""
    if rails.config.colang_version == "1.0":
        return await _run_colang_1_turn(rails, events, state, processing_log)
    return await _run_colang_2_turn(rails, events, state)


async def _run_colang_1_turn(
    rails: ColangTurnSurface,
    events: List[dict],
    state: Any,
    processing_log: List[dict],
) -> List[dict]:
    state_events = []
    if state:
        assert isinstance(state, dict)
        state_events = state["events"]

    try:
        return await rails.runtime.generate_events(
            state_events + events,
            processing_log=processing_log,
        )
    except Exception as e:
        log.error("Error in generate_async: %s", e, exc_info=True)
        streaming_handler = streaming_handler_var.get()
        if streaming_handler:
            error_message = str(e)
            error_dict = extract_error_json(error_message)
            error_payload: str = json.dumps(error_dict)
            await streaming_handler.push_chunk(error_payload)
            await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
        raise


async def _run_colang_2_turn(
    rails: ColangTurnSurface,
    events: List[dict],
    state: Any,
) -> List[dict]:
    instant_actions = ["UtteranceBotAction"]
    if rails.config.rails.actions.instant_actions is not None:
        instant_actions = rails.config.rails.actions.instant_actions

    runtime: RuntimeV2_x = cast(RuntimeV2_x, rails.runtime)
    new_events, _output_state = await runtime.process_events(
        events,
        state=state,
        instant_actions=instant_actions,
        blocking=True,
    )

    return new_events


async def generate_colang_events(rails: ColangTurnSurface, events: List[dict]) -> List[dict]:
    """Generate the next Colang 1.0 events for an event history."""
    t0 = time.time()

    # Initialize the LLM stats
    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    # Compute the new events.
    processing_log = []
    new_events = await rails.runtime.generate_events(events, processing_log=processing_log)

    # If logging is enabled, we log the conversation
    # TODO: add support for logging flag
    if rails.verbose:
        history = get_colang_history(events)
        log.info(f"Conversation history so far: \n{history}")

    log.info("--- :: Total processing took %.2f seconds." % (time.time() - t0))
    log.info("--- :: Stats: %s" % llm_stats)

    return new_events


async def process_colang_events(
    rails: ColangTurnSurface,
    events: List[dict],
    state: Union[Optional[dict], State] = None,
    blocking: bool = False,
    *,
    semaphore: asyncio.Semaphore,
) -> Tuple[List[dict], Union[dict, State]]:
    """Process Colang events in order and return output events plus state."""
    t0 = time.time()
    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    # Compute the new events.
    # We need to protect 'process_events' to be called only once at a time
    # TODO (cschueller): Why is this?
    async with semaphore:
        output_events, output_state = await rails.runtime.process_events(events, state, blocking)

    took = time.time() - t0
    # Small tweak, disable this when there were no events (or it was just too fast).
    if took > 0.1:
        log.info("--- :: Total processing took %.2f seconds." % took)
        log.info("--- :: Stats: %s" % llm_stats)

    return output_events, output_state
