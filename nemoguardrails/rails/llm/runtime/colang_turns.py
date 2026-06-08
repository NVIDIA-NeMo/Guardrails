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
import logging
import time
from typing import Any, List, Optional, Protocol, Tuple, Union

from nemoguardrails.actions.llm.utils import get_colang_history
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.context import llm_stats_var
from nemoguardrails.logging.stats import LLMStats

log = logging.getLogger(__name__)

process_events_semaphore = asyncio.Semaphore(1)

__all__ = [
    "ColangTurnRails",
    "generate_colang_events",
    "process_colang_events",
    "process_events_semaphore",
]


class ColangTurnRails(Protocol):
    @property
    def config(self) -> Any: ...

    @property
    def runtime(self) -> Any: ...

    @property
    def verbose(self) -> bool: ...


async def generate_colang_events(rails: ColangTurnRails, events: List[dict]) -> List[dict]:
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
    rails: ColangTurnRails,
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
