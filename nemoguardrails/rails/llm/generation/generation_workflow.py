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

import logging
import time
from typing import Optional, Union

from nemoguardrails.actions.llm.utils import get_colang_history
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.rails.llm.conversation.conversation_events import events_for_messages
from nemoguardrails.rails.llm.generation.bot_messages import bot_message_from_colang_events
from nemoguardrails.rails.llm.generation.generation_context import (
    GenerationRequestContext,
    start_generation_stats,
)
from nemoguardrails.rails.llm.generation.generation_response import (
    generation_event_metadata,
    generation_response_from_colang_turn,
)
from nemoguardrails.rails.llm.generation.generation_tracing import (
    export_generation_trace,
    prepare_generation_tracing,
    restore_generation_trace_log,
)
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.rails.llm.runtime.colang_turns import run_colang_turn
from nemoguardrails.rails.llm.types import StandardGenerationSurface
from nemoguardrails.rails.llm.utils import get_history_cache_key

__all__ = ["generate_standard_async"]

log = logging.getLogger(__name__)


async def generate_standard_async(
    rails: StandardGenerationSurface,
    *,
    prompt: Optional[str],
    messages: list[dict],
    gen_options: Optional[GenerationOptions],
    state: Optional[Union[dict, State]],
    request_context: GenerationRequestContext,
) -> Union[str, dict, GenerationResponse, tuple[dict, dict]]:
    rails._explain_info = request_context.explain_info

    t0 = time.time()
    llm_stats, processing_log = start_generation_stats()

    events = events_for_messages(rails, messages, state)

    new_events = await run_colang_turn(rails, events, state, processing_log)
    output_state = None

    bot_message = bot_message_from_colang_events(rails.config.colang_version, new_events)
    new_message = bot_message.message

    if rails.config.colang_version == "1.0":
        events.extend(new_events)
        events.extend(bot_message.extra_events)

        if state is None:
            cache_key = get_history_cache_key((messages) + [new_message])
            rails.events_history_cache[cache_key] = events
        else:
            output_state = {"events": events}

    rails._explain_info.colang_history = get_colang_history(events)
    if rails.verbose:
        log.info(f"Conversation history so far: \n{rails._explain_info.colang_history}")

    total_time = time.time() - t0
    log.info("--- :: Total processing took %.2f seconds. LLM Stats: %s" % (total_time, llm_stats))

    await request_context.close_streaming_handler()

    tracing_state = prepare_generation_tracing(
        tracing_enabled=rails.config.tracing.enabled,
        gen_options=gen_options,
    )
    gen_options = tracing_state.gen_options

    response_event_metadata = generation_event_metadata(new_events)
    if gen_options:
        res = generation_response_from_colang_turn(
            colang_version=rails.config.colang_version,
            prompt=prompt,
            new_message=new_message,
            events=events,
            new_events=new_events,
            processing_log=processing_log,
            gen_options=gen_options,
            state=state,
            output_state=output_state,
            event_metadata=response_event_metadata,
        )

        if rails.config.tracing.enabled:
            await export_generation_trace(
                tracing_config=rails.config.tracing,
                log_adapters=rails._log_adapters,
                messages=messages,
                response=res,
            )
            restore_generation_trace_log(
                response=res,
                original_log_options=tracing_state.original_log_options,
            )

        return res

    if response_event_metadata.reasoning_content:
        thinking_trace = f"<think>{response_event_metadata.reasoning_content}</think>\n"
        new_message["content"] = thinking_trace + new_message["content"]

    if prompt:
        return new_message["content"]

    if response_event_metadata.tool_calls:
        new_message["tool_calls"] = response_event_metadata.tool_calls
    return new_message
