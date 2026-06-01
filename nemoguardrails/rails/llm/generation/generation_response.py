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

"""Generation response envelope construction."""

from dataclasses import dataclass
from typing import Any, List, Optional

from nemoguardrails.actions.llm.utils import (
    extract_bot_thinking_from_events,
    extract_tool_calls_from_events,
    get_and_clear_response_metadata_contextvar,
    get_colang_history,
)
from nemoguardrails.colang.v1_0.runtime.flows import compute_context
from nemoguardrails.logging.processing_log import compute_generation_log
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
)

__all__ = [
    "GenerationEventMetadata",
    "generation_event_metadata",
    "generation_response_from_colang_turn",
]


@dataclass
class GenerationEventMetadata:
    tool_calls: Optional[List[dict]]
    llm_metadata: Optional[dict]
    reasoning_content: Optional[str]


def generation_event_metadata(events: List[dict]) -> GenerationEventMetadata:
    """Extract response metadata carried by generated events/context vars."""
    return GenerationEventMetadata(
        tool_calls=extract_tool_calls_from_events(events),
        llm_metadata=get_and_clear_response_metadata_contextvar(),
        reasoning_content=extract_bot_thinking_from_events(events),
    )


def generation_response_from_colang_turn(
    *,
    colang_version: str,
    prompt: Optional[str],
    new_message: dict,
    events: List[dict],
    new_events: List[dict],
    processing_log: List[dict],
    gen_options: GenerationOptions,
    state: Any,
    output_state: Any,
    event_metadata: GenerationEventMetadata,
) -> GenerationResponse:
    """Build a GenerationResponse from a completed Colang turn."""
    if prompt:
        res = GenerationResponse(response=new_message["content"])
    else:
        res = GenerationResponse(response=[new_message])

    if event_metadata.reasoning_content:
        res.reasoning_content = event_metadata.reasoning_content

    if event_metadata.tool_calls:
        res.tool_calls = event_metadata.tool_calls

    if event_metadata.llm_metadata:
        res.llm_metadata = event_metadata.llm_metadata

    if colang_version == "1.0":
        _populate_colang_1_response(
            res=res,
            events=events,
            new_events=new_events,
            processing_log=processing_log,
            gen_options=gen_options,
        )
    else:
        _validate_colang_2_response_options(gen_options)

    if state is not None:
        res.state = output_state

    return res


def _populate_colang_1_response(
    *,
    res: GenerationResponse,
    events: List[dict],
    new_events: List[dict],
    processing_log: List[dict],
    gen_options: GenerationOptions,
) -> None:
    if gen_options.output_vars:
        context = compute_context(events)
        output_vars = gen_options.output_vars
        if isinstance(output_vars, list):
            # If we have only a selection of keys, we filter to only that.
            res.output_data = {k: context.get(k) for k in output_vars}
        else:
            # Otherwise, we return the full context
            res.output_data = context

    generation_log = compute_generation_log(processing_log)
    log_options = gen_options.log

    if log_options.activated_rails or log_options.llm_calls:
        res.log = GenerationLog()

        # We always include the stats
        res.log.stats = generation_log.stats

        if log_options.activated_rails:
            res.log.activated_rails = generation_log.activated_rails

        if log_options.llm_calls:
            res.log.llm_calls = []
            for activated_rail in generation_log.activated_rails:
                for executed_action in activated_rail.executed_actions:
                    res.log.llm_calls.extend(executed_action.llm_calls)

    if log_options.internal_events:
        if res.log is None:
            res.log = GenerationLog()

        res.log.internal_events = new_events

    if log_options.colang_history:
        if res.log is None:
            res.log = GenerationLog()

        res.log.colang_history = get_colang_history(events)

    if gen_options.llm_output:
        # Currently, we include the output from the generation LLM calls.
        for activated_rail in generation_log.activated_rails:
            if activated_rail.type == "generation":
                for executed_action in activated_rail.executed_actions:
                    for llm_call in executed_action.llm_calls:
                        res.llm_output = llm_call.raw_response


def _validate_colang_2_response_options(gen_options: GenerationOptions) -> None:
    if gen_options.output_vars:
        raise ValueError("The `output_vars` option is not supported for Colang 2.0 configurations.")

    log_options = gen_options.log
    if (
        log_options.activated_rails
        or log_options.llm_calls
        or log_options.internal_events
        or log_options.colang_history
    ):
        raise ValueError("The `log` option is not supported for Colang 2.0 configurations.")

    if gen_options.llm_output:
        raise ValueError("The `llm_output` option is not supported for Colang 2.0 configurations.")
