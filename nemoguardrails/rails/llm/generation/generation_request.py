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

"""Generation request normalization."""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.serialization import json_to_state
from nemoguardrails.rails.llm.options import GenerationOptions

__all__ = [
    "GenerationRequest",
    "PreparedGenerationRequest",
    "normalize_generation_request",
    "prepare_generation_request_for_runtime",
]


@dataclass
class GenerationRequest:
    prompt: Optional[str]
    messages: List[dict]
    options: Optional[GenerationOptions]
    state: Optional[Union[dict, State]]


@dataclass
class PreparedGenerationRequest:
    prompt: Optional[str]
    request_messages: List[dict]
    runtime_messages: List[dict]
    options: Optional[GenerationOptions]
    state: Optional[Union[dict, State]]
    needs_llm: bool


def normalize_generation_request(
    *,
    prompt: Optional[str],
    messages: Optional[List[dict]],
    options: Optional[Union[dict, GenerationOptions]],
    state: Optional[Union[dict, State]],
) -> GenerationRequest:
    """Normalize public generate inputs into the internal request shape."""
    if prompt is None and messages is None:
        raise ValueError("Either prompt or messages must be provided.")

    if prompt is not None and messages is not None:
        raise ValueError("Only one of prompt or messages can be provided.")

    if prompt is not None:
        messages = [{"role": "user", "content": prompt}]

    normalized_state = _normalize_generation_state(state)
    gen_options = _normalize_generation_options(options, state=normalized_state)

    return GenerationRequest(
        prompt=prompt,
        messages=messages or [],
        options=gen_options,
        state=normalized_state,
    )


def prepare_generation_request_for_runtime(
    *,
    prompt: Optional[str],
    messages: Optional[List[dict]],
    options: Optional[Union[dict, GenerationOptions]],
    state: Optional[Union[dict, State]],
) -> PreparedGenerationRequest:
    request = normalize_generation_request(
        prompt=prompt,
        messages=messages,
        options=options,
        state=state,
    )
    gen_options = request.options
    runtime_messages = request.messages

    if gen_options:
        runtime_messages = [
            {
                "role": "context",
                "content": {"generation_options": gen_options.model_dump()},
            }
        ] + (request.messages or [])

    if (
        runtime_messages
        and runtime_messages[-1]["role"] == "assistant"
        and gen_options
        and gen_options.rails.dialog is False
    ):
        runtime_messages[0]["content"]["bot_message"] = runtime_messages[-1]["content"]
        runtime_messages = runtime_messages[0:-1]

    needs_llm = gen_options is None or gen_options.rails.dialog is not False

    return PreparedGenerationRequest(
        prompt=request.prompt,
        request_messages=request.messages,
        runtime_messages=runtime_messages,
        options=gen_options,
        state=request.state,
        needs_llm=needs_llm,
    )


def _normalize_generation_state(
    state: Optional[Union[dict, State]],
) -> Optional[Union[dict, State]]:
    if isinstance(state, dict) and state.get("version", "1.0") == "2.x":
        return json_to_state(state["state"])
    return state


def _normalize_generation_options(
    options: Optional[Union[dict, GenerationOptions]],
    *,
    state: Optional[Any],
) -> Optional[GenerationOptions]:
    if state is not None:
        if options is None:
            return GenerationOptions()
        if isinstance(options, dict):
            return GenerationOptions(**options)
        return options

    if options and isinstance(options, dict):
        return GenerationOptions(**options)
    if isinstance(options, GenerationOptions):
        return options
    if options is None:
        return None

    raise TypeError("options must be a dict or GenerationOptions")
