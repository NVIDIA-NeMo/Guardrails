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

"""Generation token stream lifecycle helpers."""

import asyncio
import json
import logging
import warnings
from typing import Any, AsyncIterator, Optional, Protocol, Union, cast

from nemoguardrails.exceptions import StreamingNotSupportedError
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.rails.llm.streaming.streaming_output_rails import (
    StreamingOutputRails,
    run_output_rails_in_streaming,
)
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.utils import extract_error_json

log = logging.getLogger(__name__)

__all__ = [
    "GenerationStreamRails",
    "generation_token_stream",
    "validate_streaming_with_output_rails",
]


class GenerationStreamRails(StreamingOutputRails, Protocol):
    async def generate_async(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Any] = None,
        streaming_handler: Optional[StreamingHandler] = None,
    ) -> object: ...


def validate_streaming_with_output_rails(config: Any) -> None:
    if len(config.rails.output.flows) > 0 and (
        not config.rails.output.streaming or not config.rails.output.streaming.enabled
    ):
        raise StreamingNotSupportedError(
            "stream_async() cannot be used when output rails are configured but "
            "rails.output.streaming.enabled is False. Either set "
            "rails.output.streaming.enabled to True in your configuration, or use "
            "generate_async() instead of stream_async()."
        )


def generation_token_stream(
    rails: GenerationStreamRails,
    *,
    prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    options: Optional[Union[dict, GenerationOptions]] = None,
    state: Optional[Any] = None,
    include_metadata: Optional[bool] = False,
    generator: Optional[AsyncIterator[str]] = None,
    include_generation_metadata: Optional[bool] = None,
) -> AsyncIterator[Union[str, dict]]:
    """Return the token stream for a generation request."""
    if include_generation_metadata is not None:
        warnings.warn(
            "include_generation_metadata is deprecated, use include_metadata instead. "
            "It will be removed in version 0.22.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        include_metadata = include_generation_metadata

    validate_streaming_with_output_rails(rails.config)

    if generator:
        if rails.config.rails.output.streaming and rails.config.rails.output.streaming.enabled:
            return run_output_rails_in_streaming(
                rails,
                streaming_handler=generator,
                output_rails_streaming_config=rails.config.rails.output.streaming,
                messages=messages,
                prompt=prompt,
            )
        return generator

    rails._explain_info = rails._ensure_explain_info()

    streaming_handler = StreamingHandler(include_metadata=include_metadata)

    async def _generation_task():
        try:
            await rails.generate_async(
                prompt=prompt,
                messages=messages,
                streaming_handler=streaming_handler,
                options=options,
                state=state,
            )
        except Exception as e:
            # If an exception occurs during generation, push it to the streaming
            # handler as a json string. This ensures the streaming pipeline is
            # properly terminated.
            log.error(f"Error in generation task: {e}", exc_info=True)
            error_message = str(e)
            error_dict = extract_error_json(error_message)
            error_payload = json.dumps(error_dict)
            await streaming_handler.push_chunk(error_payload)
            await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore

    task = asyncio.create_task(_generation_task())
    _track_generation_task(rails, task)

    output_rail_streaming_enabled = bool(
        rails.config.rails.output.streaming and rails.config.rails.output.streaming.enabled
    )
    if output_rail_streaming_enabled:
        base_iterator = run_output_rails_in_streaming(
            rails,
            streaming_handler=streaming_handler,
            output_rails_streaming_config=rails.config.rails.output.streaming,
            messages=messages,
            prompt=prompt,
        )
    else:
        base_iterator = streaming_handler

    async def wrapped_iterator():
        try:
            async for chunk in base_iterator:
                if chunk is not None:
                    yield chunk
        finally:
            await task

    return wrapped_iterator()


def _track_generation_task(rails: GenerationStreamRails, task: asyncio.Task) -> None:
    """Track background stream tasks so they are not garbage collected."""
    task_holder = cast(Any, rails)
    if not hasattr(task_holder, "_active_tasks"):
        task_holder._active_tasks = set()
    task_holder._active_tasks.add(task)

    def task_done_callback(task):
        task_holder._active_tasks.discard(task)

    task.add_done_callback(task_done_callback)
