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

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import StreamingHandler

streaming_handler_var: contextvars.ContextVar[Optional[StreamingHandler]] = contextvars.ContextVar(
    "streaming_handler", default=None
)
if TYPE_CHECKING:
    from nemoguardrails.logging.explain import ExplainInfo
    from nemoguardrails.logging.stats import LLMStats
    from nemoguardrails.rails.llm.options import GenerationOptions
    from nemoguardrails.streaming import StreamingHandler

streaming_handler_var: contextvars.ContextVar[Optional["StreamingHandler"]] = contextvars.ContextVar(
    "streaming_handler", default=None
)

# The object that holds additional explanation information.
explain_info_var: contextvars.ContextVar[Optional["ExplainInfo"]] = contextvars.ContextVar("explain_info", default=None)

# The current LLM call.
llm_call_info_var: contextvars.ContextVar[Optional[LLMCallInfo]] = contextvars.ContextVar("llm_call_info", default=None)

# All the generation options applicable to the current context.
generation_options_var: contextvars.ContextVar[Optional[GenerationOptions]] = contextvars.ContextVar(
    "generation_options", default=None
)

# The stats about the LLM calls.
llm_stats_var: contextvars.ContextVar[Optional["LLMStats"]] = contextvars.ContextVar("llm_stats", default=None)

# The raw LLM request that comes from the user.
# This is used in passthrough mode.
raw_llm_request: contextvars.ContextVar[Optional[Union[str, List[Dict[str, Any]]]]] = contextvars.ContextVar(
    "raw_llm_request", default=None
)

reasoning_trace_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("reasoning_trace", default=None)

# The tool calls from the current LLM response.
tool_calls_var: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("tool_calls", default=None)

# The response metadata from the current LLM response.
llm_response_metadata_var: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "llm_response_metadata", default=None
)

# The ID of the conversation the current request belongs to, as the caller names it (for example
# a chat thread ID). Set it with ``conversation_id()``; rails and actions read it with
# ``get_conversation_id()``.
conversation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("conversation_id", default=None)


@contextmanager
def conversation_id(value: Optional[str]) -> Iterator[None]:
    """Set the conversation ID that rails and actions see for requests made inside the block.

    NeMo Guardrails does not interpret the ID or pass it to models. A rail that reports to an
    external service can use it to group the checks of one conversation. ``None`` clears the ID
    inside the block, and the previous value is restored on exit.

    Example::

        with conversation_id(thread_id):
            response = await rails.generate_async(messages=messages)
    """
    token = conversation_id_var.set(value)
    try:
        yield
    finally:
        conversation_id_var.reset(token)


def get_conversation_id() -> Optional[str]:
    """The conversation ID of the current request, or ``None`` when the caller set none."""
    return conversation_id_var.get()
