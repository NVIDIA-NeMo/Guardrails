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

"""Generation context helpers."""

from contextvars import Token
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_stats_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler

__all__ = [
    "GenerationRequestContext",
    "ensure_explain_info",
    "explain_info_for_current_context",
    "start_generation_request_context",
    "start_generation_stats",
]


@dataclass
class GenerationRequestContext:
    """Request-scoped generation context bindings."""

    explain_info: ExplainInfo
    _generation_options_token: Token
    _raw_llm_request_token: Token
    _explain_info_token: Token
    _streaming_handler_token: Optional[Token]
    _streaming_handler: Optional[StreamingHandler]
    _streaming_handler_closed: bool = False
    _closed: bool = False

    async def close_streaming_handler(self) -> None:
        """Close the request streaming handler once."""
        if self._streaming_handler and not self._streaming_handler_closed:
            self._streaming_handler_closed = True
            await self._streaming_handler.push_chunk(cast(Any, END_OF_STREAM))

    async def close(self) -> None:
        """Close request resources and restore previous context bindings."""
        if self._closed:
            return

        self._closed = True
        try:
            await self.close_streaming_handler()
        finally:
            if self._streaming_handler_token is not None:
                streaming_handler_var.reset(self._streaming_handler_token)
            explain_info_var.reset(self._explain_info_token)
            raw_llm_request.reset(self._raw_llm_request_token)
            generation_options_var.reset(self._generation_options_token)


def ensure_explain_info() -> ExplainInfo:
    """Ensure that an ExplainInfo object is present in the current context."""
    explain_info = explain_info_var.get()
    if explain_info is None:
        explain_info = ExplainInfo()
        explain_info_var.set(explain_info)

    return explain_info


def explain_info_for_current_context(fallback: Optional[ExplainInfo]) -> ExplainInfo:
    """Return request-scoped explain info when present, otherwise use fallback."""
    explain_info = explain_info_var.get()
    if explain_info is not None:
        return explain_info

    if fallback is not None:
        return fallback

    return ensure_explain_info()


def start_generation_request_context(
    *,
    gen_options: Optional[GenerationOptions],
    messages: Optional[List[Dict[str, Any]]],
    streaming_handler: Optional[StreamingHandler],
) -> GenerationRequestContext:
    """Bind request-scoped generation context and return its cleanup scope."""
    generation_options_token = generation_options_var.set(gen_options)

    streaming_handler_token = None
    if streaming_handler:
        streaming_handler_token = streaming_handler_var.set(streaming_handler)

    explain_info = explain_info_var.get()
    if explain_info is None:
        explain_info = ExplainInfo()
    explain_info_token = explain_info_var.set(explain_info)

    raw_llm_request_token = raw_llm_request.set(messages)

    return GenerationRequestContext(
        explain_info=explain_info,
        _generation_options_token=generation_options_token,
        _raw_llm_request_token=raw_llm_request_token,
        _explain_info_token=explain_info_token,
        _streaming_handler_token=streaming_handler_token,
        _streaming_handler=streaming_handler,
    )


def start_generation_stats() -> Tuple[LLMStats, List[dict]]:
    """Initialize LLM stats and processing log for a generation request."""
    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)
    return llm_stats, []
