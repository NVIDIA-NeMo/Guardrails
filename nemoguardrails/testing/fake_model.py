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

"""Framework-agnostic fake LLM model for testing guardrails configurations.

This module exposes :class:`FakeLLMModel`, a lightweight implementation of the
``LLMModel`` protocol used by NeMo Guardrails. It is intended for use in tests
where a deterministic, scripted set of responses is preferable to calling out
to a real model provider.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Dict, List, Optional

from nemoguardrails.types import LLMResponse, LLMResponseChunk, UsageInfo


class FakeLLMModel:
    """Framework-agnostic fake LLM for testing. Implements the ``LLMModel`` protocol.

    Args:
        responses: A list of plain string responses. Each call to
            :meth:`generate_async` (or :meth:`stream_async`) consumes the next
            entry. Mutually exclusive with ``llm_responses``.
        llm_responses: A list of :class:`~nemoguardrails.types.LLMResponse`
            objects. Useful when tool calls or structured fields need to be
            asserted. Takes precedence over ``responses`` when provided.
        llm_exception: An exception instance to raise on every generation,
            useful for exercising error-handling paths.
        token_usage: Optional list of token usage dictionaries (one per
            response). Each entry may include ``prompt_tokens``,
            ``completion_tokens`` and ``total_tokens`` keys.
        should_return_token_usage: When ``True``, populate
            :attr:`LLMResponse.usage` from ``token_usage``.
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        llm_responses: Optional[List[LLMResponse]] = None,
        llm_exception: Optional[Exception] = None,
        token_usage: Optional[List[Dict[str, int]]] = None,
        should_return_token_usage: bool = False,
    ):
        if llm_responses is not None:
            self._llm_responses = llm_responses
        elif responses is not None:
            self._llm_responses = [LLMResponse(content=response) for response in responses]
        else:
            self._llm_responses = []
        self.responses = responses or [response.content for response in self._llm_responses]
        self.inference_count = 0
        self.llm_exception = llm_exception
        self.token_usage = token_usage
        self.should_return_token_usage = should_return_token_usage

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def provider_name(self) -> Optional[str]:
        return "test"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    def _next_response(self) -> LLMResponse:
        if self.llm_exception:
            raise self.llm_exception
        if self.inference_count >= len(self._llm_responses):
            raise RuntimeError(
                f"No responses available for query number {self.inference_count + 1} in FakeLLMModel. "
                "Most likely, too many LLM calls are made or additional responses need to be provided."
            )
        response = self._llm_responses[self.inference_count]
        self.inference_count += 1
        return response

    def _get_usage(self) -> Optional[UsageInfo]:
        idx = self.inference_count - 1
        if self.token_usage and self.should_return_token_usage and 0 <= idx < len(self.token_usage):
            usage = self.token_usage[idx]
            return UsageInfo(
                input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
                total_tokens=usage.get("total_tokens", 0),
            )
        return None

    async def generate_async(self, prompt, *, stop=None, **kwargs) -> LLMResponse:
        response = copy.copy(self._next_response())
        usage = self._get_usage()
        if usage:
            response.usage = usage
        return response

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        response = self._next_response()
        text = response.content or ""
        chunks = text.split(" ")
        for chunk_index, chunk in enumerate(chunks):
            content = chunk + " " if chunk_index < len(chunks) - 1 else chunk
            await asyncio.sleep(0)
            yield LLMResponseChunk(delta_content=content)
        # Final yield point so concurrent consumers (asyncio.create_task) can
        # process the last chunk before the caller continues after the async for.
        await asyncio.sleep(0)
