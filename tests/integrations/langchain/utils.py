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

import asyncio
from typing import Any, Dict, List, Mapping, Optional, Union
from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import LLM
from langchain_core.messages import AIMessage


class FakeLLM(LLM):
    responses: List
    i: int = 0
    streaming: bool = False
    exception: Optional[Exception] = None
    token_usage: Optional[List[Dict[str, int]]] = None
    should_return_token_usage: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake-list"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        if self.exception:
            raise self.exception
        if self.i >= len(self.responses):
            raise RuntimeError(f"No responses available for query number {self.i + 1} in FakeLLM.")
        response = self.responses[self.i]
        self.i += 1
        return response

    async def _acall(self, prompt, stop=None, run_manager=None, **kwargs):
        if self.exception:
            raise self.exception
        if self.i >= len(self.responses):
            raise RuntimeError(f"No responses available for query number {self.i + 1} in FakeLLM.")
        response = self.responses[self.i]
        self.i += 1
        return response

    async def _astream(self, prompt, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import GenerationChunk

        if self.i >= len(self.responses):
            raise RuntimeError(f"No responses available for query number {self.i + 1} in FakeLLM.")
        response = self.responses[self.i]
        self.i += 1
        if self.exception:
            raise self.exception
        chunks = response.split(" ")
        for j in range(len(chunks)):
            chunk = chunks[j] + " " if j < len(chunks) - 1 else chunks[j]
            await asyncio.sleep(0.05)
            yield GenerationChunk(text=chunk)

    def _get_token_usage_for_response(self, response_index):
        llm_output = {}
        if self.token_usage and 0 <= response_index < len(self.token_usage) and self.should_return_token_usage:
            llm_output = {"token_usage": self.token_usage[response_index]}
        return llm_output

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import Generation, LLMResult

        generations = [[Generation(text=self._call(prompt, stop, run_manager, **kwargs))] for prompt in prompts]
        llm_output = self._get_token_usage_for_response(self.i - 1)
        return LLMResult(generations=generations, llm_output=llm_output)

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import Generation, LLMResult

        generations = [[Generation(text=await self._acall(prompt, stop, run_manager, **kwargs))] for prompt in prompts]
        llm_output = self._get_token_usage_for_response(self.i - 1)
        return LLMResult(generations=generations, llm_output=llm_output)

    async def ainvoke(self, input, config=None, *, stop=None, **kwargs):
        from langchain_core.messages import AIMessage

        text = await self._acall(str(input), stop)
        token_usage_data = self._get_token_usage_for_response(self.i - 1)
        response_metadata = {}
        if token_usage_data:
            response_metadata = token_usage_data
        return AIMessage(content=text, response_metadata=response_metadata)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}


def get_bound_llm_magic_mock(ainvoke_return_value: Union[AIMessage, dict]) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.return_value = mock_llm

    bound_llm_mock = AsyncMock()
    if isinstance(ainvoke_return_value, dict):
        bound_llm_mock.ainvoke.return_value = MagicMock(**ainvoke_return_value)
    else:
        bound_llm_mock.ainvoke.return_value = ainvoke_return_value

    mock_llm.bind.return_value = bound_llm_mock
    if isinstance(ainvoke_return_value, dict):
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(**ainvoke_return_value))
    else:
        mock_llm.ainvoke = AsyncMock(return_value=ainvoke_return_value)
    return mock_llm
