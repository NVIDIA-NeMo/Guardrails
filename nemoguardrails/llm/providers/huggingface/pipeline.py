# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema.output import GenerationChunk
from langchain_community.llms import HuggingFacePipeline


class HuggingFacePipelineCompatible(HuggingFacePipeline):
    """
    Hackish way to add backward-compatibility functions to the Langchain class.
    TODO: Planning to add this fix directly to Langchain repo.
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Hackish way to perform a single llm call since Langchain dropped support
        """
        if not isinstance(prompt, str):
            raise ValueError(
                "Argument `prompt` is expected to be a string. Instead found "
                f"{type(prompt)}. If you want to run the LLM on multiple prompts, use "
                "`generate` instead."
            )

        # Streaming for NeMo Guardrails is not supported in sync calls.
        if self.model_kwargs and self.model_kwargs.get("streaming"):
            raise Exception(
                "Streaming mode not supported for HuggingFacePipeline in NeMo Guardrails!"
            )

        llm_result = self._generate(
            [prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return llm_result.generations[0][0].text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Hackish way to add async support
        """
        if not isinstance(prompt, str):
            raise ValueError(
                "Argument `prompt` is expected to be a string. Instead found "
                f"{type(prompt)}. If you want to run the LLM on multiple prompts, use "
                "`generate` instead."
            )

        # Handle streaming, if the flag is set
        if self.model_kwargs and self.model_kwargs.get("streaming"):
            # Retrieve the streamer object, needs to be set in model_kwargs
            streamer = self.model_kwargs.get("streamer")
            if not streamer:
                raise Exception(
                    "Cannot stream, please add HuggingFace streamer object to model_kwargs!"
                )

            loop = asyncio.get_running_loop()

            # Pass the asyncio loop to the stream so that it can send back
            # the chunks in the queue.
            streamer.loop = loop

            # Launch the generation in a separate task.
            generation_kwargs = dict(
                prompts=[prompt],
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            loop.create_task(self._agenerate(**generation_kwargs))

            # And start waiting for the chunks to come in.
            completion = ""
            async for item in streamer:
                completion += item
                chunk = GenerationChunk(text=item)
                if run_manager:
                    await run_manager.on_llm_new_token(item, chunk=chunk)

            return completion

        llm_result = await self._agenerate(
            [prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return llm_result.generations[0][0].text
