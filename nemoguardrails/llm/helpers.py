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

from typing import List, Optional, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM


def get_llm_instance_wrapper(llm_instance: LLM, llm_type: str) -> Type[LLM]:
    """Wraps an LLM instance in a class that can be registered with LLMRails.

    This is useful to create specific types of LLMs using a generic LLM provider
    from HuggingFace, e.g., HuggingFacePipelineCompatible or HuggingFaceEndpoint.
    """

    class WrapperLLM(LLM):
        """The wrapper class needs to have defined any parameters we need to be set by NeMo Guardrails.

        Currently added only streaming and temperature.
        """

        streaming: Optional[bool] = False
        temperature: Optional[float] = 1.0

        @property
        def model_kwargs(self):
            """Return the model's kwargs.

            These are needed to allow changes to the arguments of the LLM calls.
            """
            if hasattr(llm_instance, "model_kwargs"):
                return llm_instance.model_kwargs  # type: ignore[attr-defined] (We check in line above)
            return {}

        @property
        def _llm_type(self) -> str:
            """Return type of llm.

            This type can be used to customize the prompts.
            """
            return llm_type

        def _modify_instance_kwargs(self):
            """Modify the parameters of the llm_instance with the attributes set for the wrapper.

            This will allow the actual LLM instance to use the parameters at generation.
            TODO: Make this function more generic if needed.
            """

            if hasattr(llm_instance, "model_kwargs"):
                model_kwargs = getattr(llm_instance, "model_kwargs")
                if isinstance(model_kwargs, dict):
                    model_kwargs["temperature"] = self.temperature
                    model_kwargs["streaming"] = self.streaming

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs,
        ) -> str:
            self._modify_instance_kwargs()
            return llm_instance._call(prompt, stop, run_manager, **kwargs)

        async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs,
        ) -> str:
            self._modify_instance_kwargs()
            return await llm_instance._acall(prompt, stop, run_manager, **kwargs)

    return WrapperLLM
