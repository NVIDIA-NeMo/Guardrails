# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module that exposes all the supported LLM providers.

Currently, this module automatically discovers all the LLM providers available in LangChain
and registers them.

Additional providers can be registered using the `register_llm_provider` function.
"""

import asyncio
import logging
import warnings
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema.output import GenerationChunk
from langchain_community import llms
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.rails.llm.config import Model

from .nemollm import NeMoLLM
from .trtllm.llm import TRTLLM

# NOTE: this is temp
# Suppress specific warnings related to protected namespaces in Pydantic models, they must update their code.
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_.*" in .* has conflict with protected namespace "model_"',
    category=UserWarning,
    module=r"pydantic\._internal\._fields",
)
log = logging.getLogger(__name__)

# Initialize the providers with the default ones
# We set nvidia_ai_endpoints provider to None because it's only supported if `langchain_nvidia_ai_endpoints` is installed.
_providers: Dict[str, Optional[Type[BaseLLM]]] = {
    "nemollm": NeMoLLM,
    "trt_llm": TRTLLM,
    "nvidia_ai_endpoints": None,
    "nim": None,
}


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


async def _acall(self, *args, **kwargs):
    """Hackish way to add async support to LLM providers that don't have it.

    We just call the sync version of the function.
    """
    # TODO: run this in a thread pool!
    return self._call(*args, **kwargs)


def discover_langchain_providers():
    """Automatically discover all LLM providers from LangChain."""
    # To deal with deprecated stuff and avoid warnings, we compose the type_to_cls_dict here
    if hasattr(llms, "get_type_to_cls_dict"):
        type_to_cls_dict = {
            k: v()
            for k, v in llms.get_type_to_cls_dict().items()
            # Exclude deprecated ones
            if k not in ["mlflow-chat", "databricks-chat"]
        }
    else:
        type_to_cls_dict = llms.type_to_cls_dict

    _providers.update(type_to_cls_dict)

    # We make sure we have OpenAI from the right package.
    if "openai" in _providers:
        try:
            from langchain_openai import OpenAI

            del _providers["openai"]

            _providers["openai"] = OpenAI
        except ImportError:
            # If the `langchain_openai` package is not installed, the warning
            # will come from langchain.
            pass

    # We also do some monkey patching to make sure that all LLM providers have async support
    for provider_cls in _providers.values():
        # If the "_acall" method is not defined, we add it.
        if (
            provider_cls
            and issubclass(provider_cls, BaseLLM)
            and "_acall" not in provider_cls.__dict__
        ):
            log.debug("Adding async support to %s", provider_cls.__name__)
            provider_cls._acall = _acall


# Discover all the additional providers from LangChain
discover_langchain_providers()


def register_llm_provider(name: str, provider_cls: Type[BaseLLM]):
    """Register an additional LLM provider."""
    _providers[name] = provider_cls


def get_llm_provider(model_config: Model) -> Type[BaseLLM]:
    if model_config.engine not in _providers:
        raise RuntimeError(f"Could not find LLM provider '{model_config.engine}'")

    # For OpenAI, we use a different provider depending on whether it's a chat model or not
    if (
        model_config.engine == "openai"
        and ("gpt-3.5" in model_config.model or "gpt-4" in model_config.model)
        and "instruct" not in model_config.model
    ):
        try:
            from langchain_openai.chat_models import ChatOpenAI

            return ChatOpenAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai, please install it with "
                "`pip install langchain-openai`."
            )

    elif model_config.engine == "azure" and (
        "gpt-3.5" in model_config.model or "gpt-4" in model_config.model
    ):
        try:
            from langchain_openai.chat_models import AzureChatOpenAI

            return AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai, please install it with "
                "`pip install langchain-openai`."
            )
    elif model_config.engine == "nvidia_ai_endpoints" or model_config.engine == "nim":
        try:
            from ._langchain_nvidia_ai_endpoints_patch import ChatNVIDIA

            # Check the version
            package_version = version("langchain_nvidia_ai_endpoints")

            if _parse_version(package_version) < (0, 2, 0):
                raise ValueError(
                    "langchain_nvidia_ai_endpoints version must be 0.2.0 or above."
                    " Please upgrade it with `pip install langchain-nvidia-ai-endpoints --upgrade`."
                )
            return ChatNVIDIA

        except ImportError:
            raise ImportError(
                "Could not import langchain_nvidia_ai_endpoints, please install it with "
                "`pip install langchain-nvidia-ai-endpoints`."
            )

    elif model_config.engine == "vertexai":
        # To avoid a LangChainDeprecationWarning with the default langchain_community.llms.vertexai.VertexAI which  is
        # deprecated in langchain-community 0.0.12 and will be removed in 0.2.0.
        try:
            from langchain_google_vertexai import VertexAI

            return VertexAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_google_vertexai, please install it with "
                "`pip install langchain-google-vertexai`."
            )

    else:
        return _providers[model_config.engine]


def get_llm_provider_names() -> List[str]:
    """Returns the list of supported LLM providers."""
    return list(sorted(list(_providers.keys())))


def _parse_version(version_str):
    return tuple(map(int, (version_str.split("."))))
