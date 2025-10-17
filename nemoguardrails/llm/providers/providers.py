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

"""Module that exposes all the supported LLM providers.

Currently, this module automatically discovers all the LLM providers available in LangChain
and registers them.

Additional providers can be registered using the `register_llm_provider` function.
"""

import asyncio
import importlib
import logging
import warnings
from typing import Dict, List, Set, Type

from langchain.chat_models.base import BaseChatModel
from langchain_community import llms
from langchain_community.chat_models import _module_lookup
from langchain_core.language_models.llms import BaseLLM

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


def discover_langchain_providers():
    """Automatically discover all LLM providers from LangChain.

    This function is deprecated and will be removed in a future release.
    Use `_discover_langchain_providers` directly instead.
    """
    import warnings

    warnings.warn(
        "The `discover_langchain_providers` function is deprecated and will be removed in v0.15.0 release"
        "Please use `get_model_provider` directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    _discover_langchain_community_llm_providers()


DEPRECATED_LLM_PROVIDERS = ["mlflow-chat", "databricks-chat"]


def _discover_langchain_community_llm_providers():
    """Automatically discover all LLM providers from LangChain."""
    # To deal with deprecated stuff and avoid warnings, we compose the type_to_cls_dict here
    if hasattr(llms, "get_type_to_cls_dict"):
        type_to_cls_dict = {
            k: v()
            for k, v in llms.get_type_to_cls_dict().items()
            # Exclude deprecated ones
            if k not in DEPRECATED_LLM_PROVIDERS
        }
    else:
        type_to_cls_dict = llms.type_to_cls_dict

    return type_to_cls_dict


# this is needed as we perform the mapping in langchain_initializer.py
_CUSTOM_CHAT_PROVIDERS = {"nim"}


def _discover_langchain_partner_chat_providers() -> Set[str]:
    from langchain.chat_models.base import _SUPPORTED_PROVIDERS

    return _SUPPORTED_PROVIDERS | _CUSTOM_CHAT_PROVIDERS


def _discover_langchain_community_chat_providers():
    """Creates a mapping from provider name to chat model class.
    The provider name is defined as the last segment of the module path.
    For example, for module path "langchain_community.chat_models.google_palm",
    the provider name is "google_palm".
    """

    mapping = {}
    for class_name, module_path in _module_lookup.items():
        # extract provider name (we assume it is the last segment after the dot)
        provider_name = module_path.split(".")[-1]

        module = importlib.import_module(module_path)
        class_ = getattr(module, class_name)
        if provider_name in mapping:
            log.debug(
                f"Duplicate provider mapping for '{provider_name}': "
                f"existing class {mapping[provider_name]} vs new class {class_}"
            )
        mapping[provider_name] = class_

    return mapping


async def _acall(self, *args, **kwargs):
    """Adds asynchronous support to LLM providers that only have synchronous methods."""
    return await asyncio.to_thread(self._call, *args, **kwargs)


def _patch_acall_method_to(llm_providers: Dict[str, Type[BaseLLM]]):
    for provider_cls in llm_providers.values():
        # If the "_acall" method is not defined, we add it.
        if (
            provider_cls
            and issubclass(provider_cls, BaseLLM)
            and "_acall" not in provider_cls.__dict__
        ):
            log.debug("Adding async support to %s", provider_cls.__name__)
            setattr(provider_cls, "_acall", _acall)


# Initialize the providers with the default ones
_llm_providers: Dict[str, Type[BaseLLM]] = {
    "trt_llm": TRTLLM,
}

_chat_providers: Dict[str, Type[BaseChatModel]]

_llm_providers.update(_discover_langchain_community_llm_providers())
_patch_acall_method_to(_llm_providers)
_chat_providers = _discover_langchain_community_chat_providers()


def register_llm_provider(name: str, provider_cls: Type[BaseLLM]):
    """Register an additional LLM provider."""
    if not hasattr(provider_cls, "_acall"):
        raise TypeError(
            f"The provider class {provider_cls.__name__} must implement an '_acall' method."
        )
    _llm_providers[name] = provider_cls


def register_chat_provider(name: str, provider_cls: Type[BaseChatModel]):
    """Register an additional chat provider."""
    _chat_providers[name] = provider_cls


def get_llm_provider_names() -> List[str]:
    """Returns the list of supported LLM providers."""
    return list(sorted(list(_llm_providers.keys())))


def get_community_chat_provider_names() -> List[str]:
    """Returns the list of supported chat providers."""
    return list(sorted(list(_chat_providers.keys())))


def _get_all_chat_provider_names() -> List[str]:
    """Consolidates all chat provider names."""

    return list(_chat_providers.keys() | _discover_langchain_partner_chat_providers())


def get_chat_provider_names() -> List[str]:
    """Returns the list of supported chat providers."""
    return list(sorted(_get_all_chat_provider_names()))


def _get_text_completion_provider(provider_name: str) -> Type[BaseLLM]:
    if provider_name not in _llm_providers:
        raise RuntimeError(f"Could not find LLM provider '{provider_name}'")

    return _llm_providers[provider_name]


def _get_chat_completion_provider(provider_name: str) -> Type[BaseChatModel]:
    if provider_name not in _chat_providers:
        raise RuntimeError(f"Could not find chat provider '{provider_name}'")

    return _chat_providers[provider_name]


def _parse_version(version_str):
    return tuple(map(int, (version_str.split("."))))


__all__ = [
    "_llm_providers",
    "_parse_version",
    "get_llm_provider_names",
    "get_community_chat_provider_names",
    "get_chat_provider_names",
    "register_llm_provider",
    "register_chat_provider",
]
