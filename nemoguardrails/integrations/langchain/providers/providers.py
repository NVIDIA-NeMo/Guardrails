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

"""Module that exposes all the supported chat providers.

Currently, this module automatically discovers all the chat providers available in LangChain
and registers them.

Additional providers can be registered using the `register_chat_provider` function.
"""

import importlib
import logging
import warnings
from typing import Dict, List, Set, Type

from langchain_community.chat_models import _module_lookup
from langchain_core.language_models import BaseChatModel

# NOTE: this is temp
# Suppress specific warnings related to protected namespaces in Pydantic models, they must update their code.
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_.*" in .* has conflict with protected namespace "model_"',
    category=UserWarning,
    module=r"pydantic\._internal\._fields",
)

log = logging.getLogger(__name__)


# this is needed as we perform the mapping in langchain_initializer.py
_CUSTOM_CHAT_PROVIDERS = {"nim"}


def _discover_langchain_partner_chat_providers() -> Set[str]:
    import langchain.chat_models.base as _base

    # The internal variable listing supported providers was renamed across langchain versions:
    # _SUPPORTED_PROVIDERS (<=1.2.1, set) -> _SUPPORTED_PROVIDERS (1.2.1, dict) -> _BUILTIN_PROVIDERS (>=1.2.10, dict)
    _PROVIDERS = getattr(_base, "_BUILTIN_PROVIDERS", None) or getattr(_base, "_SUPPORTED_PROVIDERS", None)
    if _PROVIDERS is None:
        return _CUSTOM_CHAT_PROVIDERS

    if isinstance(_PROVIDERS, dict):
        return set(_PROVIDERS.keys()) | _CUSTOM_CHAT_PROVIDERS
    return _PROVIDERS | _CUSTOM_CHAT_PROVIDERS


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


_chat_providers: Dict[str, Type[BaseChatModel]]
_chat_providers = _discover_langchain_community_chat_providers()


def register_chat_provider(name: str, provider_cls: Type[BaseChatModel]):
    """Register an additional chat provider."""
    _chat_providers[name] = provider_cls


def get_community_chat_provider_names() -> List[str]:
    """Returns the list of supported chat providers."""
    return list(sorted(list(_chat_providers.keys())))


def _get_all_chat_provider_names() -> List[str]:
    """Consolidates all chat provider names."""

    return list(_chat_providers.keys() | _discover_langchain_partner_chat_providers())


def get_chat_provider_names() -> List[str]:
    """Returns the list of supported chat providers."""
    return list(sorted(_get_all_chat_provider_names()))


def _get_chat_completion_provider(provider_name: str) -> Type[BaseChatModel]:
    if provider_name not in _chat_providers:
        raise RuntimeError(f"Could not find chat provider '{provider_name}'")

    return _chat_providers[provider_name]


def _parse_version(version_str):
    return tuple(map(int, (version_str.split("."))))


__all__ = [
    "_parse_version",
    "get_community_chat_provider_names",
    "get_chat_provider_names",
    "register_chat_provider",
]
