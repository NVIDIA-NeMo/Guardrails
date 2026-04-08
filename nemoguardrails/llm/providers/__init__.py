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

import warnings
from typing import Any, List

from nemoguardrails.llm.frameworks import get_default_framework, get_framework


def _active_framework():
    return get_framework(get_default_framework())


def register_provider(name: str, provider_cls: Any) -> None:
    _active_framework().register_provider(name, provider_cls)


def get_provider_names() -> List[str]:
    return _active_framework().get_provider_names()


def register_chat_provider(name: str, provider_cls: Any) -> None:
    register_provider(name, provider_cls)


def register_llm_provider(name: str, provider_cls: Any) -> None:
    warnings.warn(
        "register_llm_provider is deprecated and will be removed in 0.23.0. "
        "Text completion providers are being removed. Use register_chat_provider "
        "or register_provider instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fw = _active_framework()
    if hasattr(fw, "register_llm_provider"):
        fw.register_llm_provider(name, provider_cls)  # type: ignore[attr-defined]
    else:
        fw.register_provider(name, provider_cls)


def get_chat_provider_names() -> List[str]:
    fw = _active_framework()
    if hasattr(fw, "get_chat_provider_names"):
        return fw.get_chat_provider_names()  # type: ignore[attr-defined]
    return fw.get_provider_names()


def get_llm_provider_names() -> List[str]:
    warnings.warn(
        "get_llm_provider_names is deprecated and will be removed in 0.23.0. "
        "Text completion providers are being removed. Use get_provider_names instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fw = _active_framework()
    if hasattr(fw, "get_llm_provider_names"):
        return fw.get_llm_provider_names()  # type: ignore[attr-defined]
    return fw.get_provider_names()


__all__ = [
    "register_provider",
    "get_provider_names",
    "register_chat_provider",
    "register_llm_provider",
    "get_chat_provider_names",
    "get_llm_provider_names",
]
