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

import importlib
from typing import TYPE_CHECKING

# Resolve exports lazily (PEP 562) through their narrowest modules so that
# importing `RailsConfig` does not eagerly initialize `LLMRails` (and its Colang
# runtime), and so that submodules such as `nemoguardrails.rails.llm.options`
# can initialize this package without pulling in the full runtime.
_LAZY_ATTRS = {
    "RailsConfig": "nemoguardrails.rails.llm.config",
    "LLMRails": "nemoguardrails.rails.llm.llmrails",
}

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import RailsConfig
    from nemoguardrails.rails.llm.llmrails import LLMRails


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = ["RailsConfig", "LLMRails"]
