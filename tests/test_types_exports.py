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

import pytest

PUBLIC_TYPE_NAMES = [
    "ChatMessage",
    "FinishReason",
    "LLMFramework",
    "LLMModel",
    "LLMResponse",
    "LLMResponseChunk",
    "Role",
    "ToolCall",
    "ToolCallFunction",
    "UsageInfo",
]


@pytest.mark.parametrize("name", PUBLIC_TYPE_NAMES)
def test_top_level_export(name):
    import nemoguardrails

    assert hasattr(nemoguardrails, name), f"nemoguardrails is missing {name}"
    assert name in nemoguardrails.__all__


@pytest.mark.parametrize("name", PUBLIC_TYPE_NAMES)
def test_llm_subpackage_export(name):
    import nemoguardrails.llm as llm

    assert hasattr(llm, name), f"nemoguardrails.llm is missing {name}"
    assert name in llm.__all__


@pytest.mark.parametrize("name", PUBLIC_TYPE_NAMES)
def test_top_level_and_llm_export_same_object(name):
    import nemoguardrails
    import nemoguardrails.llm as llm
    import nemoguardrails.types as types_mod

    canonical = getattr(types_mod, name)
    assert getattr(nemoguardrails, name) is canonical
    assert getattr(llm, name) is canonical


def test_direct_from_imports_top_level():
    from nemoguardrails import (  # noqa: F401
        ChatMessage,
        FinishReason,
        LLMFramework,
        LLMModel,
        LLMResponse,
        LLMResponseChunk,
        Role,
        ToolCall,
        ToolCallFunction,
        UsageInfo,
    )


def test_direct_from_imports_llm_subpackage():
    from nemoguardrails.llm import (  # noqa: F401
        ChatMessage,
        FinishReason,
        LLMFramework,
        LLMModel,
        LLMResponse,
        LLMResponseChunk,
        Role,
        ToolCall,
        ToolCallFunction,
        UsageInfo,
    )


# Registry functions and the submodule that owns each one.
REGISTRY_FUNCTIONS = [
    ("get_default_framework", "nemoguardrails.llm.frameworks"),
    ("register_framework", "nemoguardrails.llm.frameworks"),
    ("set_default_framework", "nemoguardrails.llm.frameworks"),
    ("register_provider", "nemoguardrails.llm.providers"),
]


@pytest.mark.parametrize("name,canonical_module", REGISTRY_FUNCTIONS)
def test_registry_function_top_level_export(name, canonical_module):
    import nemoguardrails

    assert hasattr(nemoguardrails, name), f"nemoguardrails is missing {name}"
    assert name in nemoguardrails.__all__


@pytest.mark.parametrize("name,canonical_module", REGISTRY_FUNCTIONS)
def test_registry_function_llm_subpackage_export(name, canonical_module):
    import nemoguardrails.llm as llm

    assert hasattr(llm, name), f"nemoguardrails.llm is missing {name}"
    assert name in llm.__all__


@pytest.mark.parametrize("name,canonical_module", REGISTRY_FUNCTIONS)
def test_registry_function_paths_resolve_to_same_object(name, canonical_module):
    import importlib

    import nemoguardrails
    import nemoguardrails.llm as llm

    canonical = getattr(importlib.import_module(canonical_module), name)
    assert getattr(nemoguardrails, name) is canonical
    assert getattr(llm, name) is canonical
