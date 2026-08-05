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

"""NeMo Guardrails Toolkit."""

import os
from importlib.metadata import version

# If no explicit value is set for TOKENIZERS_PARALLELISM, we disable it
# to get rid of the annoying warning.
if not os.environ.get("TOKENIZERS_PARALLELISM"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


import importlib
import warnings
from typing import TYPE_CHECKING

import nemoguardrails.patch_asyncio

nemoguardrails.patch_asyncio.apply()

# Ignore a warning message from torch.
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Use Guardrails top-level if this environment variable is set
_use_guardrails_wrapper = os.environ.get("NEMO_GUARDRAILS_IORAILS_ENGINE", "").lower() in (
    "true",
    "1",
    "yes",
)

# Public names are resolved lazily on attribute access (PEP 562) so that a bare
# `import nemoguardrails` does not eagerly boot the entire runtime (Colang
# parsers, tracing/OpenTelemetry, aiohttp, jinja2, ...). Resolution is not cached
# into module globals so that `importlib.reload(nemoguardrails)` re-reads the
# NEMO_GUARDRAILS_IORAILS_ENGINE alias; after the first access the underlying
# module is already in `sys.modules`, so repeated lookups are cheap.
_LAZY_ATTRS = {
    "RailsConfig": "nemoguardrails.rails.llm.config",
    "LLMRails": "nemoguardrails.rails.llm.llmrails",
    "Guardrails": "nemoguardrails.guardrails.guardrails",
    "get_default_framework": "nemoguardrails.llm.frameworks",
    "register_framework": "nemoguardrails.llm.frameworks",
    "set_default_framework": "nemoguardrails.llm.frameworks",
    "register_provider": "nemoguardrails.llm.providers",
    "ChatMessage": "nemoguardrails.types",
    "FinishReason": "nemoguardrails.types",
    "LLMFramework": "nemoguardrails.types",
    "LLMModel": "nemoguardrails.types",
    "LLMResponse": "nemoguardrails.types",
    "LLMResponseChunk": "nemoguardrails.types",
    "Role": "nemoguardrails.types",
    "ToolCall": "nemoguardrails.types",
    "ToolCallFunction": "nemoguardrails.types",
    "UsageInfo": "nemoguardrails.types",
}

if TYPE_CHECKING:
    # Give type checkers and IDEs the real symbols even though they are loaded
    # lazily at runtime.
    from nemoguardrails.guardrails.guardrails import Guardrails
    from nemoguardrails.llm.frameworks import (
        get_default_framework,
        register_framework,
        set_default_framework,
    )
    from nemoguardrails.llm.providers import register_provider
    from nemoguardrails.rails import LLMRails, RailsConfig
    from nemoguardrails.types import (
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


def __getattr__(name: str):
    # When NEMO_GUARDRAILS_IORAILS_ENGINE is set, LLMRails is an alias for
    # Guardrails for backwards-compatibility.
    if name == "LLMRails" and _use_guardrails_wrapper:
        module_name, attr = "nemoguardrails.guardrails.guardrails", "Guardrails"
    else:
        module_name = _LAZY_ATTRS.get(name)
        if module_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        attr = name
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def __dir__():
    return sorted(set(globals()) | set(__all__))


__version__ = version("nemoguardrails")
__all__ = [
    "ChatMessage",
    "FinishReason",
    "Guardrails",
    "LLMFramework",
    "LLMModel",
    "LLMRails",
    "LLMResponse",
    "LLMResponseChunk",
    "RailsConfig",
    "Role",
    "ToolCall",
    "ToolCallFunction",
    "UsageInfo",
    "get_default_framework",
    "register_framework",
    "register_provider",
    "set_default_framework",
]
