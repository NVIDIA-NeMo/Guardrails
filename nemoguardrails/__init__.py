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


import warnings

import nemoguardrails.patch_asyncio

# Import order matters: the `nemoguardrails.rails` package must be fully
# initialized before `nemoguardrails.guardrails.guardrails`. The latter
# pulls in `actions/llm/utils.py` -> `context.py` -> `rails.llm.options`,
# which re-enters `rails/__init__.py` and would otherwise circularly import
# names from a half-loaded `actions/llm/utils.py`.
# isort: off
from nemoguardrails.rails import RailsConfig
from nemoguardrails.guardrails.guardrails import Guardrails

# isort: on

nemoguardrails.patch_asyncio.apply()

# Ignore a warning message from torch.
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Use Guardrails top-level if this environment variable is set
_use_guardrails_wrapper = os.environ.get("NEMO_GUARDRAILS_IORAILS_ENGINE", "").lower() in (
    "true",
    "1",
    "yes",
)

if _use_guardrails_wrapper:
    # For backwards-compatibility, instantiate Guardrails instead of LLMRails.
    LLMRails = Guardrails
else:
    # Use the original LLMRails class
    from nemoguardrails.rails import LLMRails

from nemoguardrails.llm.frameworks import (  # noqa: E402
    get_default_framework,
    register_framework,
    set_default_framework,
)
from nemoguardrails.llm.providers import register_provider  # noqa: E402
from nemoguardrails.types import (  # noqa: E402
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
