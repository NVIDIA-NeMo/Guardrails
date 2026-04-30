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

"""0.21 -> 0.22 LangChain config migration helper.

Detects LangChain Python-side flags in ``model.parameters`` when the
default framework is active and raises a clear error at LLMRails
construction, so a stale 0.21 LangChain config surfaces during init
rather than as an opaque HTTP 400 deep in a guardrail call.

Remove in 0.23.0. After 0.23 any unrecognized parameter is forwarded
verbatim to the OpenAI-compatible HTTP client; the wire's HTTP 400 is
the user's signal to clean up.
"""

# TODO(0.23): delete this module along with its call site in
#   nemoguardrails.rails.llm.llmrails.LLMRails._init_llms.

from typing import Dict, Iterable, List, Tuple

# Python-side flags inherited by every LangChain BaseChatModel
# (ChatOpenAI / ChatNVIDIA / ChatAnthropic / ChatVertexAI / etc.). Sourced
# from langchain-core BaseChatModel and langchain-openai 1.1.x.
_LANGCHAIN_BASE_FLAGS = frozenset(
    {
        "streaming",
        "disable_streaming",
        "verbose",
        "cache",
        "callbacks",
        "tags",
        "metadata",
        "name",
        "model_kwargs",
    }
)

# Per-engine Python-side aliases: keys that LangChain's provider package
# accepts under one name but that the default framework expects under
# another. Only providers DefaultFramework actually serves over OpenAI-
# compatible HTTP need entries here.
_LANGCHAIN_PROVIDER_ALIASES: Dict[str, Dict[str, str]] = {
    "nim": {"nvidia_api_key": "api_key", "nvidia_base_url": "base_url"},
    "nvidia_ai_endpoints": {"nvidia_api_key": "api_key", "nvidia_base_url": "base_url"},
}


def _violations_for(model_type: str, model_engine: str, parameters: dict) -> List[Tuple[str, str]]:
    """Return a list of (model_type, action) tuples for one model."""
    out: List[Tuple[str, str]] = []
    aliases = _LANGCHAIN_PROVIDER_ALIASES.get(model_engine, {})
    for flag in sorted(_LANGCHAIN_BASE_FLAGS & set(parameters)):
        if flag == "model_kwargs":
            out.append((model_type, "unpack `model_kwargs` contents directly into `parameters`"))
        else:
            out.append((model_type, f"remove `{flag}`"))
    for old in sorted(aliases.keys() & set(parameters)):
        out.append((model_type, f"rename `{old}` to `{aliases[old]}`"))
    return out


def check_langchain_kwargs(models: Iterable, active_framework: str) -> None:
    """Raise ValueError if any model carries LangChain Python-side flags.

    No-op when the active framework is anything other than ``default``;
    LangChain-flavored kwargs are valid on the LangChain framework.
    """
    if active_framework != "default":
        return
    violations: List[Tuple[str, str]] = []
    for model in models:
        params = getattr(model, "parameters", None) or {}
        if not params:
            continue
        violations.extend(
            _violations_for(getattr(model, "type", ""), getattr(model, "engine", ""), params),
        )
    if not violations:
        return
    header = (
        "Your config has a LangChain-only flag in `parameters` that the default\nframework doesn't forward:"
        if len(violations) == 1
        else "Your config has LangChain-only flags in `parameters` that the default\nframework doesn't forward:"
    )
    body = "\n".join(f"  models[{model_type}]: {action}" for model_type, action in violations)
    raise ValueError(
        f"{header}\n\n"
        f"{body}\n\n"
        "To keep 0.21 LangChain behavior instead, set NEMOGUARDRAILS_LLM_FRAMEWORK=langchain.\n"
        "(Migration check; removed in 0.23.0.)"
    )
