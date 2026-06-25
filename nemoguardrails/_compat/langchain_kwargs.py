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

import re
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from nemoguardrails.llm.constants import AZURE_PROVIDERS

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import Model

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

_PROVIDER_PREFIXED_ALIAS = re.compile(r"^(?P<prefix>[a-zA-Z]\w*?)_(?P<canonical>api_key|base_url|api_base|endpoint)$")


def _canonical_name_for(matched_canonical: str) -> str:
    if matched_canonical == "api_key":
        return "api_key"
    return "base_url"


def _detect_provider_alias(name: str) -> Optional[str]:
    match = _PROVIDER_PREFIXED_ALIAS.fullmatch(name)
    if match is None:
        return None
    return _canonical_name_for(match.group("canonical"))


def _violations_for(model_type: str, engine: str, parameters: dict) -> List[Tuple[str, str]]:
    """Return a list of (model_type, action) tuples for one model."""
    out: List[Tuple[str, str]] = []
    for flag in sorted(_LANGCHAIN_BASE_FLAGS & set(parameters)):
        if flag == "model_kwargs":
            out.append((model_type, "unpack `model_kwargs` contents directly into `parameters`"))
        else:
            out.append((model_type, f"remove `{flag}`"))
    for name in sorted(parameters):
        if name in _LANGCHAIN_BASE_FLAGS:
            continue
        canonical = _detect_provider_alias(name)
        if canonical is None:
            continue
        if engine in AZURE_PROVIDERS and name == "azure_endpoint":
            continue
        out.append((model_type, f"rename `{name}` to `{canonical}`"))
    return out


def check_langchain_kwargs(models: "Iterable[Model]", active_framework: str) -> None:
    """Raise ValueError if any model carries LangChain Python-side flags.

    No-op when the active framework is anything other than ``default``;
    LangChain-flavored kwargs are valid on the LangChain framework.
    """
    if active_framework != "default":
        return
    violations: List[Tuple[str, str]] = []
    for model in models:
        if not model.parameters:
            continue
        violations.extend(_violations_for(model.type, model.engine, model.parameters))
    if not violations:
        return
    body = "\n".join(f"  models[{model_type}]: {action}" for model_type, action in violations)
    raise ValueError(
        "Your config uses 0.21-style LangChain conventions that the default framework\n"
        "doesn't forward:\n\n"
        f"{body}\n\n"
        "Two paths:\n"
        "  - Adapt to the default framework: apply the renames/removals above.\n"
        "    Only do this if your endpoint is OpenAI-compatible.\n"
        "  - Keep 0.21 LangChain behavior: set NEMOGUARDRAILS_LLM_FRAMEWORK=langchain.\n\n"
        "(Migration check; removed in 0.23.0.)"
    )
