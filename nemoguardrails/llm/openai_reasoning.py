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


from typing import Any, Dict

_REASONING_DROP_PARAMS = frozenset({"temperature", "stop"})


def is_openai_reasoning_model(model_name: str) -> bool:
    """True for OpenAI reasoning models (o1/o3/o4 series, gpt-5+).

    Used to decide whether to strip parameters that OpenAI rejects on these
    models (temperature, stop, etc.) and how to remap others
    (max_tokens -> max_completion_tokens). Shared by the DefaultFramework
    OpenAI client and the LangChain adapter so the two paths cannot drift.
    """
    name = model_name.lower()
    if name in ("o1", "o3", "o4") or name.startswith(("o1-", "o3-", "o4-")):
        return True
    if name == "gpt-5" or name.startswith("gpt-5-"):
        return "chat" not in name
    if name.startswith(("gpt-5.", "gpt-6")):
        return True
    return False


def apply_openai_reasoning_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unsupported params and remap max_tokens for OpenAI reasoning models.

    OpenAI reasoning models (o-series, gpt-5+) require ``max_completion_tokens``
    instead of ``max_tokens`` and reject ``temperature`` / ``stop`` on most
    members of the family. This helper applies the rename and the drop list
    in one place so the DefaultFramework OpenAI client and the LangChain
    adapter cannot drift on what they strip.

    The drop list is deliberately limited to the params OpenAI's API itself
    is documented to reject (verified by the live probe in
    tests/integrations/langchain/test_openai_param_filter.py); other
    sometimes-rejected params (top_p, presence_penalty, etc.) are accepted
    by some reasoning models in the family and are left to the caller.

    Returns a new dict; does not mutate the input.

    Drops: ``temperature``, ``stop``.

    Renames: ``max_tokens`` -> ``max_completion_tokens`` (only if
    ``max_tokens`` carries a non-None value and ``max_completion_tokens``
    is not already present, so an explicit ``max_completion_tokens``
    always wins and an unset ``max_tokens`` is not promoted to a null
    wire field).
    """
    out = {key: value for key, value in params.items() if key not in _REASONING_DROP_PARAMS}
    if out.get("max_tokens") is not None and "max_completion_tokens" not in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    else:
        out.pop("max_tokens", None)
    return out
