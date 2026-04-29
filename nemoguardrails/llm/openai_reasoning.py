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
