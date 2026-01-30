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

# Pydantic models enforcing type-correctness in public methods

from pydantic import BaseModel, Field

from nemoguardrails.guardrails.guardrails_types import MessageRole


class GuardrailsRequest(BaseModel):
    """Request model for Guardrails"""

    prompt: str | list[dict[MessageRole, str]] = Field(
        description="LLM request. Can be a string, a list of messages, or GuardrailsRequest."
    )

    llm_params: dict | None = Field(
        default=None,
        description="Additional model-specific parameters that should be used for the LLM call",
    )
