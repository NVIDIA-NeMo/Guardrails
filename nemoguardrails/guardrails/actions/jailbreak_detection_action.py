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

"""Jailbreak detection rail action for IORails."""

from typing import Any, Optional

from nemoguardrails.guardrails.guardrails_types import LLMMessages, RailResult
from nemoguardrails.guardrails.rail_action import RailAction


class JailbreakDetectionAction(RailAction):
    """Detect jailbreak attempts via the NIM jailbreak detection API."""

    action_name = "jailbreak detection model"
    requires_model = False

    def _extract_messages(self, messages: LLMMessages, bot_response: Optional[str]) -> dict[str, Any]:
        return {"user_input": self._last_user_content(messages)}

    def _create_prompt(self, model_type: Optional[str], extracted: dict[str, Any]) -> dict[str, str]:
        return {"input": extracted["user_input"]}

    async def _get_response(self, model_type: Optional[str], prompt: Any) -> dict:
        return await self._get_api_response("jailbreak_detection", prompt)

    def _parse_response(self, response: Any) -> RailResult:
        if "jailbreak" not in response:
            raise RuntimeError(f"Jailbreak response missing 'jailbreak' field: {response}")

        score = response.get("score", "unknown")
        if response["jailbreak"]:
            return RailResult(is_safe=False, reason=f"Score: {score}")
        return RailResult(is_safe=True, reason=f"Score: {score}")
