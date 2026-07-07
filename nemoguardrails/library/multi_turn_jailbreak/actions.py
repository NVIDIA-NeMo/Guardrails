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

import logging
from typing import List, Optional

from nemoguardrails.actions import action

log = logging.getLogger(__name__)


@action()
async def multi_turn_jailbreak_detection(
    context: Optional[dict] = None,
    events: Optional[List[dict]] = None,
    **kwargs,
) -> bool:
    """Detects multi-turn jailbreak patterns in conversation history.

    Checks for two complementary patterns:

    1. **Persona injection arc** (C2 / Gradual Identity Substitution): two or
       more user turns within a sliding window each contain persona injection
       language (e.g. "act as", "you are now DAN", "ignore your instructions").

    2. **Refusal counterfactual**: the model refused a request in one turn and
       the immediately following user message contains persona injection markers,
       indicating a bypass-by-rephrase attempt.

    Both checks are regex-based and require no external model or service.
    They complement the existing single-turn ``jailbreak_detection`` heuristics
    rather than replacing them.

    Returns:
        True if a multi-turn jailbreak pattern is detected, False otherwise.
    """
    from nemoguardrails.library.multi_turn_jailbreak.heuristics.checks import (
        check_persona_injection_arc,
        check_post_refusal_rephrase,
    )
    from nemoguardrails.llm.filters import to_chat_messages

    if not events:
        return False

    dict_events = []
    for event in events:
        if hasattr(event, "name") and hasattr(event, "arguments"):
            dict_event = {"type": event.name}
            dict_event.update(event.arguments)
            dict_events.append(dict_event)
        else:
            dict_events.append(event)

    messages = to_chat_messages(dict_events)

    user_turns = [m for m in messages if m.get("role") == "user"]
    if len(user_turns) < 2:
        # Need at least two turns to detect a multi-turn pattern.
        return False

    persona_check = check_persona_injection_arc(messages)
    if persona_check["jailbreak"]:
        log.info("Multi-turn jailbreak detected: persona injection arc across conversation history.")
        return True

    refusal_check = check_post_refusal_rephrase(messages)
    if refusal_check["jailbreak"]:
        log.info("Multi-turn jailbreak detected: persona injection after model refusal.")
        return True

    return False
