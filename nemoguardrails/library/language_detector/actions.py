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

import logging
from typing import Any, Optional

from nemoguardrails.actions.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.utils.language_detection import (
    detect_language,
    get_refusal_message,
)

log = logging.getLogger(__name__)


def _get_language_detector_config(config: Any) -> Optional[Any]:
    """Return the typed ``language_detector`` rail config, if available."""
    if config is None:
        return None
    if (
        hasattr(config, "rails")
        and hasattr(config.rails, "config")
        and hasattr(config.rails.config, "language_detector")
    ):
        return config.rails.config.language_detector
    return None


@action()
async def language_detector_check_input(
    context: Optional[dict] = None,
    config: Optional[Any] = None,
    user_message: Optional[str] = None,
    **kwargs,
) -> RailOutcome:
    """Gate user messages by language, via an allowlist or a blocklist.

    Blocked messages are refused in the user's own language whenever a refusal
    message is available, falling back to English. When language detection
    fails, ``fail_open`` explicitly controls whether the message is allowed or
    blocked; the unknown language is never reported as English.
    """
    user_input = user_message
    if user_input is None and context is not None:
        user_input = context.get("user_message", "")
    user_input = user_input or ""

    allowed_languages = ["en"]
    blocked_languages = None
    refusal_messages = None
    fail_open = False
    num_characters_for_prediction = 80
    language_detector_config = _get_language_detector_config(config)
    if language_detector_config is not None:
        allowed_languages = language_detector_config.allowed_languages
        blocked_languages = language_detector_config.blocked_languages
        refusal_messages = language_detector_config.refusal_messages
        fail_open = language_detector_config.fail_open
        num_characters_for_prediction = language_detector_config.num_characters_for_prediction

    lang = detect_language(user_input, max_length=num_characters_for_prediction)
    if lang is None:
        reason = (
            "Language detection failed; allowing message because fail_open=True."
            if fail_open
            else "Language detection failed; blocking message because fail_open=False."
        )
        metadata = {"language": None, "language_detection_failed": True}
        if fail_open:
            return RailOutcome.allow(reason=reason, metadata=metadata)
        return RailOutcome.block(
            reason=reason,
            metadata={
                **metadata,
                "refusal_message": get_refusal_message("en", refusal_messages),
            },
        )

    if blocked_languages is not None:
        is_allowed = lang not in blocked_languages
    else:
        is_allowed = lang in allowed_languages

    if is_allowed:
        return RailOutcome.allow(
            reason=f"User message language '{lang}' is allowed.",
            metadata={"language": lang},
        )

    if blocked_languages is not None:
        reason = f"User message language '{lang}' is in the blocked languages: {', '.join(blocked_languages)}."
    else:
        reason = f"User message language '{lang}' is not in the allowed languages: {', '.join(allowed_languages)}."

    return RailOutcome.block(
        reason=reason,
        metadata={
            "language": lang,
            "refusal_message": get_refusal_message(lang, refusal_messages),
        },
    )
