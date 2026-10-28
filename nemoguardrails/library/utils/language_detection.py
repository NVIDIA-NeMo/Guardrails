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

"""Shared language-detection utilities for rails.

Detection is powered by fast-langdetect (the fastText ``lid.176`` model),
which supports the 176 language codes listed in ``FASTTEXT_LANGUAGES``:
https://fasttext.cc/docs/en/language-identification.html
"""

import logging
from typing import Any, Dict, FrozenSet, Optional, cast

log = logging.getLogger(__name__)

# The language codes supported by the fastText lid.176 model used by
# fast-langdetect. Source: https://fasttext.cc/docs/en/language-identification.html


DEFAULT_REFUSAL_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry, I can't respond to that.",
    "es": "Lo siento, no puedo responder a eso.",
    "zh": "抱歉，我无法回应。",
    "de": "Es tut mir leid, darauf kann ich nicht antworten.",
    "fr": "Je suis désolé, je ne peux pas répondre à cela.",
    "hi": "मुझे खेद है, मैं इसका जवाब नहीं दे सकता।",
    "ja": "申し訳ありませんが、それには回答できません。",
    "ar": "عذراً، لا أستطيع الرد على ذلك.",
    "th": "ขออภัย ฉันไม่สามารถตอบได้",
}

# The languages with a built-in localized refusal message.
SUPPORTED_LANGUAGES: FrozenSet[str] = frozenset(DEFAULT_REFUSAL_MESSAGES)
_DETECTORS: Dict[Optional[int], Any] = {}


def detect_language(text: str, max_length: Optional[int] = None) -> Optional[str]:
    """Detect the language of ``text`` using fast-langdetect.

    Returns the ISO code of the most likely language (one of
    ``FASTTEXT_LANGUAGES``), or ``None`` when detection is unavailable
    (fast-langdetect not installed) or fails.
    """

    if not text or not text.strip():
        log.warning("detect_language received an empty text argument, skipping")
        return None

    try:
        from fast_langdetect import LangDetectConfig, LangDetector

        if max_length not in _DETECTORS:
            config = LangDetectConfig(max_input_length=max_length) if max_length is not None else None
            _DETECTORS[max_length] = LangDetector(config)

        result = _DETECTORS[max_length].detect(text, model=None, k=1, threshold=0.0)
        if result and len(result) > 0:
            return cast(Optional[str], result[0].get("lang"))
        return None
    except ImportError:
        log.warning("fast-langdetect not installed, skipping")
        return None
    except Exception as e:
        log.warning(f"fast-langdetect detection failed: {e}")
        return None


def get_refusal_message(lang: str, custom_messages: Optional[Dict[str, str]] = None) -> str:
    """Return the refusal message for ``lang``.

    Preference order: a custom message for the language, the built-in default
    for the language, a custom English message, then the built-in English one.
    """
    if custom_messages and lang in custom_messages:
        return custom_messages[lang]
    if lang in DEFAULT_REFUSAL_MESSAGES:
        return DEFAULT_REFUSAL_MESSAGES[lang]
    if custom_messages and "en" in custom_messages:
        return custom_messages["en"]
    return DEFAULT_REFUSAL_MESSAGES["en"]
