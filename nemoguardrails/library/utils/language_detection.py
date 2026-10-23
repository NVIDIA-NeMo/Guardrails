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
from typing import Dict, FrozenSet, Optional, cast

log = logging.getLogger(__name__)

# The language codes supported by the fastText lid.176 model used by
# fast-langdetect. Source: https://fasttext.cc/docs/en/language-identification.html
FASTTEXT_LANGUAGES: FrozenSet[str] = frozenset(
    {
        "af",
        "als",
        "am",
        "an",
        "ar",
        "arz",
        "as",
        "ast",
        "av",
        "az",
        "azb",
        "ba",
        "bar",
        "bcl",
        "be",
        "bg",
        "bh",
        "bn",
        "bo",
        "bpy",
        "br",
        "bs",
        "bxr",
        "ca",
        "cbk",
        "ce",
        "ceb",
        "ckb",
        "co",
        "cs",
        "cv",
        "cy",
        "da",
        "de",
        "diq",
        "dsb",
        "dty",
        "dv",
        "el",
        "eml",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "frr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gn",
        "gom",
        "gu",
        "gv",
        "he",
        "hi",
        "hif",
        "hr",
        "hsb",
        "ht",
        "hu",
        "hy",
        "ia",
        "id",
        "ie",
        "ilo",
        "io",
        "is",
        "it",
        "ja",
        "jbo",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "krc",
        "ku",
        "kv",
        "kw",
        "ky",
        "la",
        "lb",
        "lez",
        "li",
        "lmo",
        "lo",
        "lrc",
        "lt",
        "lv",
        "mai",
        "mg",
        "mhr",
        "min",
        "mk",
        "ml",
        "mn",
        "mr",
        "mrj",
        "ms",
        "mt",
        "mwl",
        "my",
        "myv",
        "mzn",
        "nah",
        "nap",
        "nds",
        "ne",
        "new",
        "nl",
        "nn",
        "no",
        "oc",
        "or",
        "os",
        "pa",
        "pam",
        "pfl",
        "pl",
        "pms",
        "pnb",
        "ps",
        "pt",
        "qu",
        "rm",
        "ro",
        "ru",
        "rue",
        "sa",
        "sah",
        "sc",
        "scn",
        "sco",
        "sd",
        "sh",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "tyv",
        "ug",
        "uk",
        "ur",
        "uz",
        "vec",
        "vep",
        "vi",
        "vls",
        "vo",
        "wa",
        "war",
        "wuu",
        "xal",
        "xmf",
        "yi",
        "yo",
        "yue",
        "zh",
    }
)

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


def detect_language(text: str) -> Optional[str]:
    """Detect the language of ``text`` using fast-langdetect.

    Returns the ISO code of the most likely language (one of
    ``FASTTEXT_LANGUAGES``), or ``None`` when detection is unavailable
    (fast-langdetect not installed) or fails.
    """
    try:
        from fast_langdetect import detect

        result = detect(text, k=1)
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
