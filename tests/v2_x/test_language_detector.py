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

from unittest.mock import patch

from nemoguardrails import RailsConfig
from tests.utils import TestChat

yaml_content = """
colang_version: "2.x"
models:
  - type: main
    engine: openai
    model: gpt-4-turbo
rails:
  config:
    language_detector:
      allowed_languages:
        - en
"""

colang_content = """
import core
import guardrails
import nemoguardrails.library.language_detector

flow main
  await user said something
  bot say "OK"

flow input rails $input_text
  language detector check input
"""


def test_language_detector_v2_allows_english():
    """The Colang 2.x flow lets through messages in the allowed language."""
    with patch(
        "nemoguardrails.library.utils.language_detection.detect_language",
        return_value="en",
    ):
        chat = TestChat(
            RailsConfig.from_content(colang_content, yaml_content),
            llm_completions=[],
        )
        chat >> "Can you explain your return policy?"
        chat << "OK"


def test_language_detector_v2_blocks_spanish():
    """The Colang 2.x flow blocks other languages and refuses in the user's language."""
    with patch(
        "nemoguardrails.library.utils.language_detection.detect_language",
        return_value="es",
    ):
        chat = TestChat(
            RailsConfig.from_content(colang_content, yaml_content),
            llm_completions=[],
        )
        chat >> "¿Puedes explicar tu política de devoluciones?"
        chat << "Lo siento, no puedo responder a eso."


def test_language_detector_v2_emits_exception_for_blocked_spanish():
    """The Colang 2.x flow emits its configured rail exception for blocked input."""
    with patch(
        "nemoguardrails.library.utils.language_detection.detect_language",
        return_value="es",
    ):
        exception_yaml_content = yaml_content.replace(
            'colang_version: "2.x"',
            'colang_version: "2.x"\nenable_rails_exceptions: true',
        )
        chat = TestChat(
            RailsConfig.from_content(colang_content, exception_yaml_content),
            llm_completions=[],
        )
        chat >> "¿Puedes explicar tu política de devoluciones?"
        chat.bot({"type": "LanguageDetectorRailException"})
