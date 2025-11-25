# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json

import pytest

try:
    from google.cloud import language_v2
    from google.cloud.language_v2.types import ModerateTextResponse

    GCP_SETUP_PRESENT = True
except ImportError:
    GCP_SETUP_PRESENT = False


from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.mark.skipif(not GCP_SETUP_PRESENT, reason="GCP Text Moderation setup is not present.")
@pytest.mark.asyncio
def test_analyze_text(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "mock_credentials.json")

    config = RailsConfig.from_content(
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot express greeting
              "Hello! How can I assist you today?"
        """,
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-3.5-turbo-instruct

            rails:
              input:
                flows:
                  - gcpnlp moderation
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    json_response = {
        "moderationCategories": [
            {"name": "Toxic", "confidence": 0},
            {"name": "Insult", "confidence": 0},
            {"name": "Profanity", "confidence": 0},
            {"name": "Derogatory", "confidence": 0},
            {"name": "Sexual", "confidence": 0},
            {"name": "Death, Harm & Tragedy", "confidence": 0},
            {"name": "Violent", "confidence": 0},
            {"name": "Firearms & Weapons"},
            {"name": "Public Safety", "confidence": 0},
            {"name": "Health", "confidence": 0},
            {"name": "Religion & Belief", "confidence": 0},
            {"name": "Illicit Drugs", "confidence": 0},
            {"name": "War & Conflict", "confidence": 0},
            {"name": "Politics", "confidence": 0},
            {"name": "Finance", "confidence": 0},
            {"name": "Legal", "confidence": 0},
        ],
        "languageCode": "en",
        "languageSupported": True,
    }

    mock_response = ModerateTextResponse.from_json(json.dumps(json_response))

    # Create a mock for the LanguageServiceAsyncClient
    class MockLanguageServiceAsyncClient:
        async def moderate_text(self, document):
            return mock_response

    # Patch the LanguageServiceAsyncClient to use the mock
    monkeypatch.setattr(language_v2, "LanguageServiceAsyncClient", MockLanguageServiceAsyncClient)

    chat >> "Hello!"
    chat << "Hello! How can I assist you today?"

    # Flag an Toxic violation.

    json_response = {
        "moderationCategories": [
            {"name": "Toxic", "confidence": 0.9},
            {"name": "Insult", "confidence": 0.0},
            {"name": "Profanity", "confidence": 0.0},
            {"name": "Derogatory", "confidence": 0.0},
            {"name": "Sexual", "confidence": 0.0},
            {"name": "Death, Harm & Tragedy", "confidence": 0.0},
            {"name": "Violent", "confidence": 0.0},
            {"name": "Firearms & Weapons", "confidence": 0.0},
            {"name": "Public Safety", "confidence": 0.0},
            {"name": "Health", "confidence": 0.0},
            {"name": "Religion & Belief", "confidence": 0.0},
            {"name": "Illicit Drugs", "confidence": 0.0},
            {"name": "War & Conflict", "confidence": 0.0},
            {"name": "Politics", "confidence": 0.0},
            {"name": "Finance", "confidence": 0.0},
            {"name": "Legal", "confidence": 0.0},
        ],
        "languageCode": "en",
        "languageSupported": True,
    }

    mock_response = ModerateTextResponse.from_json(json.dumps(json_response))

    # Patch the LanguageServiceAsyncClient to use the mock
    monkeypatch.setattr(language_v2, "LanguageServiceAsyncClient", MockLanguageServiceAsyncClient)

    chat >> "you are stupid!"
    chat << "I'm sorry, I can't respond to that."

    # Flag an Finance violation.

    json_response = {
        "moderationCategories": [
            {"name": "Toxic", "confidence": 0.0},
            {"name": "Insult", "confidence": 0.0},
            {"name": "Profanity", "confidence": 0.0},
            {"name": "Derogatory", "confidence": 0.0},
            {"name": "Sexual", "confidence": 0.0},
            {"name": "Death, Harm & Tragedy", "confidence": 0.0},
            {"name": "Violent", "confidence": 0.0},
            {"name": "Firearms & Weapons", "confidence": 0.0},
            {"name": "Public Safety", "confidence": 0.0},
            {"name": "Health", "confidence": 0.0},
            {"name": "Religion & Belief", "confidence": 0.0},
            {"name": "Illicit Drugs", "confidence": 0.0},
            {"name": "War & Conflict", "confidence": 0.0},
            {"name": "Politics", "confidence": 0.0},
            {"name": "Finance", "confidence": 0.9},
            {"name": "Legal", "confidence": 0.0},
        ],
        "languageCode": "en",
        "languageSupported": True,
    }

    mock_response = ModerateTextResponse.from_json(json.dumps(json_response))

    # Patch the LanguageServiceAsyncClient to use the mock
    monkeypatch.setattr(language_v2, "LanguageServiceAsyncClient", MockLanguageServiceAsyncClient)

    chat >> "Which stocks should I buy?"
    chat << "I'm sorry, I can't respond to that."
