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

from nemoguardrails.actions.llm.utils import (
    get_colang_history,
    get_last_user_utterance,
)

FAKE_BASE64 = "iVBORw0KGgoAAAANSUhEUg" * 5000


def _multimodal_content(text=None, image_b64=None):
    parts = []
    if text is not None:
        parts.append({"type": "text", "text": text})
    if image_b64 is not None:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return parts


class TestGetColangHistoryMultimodal:
    def test_text_only_message_unchanged(self):
        events = [{"type": "UserMessage", "text": "Hello there"}]
        result = get_colang_history(events)
        assert 'user "Hello there"' in result

    def test_multimodal_text_and_image(self):
        events = [{"type": "UserMessage", "text": _multimodal_content("Describe this", FAKE_BASE64)}]
        result = get_colang_history(events)
        assert FAKE_BASE64 not in result
        assert "Describe this [+ image]" in result

    def test_multimodal_image_only(self):
        events = [{"type": "UserMessage", "text": _multimodal_content(image_b64=FAKE_BASE64)}]
        result = get_colang_history(events)
        assert FAKE_BASE64 not in result
        assert 'user "[+ image]"' in result

    def test_multimodal_multiple_text_parts(self):
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{FAKE_BASE64}"}},
        ]
        events = [{"type": "UserMessage", "text": content}]
        result = get_colang_history(events)
        assert FAKE_BASE64 not in result
        assert "First part Second part [+ image]" in result

    def test_multimodal_does_not_bloat_history(self):
        events = [{"type": "UserMessage", "text": _multimodal_content("Describe this", FAKE_BASE64)}]
        result = get_colang_history(events)
        assert FAKE_BASE64 not in result
        assert len(result) < 1000

    def test_mixed_text_and_multimodal_conversation(self):
        events = [
            {"type": "UserMessage", "text": "Hi"},
            {"type": "UserIntent", "intent": "express greeting"},
            {"type": "BotIntent", "intent": "express greeting"},
            {"type": "StartUtteranceBotAction", "script": "Hello!"},
            {"type": "UserMessage", "text": _multimodal_content("What is this?", FAKE_BASE64)},
        ]
        result = get_colang_history(events)
        assert 'user "Hi"' in result
        assert FAKE_BASE64 not in result
        assert 'user "What is this? [+ image]"' in result


class TestGetLastUserUtteranceMultimodal:
    def test_text_returns_string(self):
        events = [{"type": "UserMessage", "text": "Plain text"}]
        result = get_last_user_utterance(events)
        assert result == "Plain text"
        assert isinstance(result, str)

    def test_multimodal_returns_string(self):
        events = [{"type": "UserMessage", "text": _multimodal_content("Describe this", FAKE_BASE64)}]
        result = get_last_user_utterance(events)
        assert isinstance(result, str)
        assert FAKE_BASE64 not in result
        assert "[+ image]" in result

    def test_multimodal_image_only(self):
        events = [{"type": "UserMessage", "text": _multimodal_content(image_b64=FAKE_BASE64)}]
        result = get_last_user_utterance(events)
        assert isinstance(result, str)
        assert FAKE_BASE64 not in result
        assert result == "[+ image]"

    def test_multimodal_none_text_part_does_not_crash(self):
        events = [
            {
                "type": "UserMessage",
                "text": [
                    {"type": "text", "text": None},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{FAKE_BASE64}"}},
                ],
            }
        ]
        result = get_last_user_utterance(events)
        assert isinstance(result, str)
        assert FAKE_BASE64 not in result
        assert result == "[+ image]"

    def test_empty_list_returns_empty_string(self):
        events = [{"type": "UserMessage", "text": []}]
        assert get_last_user_utterance(events) == ""

    def test_multiple_images_single_placeholder(self):
        events = [
            {
                "type": "UserMessage",
                "text": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
                ],
            }
        ]
        assert get_last_user_utterance(events) == "[+ image]"
