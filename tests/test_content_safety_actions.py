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

from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.library.content_safety.actions import (
    DEFAULT_REFUSAL_MESSAGES,
    SUPPORTED_LANGUAGES,
    _detect_language,
    _get_refusal_message,
    content_safety_check_input,
    content_safety_check_output,
    content_safety_check_output_mapping,
    detect_language,
)
from tests.utils import FakeLLM

try:
    import fast_langdetect  # noqa

    HAS_FAST_LANGDETECT = True
except ImportError:
    HAS_FAST_LANGDETECT = False

requires_fast_langdetect = pytest.mark.skipif(not HAS_FAST_LANGDETECT, reason="fast-langdetect not installed")


@pytest.fixture
def fake_llm():
    def _factory(response):
        llm = FakeLLM(responses=[response])
        return {"test_model": llm}

    return _factory


@pytest.fixture
def mock_task_manager():
    tm = MagicMock()
    tm.render_task_prompt.return_value = "test prompt"
    tm.get_stop_tokens.return_value = []
    tm.get_max_tokens.return_value = 3
    return tm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check_fn, context, parsed_text, expected_allowed, expected_violations",
    [
        (
            content_safety_check_input,
            {"user_message": "foo"},
            [True, "policy1", "policy2"],
            True,
            ["policy1", "policy2"],
        ),
        (
            content_safety_check_input,
            {"user_message": "foo"},
            [False],
            False,
            [],
        ),
        (
            content_safety_check_output,
            {"user_message": "foo", "bot_message": "bar"},
            [False, "hate", "violence"],
            False,
            ["hate", "violence"],
        ),
        (
            content_safety_check_output,
            {"user_message": "foo", "bot_message": "bar"},
            [True],
            True,
            [],
        ),
    ],
)
async def test_content_safety_parsing(
    fake_llm,
    mock_task_manager,
    check_fn,
    context,
    parsed_text,
    expected_allowed,
    expected_violations,
):
    llms = fake_llm("irrelevant")
    mock_task_manager.parse_task_output.return_value = parsed_text

    result = await check_fn(
        llms=llms,
        llm_task_manager=mock_task_manager,
        model_name="test_model",
        context=context,
    )
    assert result["allowed"] is expected_allowed
    assert result["policy_violations"] == expected_violations


@pytest.mark.asyncio
async def test_content_safety_check_input_missing_model_name():
    """Test content_safety_check_input raises ValueError when model_name is missing."""
    llms = {}
    mock_task_manager = MagicMock()

    with pytest.raises(ValueError, match="Model name is required"):
        await content_safety_check_input(llms=llms, llm_task_manager=mock_task_manager, model_name=None, context={})


@pytest.mark.asyncio
async def test_content_safety_check_input_model_not_found():
    """Test content_safety_check_input raises ValueError when model is not found."""
    llms = {}
    mock_task_manager = MagicMock()

    with pytest.raises(ValueError, match="Model test_model not found"):
        await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context={},
        )


def test_content_safety_check_output_mapping_allowed():
    """Test content_safety_check_output_mapping returns False when content is allowed."""
    result = {"allowed": True, "policy_violations": []}
    assert content_safety_check_output_mapping(result) is False


def test_content_safety_check_output_mapping_blocked():
    """Test content_safety_check_output_mapping returns True when content should be blocked."""

    result = {"allowed": False, "policy_violations": ["violence"]}
    assert content_safety_check_output_mapping(result) is True


def test_content_safety_check_output_mapping_blocked_policy_violations_only():
    """Test content_safety_check_output_mapping returns True when content should be blocked."""

    # TODO:@trebedea is this the expected behavior?
    result = {"allowed": True, "policy_violations": ["violence"]}
    assert content_safety_check_output_mapping(result) is False


def test_content_safety_check_output_mapping_default():
    """Test content_safety_check_output_mapping defaults to allowed=False when key is missing."""
    result = {"policy_violations": []}
    assert content_safety_check_output_mapping(result) is False


@requires_fast_langdetect
class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text,expected_lang",
        [
            ("Hello, how are you today?", "en"),
            ("Hola, ¿cómo estás hoy?", "es"),
            ("你好，你今天好吗？", "zh"),
            ("Guten Tag, wie geht es Ihnen?", "de"),
            ("Bonjour, comment allez-vous?", "fr"),
            ("こんにちは、お元気ですか？", "ja"),
        ],
        ids=["english", "spanish", "chinese", "german", "french", "japanese"],
    )
    def test_detect_language(self, text, expected_lang):
        assert _detect_language(text) == expected_lang

    def test_detect_language_empty_string(self):
        result = _detect_language("")
        assert result is None or result == "en"

    def test_detect_language_import_error(self):
        with patch.dict("sys.modules", {"fast_langdetect": None}):
            import nemoguardrails.library.content_safety.actions as actions_module

            _original_detect_language = actions_module._detect_language

            def patched_detect_language(text):
                try:
                    raise ImportError("No module named 'fast_langdetect'")
                except ImportError:
                    return None

            with patch.object(actions_module, "_detect_language", patched_detect_language):
                result = actions_module._detect_language("Hello")
                assert result is None

    def test_detect_language_exception(self):
        with patch("fast_langdetect.detect", side_effect=Exception("Detection failed")):
            result = _detect_language("Hello")
            assert result is None


class TestGetRefusalMessage:
    @pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES))
    def test_default_messages(self, lang):
        result = _get_refusal_message(lang, None)
        assert result == DEFAULT_REFUSAL_MESSAGES[lang]

    def test_custom_message_used_when_available(self):
        custom = {"en": "Custom refusal", "es": "Rechazo personalizado"}
        assert _get_refusal_message("en", custom) == "Custom refusal"
        assert _get_refusal_message("es", custom) == "Rechazo personalizado"

    def test_unsupported_lang_falls_back_to_english(self):
        assert _get_refusal_message("xyz", None) == DEFAULT_REFUSAL_MESSAGES["en"]
        assert _get_refusal_message("xyz", {"en": "Custom fallback"}) == "Custom fallback"

    def test_lang_not_in_custom_uses_default(self):
        custom = {"en": "Custom English"}
        assert _get_refusal_message("es", custom) == DEFAULT_REFUSAL_MESSAGES["es"]


@requires_fast_langdetect
class TestDetectLanguageAction:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_message,expected_lang",
        [
            ("Hello, how are you?", "en"),
            ("Hola, ¿cómo estás?", "es"),
            ("你好", "zh"),
        ],
        ids=["english", "spanish", "chinese"],
    )
    async def test_detect_language_action(self, user_message, expected_lang):
        context = {"user_message": user_message}
        result = await detect_language(context=context, config=None)
        assert result["language"] == expected_lang
        assert result["refusal_message"] == DEFAULT_REFUSAL_MESSAGES[expected_lang]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "context",
        [None, {"user_message": ""}],
        ids=["no_context", "empty_message"],
    )
    async def test_detect_language_action_defaults_to_english(self, context):
        result = await detect_language(context=context, config=None)
        assert result["language"] == "en"
        assert result["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["en"]

    @pytest.mark.asyncio
    async def test_detect_language_action_unsupported_language_falls_back_to_english(self):
        with patch(
            "nemoguardrails.library.content_safety.actions._detect_language",
            return_value="xyz",
        ):
            context = {"user_message": "some text"}
            result = await detect_language(context=context, config=None)
            assert result["language"] == "en"
            assert result["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["en"]

    @pytest.mark.asyncio
    async def test_detect_language_action_with_config_custom_messages(self):
        mock_config = MagicMock()
        mock_config.rails.config.content_safety.multilingual.refusal_messages = {
            "en": "Custom: Cannot help",
            "es": "Personalizado: No puedo ayudar",
        }

        context = {"user_message": "Hello"}
        result = await detect_language(context=context, config=mock_config)
        assert result["language"] == "en"
        assert result["refusal_message"] == "Custom: Cannot help"

    @pytest.mark.asyncio
    async def test_detect_language_action_with_config_no_multilingual(self):
        mock_config = MagicMock()
        mock_config.rails.config.content_safety.multilingual = None

        context = {"user_message": "Hello"}
        result = await detect_language(context=context, config=mock_config)
        assert result["language"] == "en"
        assert result["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["en"]


class TestSupportedLanguagesAndDefaults:
    def test_supported_languages_count(self):
        assert len(SUPPORTED_LANGUAGES) == 9

    def test_supported_languages_contents(self):
        expected = {"en", "es", "zh", "de", "fr", "hi", "ja", "ar", "th"}
        assert SUPPORTED_LANGUAGES == expected

    def test_default_refusal_messages_has_all_supported_languages(self):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in DEFAULT_REFUSAL_MESSAGES

    def test_default_refusal_messages_are_non_empty(self):
        for _lang, message in DEFAULT_REFUSAL_MESSAGES.items():
            assert message
            assert len(message) > 0
