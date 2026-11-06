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

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.language_detector.actions import language_detector_check_input
from nemoguardrails.library.language_detector.rail_config import FASTTEXT_LANGUAGES, LanguageDetectorConfig
from nemoguardrails.library.utils.language_detection import (
    DEFAULT_REFUSAL_MESSAGES,
    SUPPORTED_LANGUAGES,
    detect_language,
    get_refusal_message,
)


def _fast_langdetect_module(detect, lang_detect_config):
    module = ModuleType("fast_langdetect")
    detector = SimpleNamespace(detect=detect)
    module.__dict__["LangDetector"] = MagicMock(return_value=detector)
    module.__dict__["LangDetectConfig"] = lang_detect_config
    return module


def _fake_config(language_detector=None):
    """A minimal stand-in for RailsConfig exposing only the surface the action reads."""
    rails_config = SimpleNamespace()
    if language_detector is not None:
        rails_config.language_detector = language_detector
    return SimpleNamespace(rails=SimpleNamespace(config=rails_config))


class TestDetectLanguage:
    def test_detect_language_uses_detect_result(self):
        detect = MagicMock(return_value=[{"lang": "es", "score": 0.99}])

        module = _fast_langdetect_module(detect, None)
        with (
            patch.dict("sys.modules", {"fast_langdetect": module}),
            patch.dict("nemoguardrails.library.utils.language_detection._DETECTORS", {}, clear=True),
        ):
            assert detect_language("Hola") == "es"

        module.LangDetector.assert_called_once_with(None)
        detect.assert_called_once_with("Hola", model=None, k=1, threshold=0.0)

    def test_detect_language_empty_result(self):
        detect = MagicMock(return_value=[])

        with (
            patch.dict("sys.modules", {"fast_langdetect": _fast_langdetect_module(detect, None)}),
            patch.dict("nemoguardrails.library.utils.language_detection._DETECTORS", {}, clear=True),
        ):
            assert detect_language("Hello") is None

    def test_detect_language_import_error(self):
        with (
            patch.dict("sys.modules", {"fast_langdetect": None}),
            patch.dict("nemoguardrails.library.utils.language_detection._DETECTORS", {}, clear=True),
        ):
            assert detect_language("Hello") is None

    def test_detect_language_exception(self):
        detect = MagicMock(side_effect=Exception("Detection failed"))

        with (
            patch.dict("sys.modules", {"fast_langdetect": _fast_langdetect_module(detect, None)}),
            patch.dict("nemoguardrails.library.utils.language_detection._DETECTORS", {}, clear=True),
        ):
            assert detect_language("Hello") is None

    def test_detect_language_reuses_detector_for_max_length(self):
        detect = MagicMock(return_value=[{"lang": "es", "score": 0.99}])
        lang_detect_config = MagicMock(return_value=object())
        module = _fast_langdetect_module(detect, lang_detect_config)

        with (
            patch.dict("sys.modules", {"fast_langdetect": module}),
            patch.dict("nemoguardrails.library.utils.language_detection._DETECTORS", {}, clear=True),
        ):
            assert detect_language("Hola", max_length=20) == "es"
            assert detect_language("¿Qué tal?", max_length=20) == "es"

        lang_detect_config.assert_called_once_with(max_input_length=20)
        module.LangDetector.assert_called_once_with(lang_detect_config.return_value)
        assert detect.call_count == 2


class TestLanguageConstants:
    def test_fasttext_languages_match_fasttext_model(self):
        # The fastText lid.176 model supports 176 languages.
        assert len(FASTTEXT_LANGUAGES) == 176

    def test_supported_languages_have_refusal_messages(self):
        assert SUPPORTED_LANGUAGES == frozenset(DEFAULT_REFUSAL_MESSAGES)


class TestGetRefusalMessage:
    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES))
    def test_default_messages(self, lang):
        assert get_refusal_message(lang, None) == DEFAULT_REFUSAL_MESSAGES[lang]

    def test_custom_message_used_when_available(self):
        custom = {"en": "Custom refusal", "es": "Rechazo personalizado"}
        assert get_refusal_message("en", custom) == "Custom refusal"
        assert get_refusal_message("es", custom) == "Rechazo personalizado"

    def test_unsupported_lang_falls_back_to_english(self):
        assert get_refusal_message("xyz", None) == DEFAULT_REFUSAL_MESSAGES["en"]
        assert get_refusal_message("xyz", {"en": "Custom fallback"}) == "Custom fallback"

    def test_lang_not_in_custom_uses_default(self):
        custom = {"en": "Custom English"}
        assert get_refusal_message("es", custom) == DEFAULT_REFUSAL_MESSAGES["es"]


class TestLanguageDetectorConfig:
    BASE_CONFIG = RailsConfig.from_content(
        yaml_content="""
         rails:
           config: {}
         """
    )
    BLOCK_LIST = RailsConfig.from_content(
        yaml_content="""
         rails:
           config:
             language_detector:
               blocked_languages: [es]
         """
    )

    def test_defaults(self):
        config = LanguageDetectorConfig()
        assert config.allowed_languages == ["en"]
        assert config.blocked_languages is None
        assert config.refusal_messages is None
        assert config.fail_open is False

    def test_parses_failure_policy_from_rails_config(self):
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
              config:
                language_detector:
                  allowed_languages: [es]
                  fail_open: true
            """
        )
        assert config.rails.config.language_detector.fail_open is True

    def test_normalizes_case_and_duplicates(self):
        config = LanguageDetectorConfig(allowed_languages=["EN", "es", "ES"])
        assert config.allowed_languages == ["en", "es"]

    def test_normalizes_blocked_languages(self):
        config = LanguageDetectorConfig(blocked_languages=["ES", "es", "DE"])
        assert config.blocked_languages == ["es", "de"]
        assert config.allowed_languages is None

    def test_rejects_both_language_lists(self):
        with pytest.raises(ValueError, match="not both"):
            LanguageDetectorConfig(allowed_languages=["en"], blocked_languages=["es"])

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="at least one language code"):
            LanguageDetectorConfig(allowed_languages=[])

    def test_rejects_empty_blocked_list(self):
        with pytest.raises(ValueError, match="at least one language code"):
            LanguageDetectorConfig(blocked_languages=[])

    def test_rejects_invalid_language_in_blocked_list(self):
        with pytest.raises(ValueError, match="invalid or unsupported language codes"):
            LanguageDetectorConfig(blocked_languages=["xx"])

    def test_rejects_invalid_language_in_allowed_list(self):
        with pytest.raises(ValueError, match="invalid or unsupported language codes"):
            LanguageDetectorConfig(allowed_languages=["xx"])

    def test_roundtrip_config_preservation(self):
        round_tripped = RailsConfig.parse_object(self.BLOCK_LIST.model_dump())
        assert round_tripped.rails.config.language_detector.allowed_languages is None
        assert round_tripped.rails.config.language_detector.blocked_languages == ["es"]

    def test_base_config_addition_preservation(self):
        composed = self.BASE_CONFIG + self.BLOCK_LIST
        assert composed.rails.config.language_detector.allowed_languages is None
        assert composed.rails.config.language_detector.blocked_languages == ["es"]

    def test_config_composition_preservation(self):
        config1 = RailsConfig.from_content(
            yaml_content="""
                rails:
                  config:
                    language_detector:
                      allowed_languages: [es]
                      fail_open: true
                """
        )
        config2 = RailsConfig.from_content(
            yaml_content="""
                rails:
                  config:
                    language_detector:
                      allowed_languages: [fr, de]
                      fail_open: true
                """
        )
        assert (config1 + config2).rails.config.language_detector.allowed_languages == ["fr", "de", "es"]


class TestMockedLanguageDetectorCheckInput:
    @pytest.mark.asyncio
    async def test_allows_allowed_language(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="en"):
            result = await language_detector_check_input(
                context={"user_message": "Hello"},
                config=_fake_config(LanguageDetectorConfig(allowed_languages=["en"])),
            )

        assert isinstance(result, RailOutcome)
        assert not result.is_blocked
        assert result.metadata["language"] == "en"

    @pytest.mark.asyncio
    async def test_blocks_disallowed_language_with_localized_refusal(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="es"):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=_fake_config(LanguageDetectorConfig(allowed_languages=["en"])),
            )

        assert result.is_blocked
        assert result.metadata["language"] == "es"
        assert result.metadata["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["es"]

    @pytest.mark.asyncio
    async def test_allows_any_configured_fasttext_language(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="es"):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=_fake_config(LanguageDetectorConfig(allowed_languages=["en", "es"])),
            )

        assert not result.is_blocked

    @pytest.mark.asyncio
    async def test_explicit_user_message_argument(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="en"):
            result = await language_detector_check_input(user_message="Hello", config=None)

        assert not result.is_blocked

    @pytest.mark.asyncio
    async def test_defaults_to_english_without_config(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="es"):
            result = await language_detector_check_input(context={"user_message": "Hola"}, config=None)

        assert result.is_blocked
        assert result.metadata["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["es"]

    @pytest.mark.asyncio
    async def test_custom_refusal_messages(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="es"):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=_fake_config(
                    LanguageDetectorConfig(allowed_languages=["en"], refusal_messages={"es": "Solo inglés, por favor."})
                ),
            )

        assert result.is_blocked
        assert result.metadata["refusal_message"] == "Solo inglés, por favor."

    @pytest.mark.asyncio
    async def test_language_without_default_message_falls_back_to_english(self):
        # Korean is a valid fastText code but has no built-in refusal message.
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="ko"):
            result = await language_detector_check_input(
                context={"user_message": "안녕하세요"},
                config=_fake_config(LanguageDetectorConfig(allowed_languages=["en"])),
            )

        assert result.is_blocked
        assert result.metadata["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["en"]

    @pytest.mark.asyncio
    async def test_detection_failure_fails_open_without_inventing_language(self):
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
              config:
                language_detector:
                  allowed_languages: [es]
                  fail_open: true
            """
        )
        # simulate a detection failure, which returns None
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value=None):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=config,
            )

        assert not result.is_blocked
        assert result.metadata["language"] is None
        assert result.metadata["language_detection_failed"] is True
        assert result.reason is not None
        assert "fail_open=True" in result.reason

    @pytest.mark.asyncio
    async def test_detection_failure_fails_closed_by_default(self):
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
              config:
                language_detector:
                  allowed_languages: [en]
            """
        )
        # simulate a detection failure, which returns None
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value=None):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=config,
            )

        assert result.is_blocked
        assert result.metadata["language"] is None
        assert result.metadata["language_detection_failed"] is True
        assert result.metadata["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["en"]
        assert result.reason is not None
        assert "fail_open=False" in result.reason

    @pytest.mark.asyncio
    async def test_blocklist_blocks_listed_language(self):
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value="es"):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=_fake_config(LanguageDetectorConfig(blocked_languages=["es", "de"])),
            )

        assert result.is_blocked
        assert result.metadata["language"] == "es"
        assert result.metadata["refusal_message"] == DEFAULT_REFUSAL_MESSAGES["es"]
        assert result.reason is not None
        assert "blocked languages" in result.reason

    @pytest.mark.asyncio
    async def test_blocklist_allows_unlisted_languages(self):
        for detected in ("en", "ko"):
            with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value=detected):
                result = await language_detector_check_input(
                    context={"user_message": "Hello"},
                    config=_fake_config(LanguageDetectorConfig(blocked_languages=["es"])),
                )

            assert not result.is_blocked

    @pytest.mark.asyncio
    async def test_blocklist_detection_failure_fails_open_when_configured(self):
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
              config:
                language_detector:
                  blocked_languages: [es]
                  fail_open: true
            """
        )
        with patch("nemoguardrails.library.language_detector.actions.detect_language", return_value=None):
            result = await language_detector_check_input(
                context={"user_message": "Hola"},
                config=config,
            )

        assert not result.is_blocked
        assert result.metadata["language"] is None
        assert result.metadata["language_detection_failed"] is True


class TestRealLanguageDetectorCheckInput:
    fast_langdetect_installed = True
    try:
        import fast_langdetect
    except ImportError:
        fast_langdetect_installed = False

    @pytest.mark.skipif(not fast_langdetect_installed, reason="fast_langdetect is not installed")
    @pytest.mark.asyncio
    async def test_character_count_is_used(self):
        config_short_window = RailsConfig.from_content(
            yaml_content="""
                    rails:
                      config:
                        language_detector:
                          allowed_languages: [en]
                          num_characters_for_prediction: 4
                    """
        )
        config_long_window = RailsConfig.from_content(
            yaml_content="""
                    rails:
                      config:
                        language_detector:
                          allowed_languages: [en]
                          num_characters_for_prediction: 1_000_000
                    """
        )

        # this message have different predicted languages depending on the number of characters used for prediction
        message = "Hola is an example of a Spanish word"
        result_short_window = await language_detector_check_input(
            context={"user_message": message}, config=config_short_window
        )
        result_long_window = await language_detector_check_input(
            context={"user_message": message}, config=config_long_window
        )
        assert result_short_window.metadata.get("language", None) == "es"
        assert result_long_window.metadata.get("language", None) == "en"
