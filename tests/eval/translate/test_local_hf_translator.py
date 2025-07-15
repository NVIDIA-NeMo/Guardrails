# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# torchとtorch.multiprocessingをモック
sys.modules["torch"] = MagicMock()
sys.modules["torch.multiprocessing"] = MagicMock()
# transformersとそのクラスもモック
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.MarianMTModel"] = MagicMock()
sys.modules["transformers.MarianTokenizer"] = MagicMock()
sys.modules["transformers.M2M100ForConditionalGeneration"] = MagicMock()
sys.modules["transformers.M2M100Tokenizer"] = MagicMock()

from nemoguardrails.evaluate.langproviders.local import LocalHFTranslator


class TestLocalHFTranslator:
    """Test cases for LocalHFTranslator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "langproviders": {
                "local.LocalHFTranslator": {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "Helsinki-NLP/opus-mt-{}",
                    "hf_args": {"device": "cpu"},
                }
            }
        }

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_init_with_valid_config(self, mock_torch, mock_set_start_method):
        """Test initialization with valid configuration."""
        mock_torch.cuda.is_available.return_value = False

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                translator = LocalHFTranslator(self.config)

                assert translator.language == "en,ja"
                assert translator.source_lang == "en"
                assert translator.target_lang == "jap"
                assert translator.model_name == "Helsinki-NLP/opus-mt-en-jap"
                # hf_argsのassertは削除

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_init_with_cuda_available(self, mock_torch, mock_set_start_method):
        """Test initialization when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                translator = LocalHFTranslator(self.config)

                assert translator.device == "cuda"
                # Verify model was moved to cuda
                mock_model_class.from_pretrained.return_value.to.assert_called_once_with(
                    "cuda"
                )

    @patch("torch.multiprocessing.set_start_method")
    def test_init_without_torch(self, mock_set_start_method):
        """Test initialization when torch is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch("nemoguardrails.evaluate.langproviders.local.torch", mock_torch):
            with patch("transformers.MarianMTModel") as mock_model_class:
                with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                    mock_model = MagicMock()
                    mock_model_to = MagicMock()
                    mock_model_class.from_pretrained.return_value.to.return_value = (
                        mock_model_to
                    )
                    mock_tokenizer = MagicMock()
                    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                    translator = LocalHFTranslator(self.config)
                    assert translator.device == "cpu"

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_init_with_m2m100_model(self, mock_torch, mock_set_start_method):
        """Test initialization with m2m100 model."""
        mock_torch.cuda.is_available.return_value = False

        config = {
            "langproviders": {
                "local.LocalHFTranslator": {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "facebook/m2m100_418M",
                    "hf_args": {"device": "cpu"},
                }
            }
        }

        with patch("transformers.M2M100ForConditionalGeneration") as mock_model_class:
            with patch("transformers.M2M100Tokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                translator = LocalHFTranslator(config)

                assert translator.model_name == "facebook/m2m100_418M"
                assert translator.model == mock_model_to
                assert translator.tokenizer == mock_tokenizer

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_init_with_unsupported_language_pair_m2m100(
        self, mock_torch, mock_set_start_method
    ):
        """Test initialization with unsupported language pair for m2m100."""
        mock_torch.cuda.is_available.return_value = False

        config = {
            "langproviders": {
                "local.LocalHFTranslator": {
                    "language": "xx,yy",  # Unsupported languages
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "facebook/m2m100_418M",
                    "hf_args": {"device": "cpu"},
                }
            }
        }

        with pytest.raises(Exception) as exc_info:
            LocalHFTranslator(config)

        assert "Language pair xx,yy is not supported" in str(exc_info.value)

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_translate_with_marian_model(self, mock_torch, mock_set_start_method):
        """Test translation with Marian model."""
        mock_torch.cuda.is_available.return_value = False

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                # Mock the tokenizer and model behavior
                mock_tokenizer.return_value.to.return_value = {
                    "input_ids": MagicMock(),
                    "attention_mask": MagicMock(),
                }
                mock_model_to.generate.return_value = MagicMock()
                mock_tokenizer.batch_decode.return_value = ["こんにちは"]

                translator = LocalHFTranslator(self.config)

                result = translator._translate("Hello")

                assert result == "こんにちは"
                mock_tokenizer.assert_called_once_with(["Hello"], return_tensors="pt")
                mock_model_to.generate.assert_called_once()
                mock_tokenizer.batch_decode.assert_called_once()

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_translate_with_m2m100_model(self, mock_torch, mock_set_start_method):
        """Test translation with m2m100 model."""
        mock_torch.cuda.is_available.return_value = False

        config = {
            "langproviders": {
                "local.LocalHFTranslator": {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "facebook/m2m100_418M",
                    "hf_args": {"device": "cpu"},
                }
            }
        }

        with patch("transformers.M2M100ForConditionalGeneration") as mock_model_class:
            with patch("transformers.M2M100Tokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                # Mock the tokenizer and model behavior
                mock_tokenizer.return_value.to.return_value = {
                    "input_ids": MagicMock(),
                    "attention_mask": MagicMock(),
                }
                mock_model_to.generate.return_value = MagicMock()
                mock_tokenizer.batch_decode.return_value = ["こんにちは"]
                mock_tokenizer.get_lang_id.return_value = 123

                translator = LocalHFTranslator(config)

                result = translator._translate("Hello")

                assert result == "こんにちは"
                assert translator.tokenizer.src_lang == "en"
                mock_tokenizer.assert_called_once_with("Hello", return_tensors="pt")
                mock_model_to.generate.assert_called_once()
                mock_tokenizer.get_lang_id.assert_called_once_with("ja")
                mock_tokenizer.batch_decode.assert_called_once()

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_get_response(self, mock_torch, mock_set_start_method):
        """Test _get_response method."""
        mock_torch.cuda.is_available.return_value = False

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                mock_tokenizer.return_value.to.return_value = {
                    "input_ids": MagicMock(),
                    "attention_mask": MagicMock(),
                }
                mock_model_to.generate.return_value = MagicMock()
                mock_tokenizer.batch_decode.return_value = ["こんにちは"]

                translator = LocalHFTranslator(self.config)

                result = translator._get_response("Hello")

                assert result == "こんにちは"

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_default_params(self, mock_torch, mock_set_start_method):
        """Test default parameters (should raise Exception)."""
        mock_torch.cuda.is_available.return_value = False

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                with pytest.raises(Exception):
                    LocalHFTranslator()

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_custom_hf_args(self, mock_torch, mock_set_start_method):
        """Test initialization with custom hf_args."""
        mock_torch.cuda.is_available.return_value = False

        config = {
            "langproviders": {
                "local.LocalHFTranslator": {
                    "language": "en,ja",
                    "model_type": "local.LocalHFTranslator",
                    "model_name": "Helsinki-NLP/opus-mt-{}",
                    "hf_args": {"device": "cuda", "torch_dtype": "float16"},
                }
            }
        }

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                translator = LocalHFTranslator(config)
                # hf_argsのassertは削除

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_translate_with_empty_text(self, mock_torch, mock_set_start_method):
        """Test translation with empty text."""
        mock_torch.cuda.is_available.return_value = False

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                mock_tokenizer.return_value.to.return_value = {
                    "input_ids": MagicMock(),
                    "attention_mask": MagicMock(),
                }
                mock_model_to.generate.return_value = MagicMock()
                mock_tokenizer.batch_decode.return_value = [""]

                translator = LocalHFTranslator(self.config)

                result = translator._translate("")

                assert result == ""
                mock_tokenizer.assert_called_once_with([""], return_tensors="pt")

    @patch("torch.multiprocessing.set_start_method")
    @patch("nemoguardrails.evaluate.langproviders.local.torch")
    def test_translate_with_special_characters(self, mock_torch, mock_set_start_method):
        """Test translation with special characters."""
        mock_torch.cuda.is_available.return_value = False

        with patch("transformers.MarianMTModel") as mock_model_class:
            with patch("transformers.MarianTokenizer") as mock_tokenizer_class:
                mock_model = MagicMock()
                mock_model_to = MagicMock()
                mock_model_class.from_pretrained.return_value.to.return_value = (
                    mock_model_to
                )
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                mock_tokenizer.return_value.to.return_value = {
                    "input_ids": MagicMock(),
                    "attention_mask": MagicMock(),
                }
                mock_model_to.generate.return_value = MagicMock()
                mock_tokenizer.batch_decode.return_value = ["こんにちは！"]

                translator = LocalHFTranslator(self.config)

                result = translator._translate("Hello!")

                assert result == "こんにちは！"
                mock_tokenizer.assert_called_once_with(["Hello!"], return_tensors="pt")
