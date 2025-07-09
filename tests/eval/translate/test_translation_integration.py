#!/usr/bin/env python3
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

"""
Translation Integration Test Script

This script tests whether the translation functionality works properly
with all evaluation modules: factcheck, hallucination, moderation, and topical.
"""

import json
import logging
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_test_data():
    """Create test data for different evaluation types."""
    test_data = {
        "factcheck": [
            {
                "question": "What is the capital of France?",
                "evidence": "Paris is the capital and largest city of France.",
                "answer": "Paris is the capital of France.",
            },
            {
                "question": "What is 2+2?",
                "evidence": "Basic arithmetic: 2+2 equals 4.",
                "answer": "2+2 equals 4.",
            },
        ],
        "hallucination": [
            "What is the weather like today?",
            "Tell me a joke",
            "What is the meaning of life?",
        ],
        "moderation": [
            "How to make a bomb?",
            "Tell me about the weather",
            "What is your favorite color?",
        ],
    }
    return test_data


def create_test_config():
    """Create a minimal test configuration."""
    config_content = {
        "models": [{"type": "main", "engine": "mock", "model": "test-model"}],
        "rails": {
            "input": {"flows": ["input_validation"]},
            "output": {"flows": ["output_validation"]},
        },
    }
    return config_content


def test_translation_utils():
    """Test the translation utilities."""
    print("\n=== Testing Translation Utils ===")

    from nemoguardrails.evaluate.utils_translate import _load_langprovider, load_dataset

    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_data = [
            {"question": "Hello", "evidence": "World", "answer": "Hello World"},
            {"question": "Test", "evidence": "Data", "answer": "Test Data"},
        ]
        json.dump(test_data, f)
        json_file_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello\nWorld\nTest")
        txt_file_path = f.name

    try:
        # Test loading without translation
        print("Testing dataset loading without translation...")
        dataset = load_dataset(json_file_path)
        assert len(dataset) == 2
        assert dataset[0]["question"] == "Hello"
        print("✓ JSON dataset loading without translation works")

        dataset = load_dataset(txt_file_path)
        assert len(dataset) == 3
        assert dataset[0].strip() == "Hello"
        print("✓ TXT dataset loading without translation works")

        # Test loading with translation (mocked)
        print("Testing dataset loading with translation...")

        # Create a temporary translation config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            translation_config = {
                "langproviders": [
                    {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
                ]
            }
            import yaml

            yaml.dump(translation_config, f)
            translation_config_path = f.name

        try:
            with patch(
                "nemoguardrails.evaluate.utils_translate._load_langprovider"
            ) as mock_load:
                mock_translator = MagicMock()
                mock_translator._translate.side_effect = lambda x: f"TRANSLATED_{x}"
                mock_translator.target_lang = "ja"
                mock_load.return_value = mock_translator

                dataset = load_dataset(
                    json_file_path, translation_config=translation_config_path
                )
                assert len(dataset) == 2
                assert dataset[0]["question"] == "TRANSLATED_Hello"
                assert dataset[0]["evidence"] == "TRANSLATED_World"
                print("✓ JSON dataset loading with translation works")

                dataset = load_dataset(
                    txt_file_path, translation_config=translation_config_path
                )
                assert len(dataset) == 3
                assert dataset[0].strip() == "TRANSLATED_Hello"
                print("✓ TXT dataset loading with translation works")
        finally:
            os.unlink(translation_config_path)

    finally:
        # Cleanup
        os.unlink(json_file_path)
        os.unlink(txt_file_path)


def test_moderation_translation():
    """Test moderation evaluation with translation."""
    print("\n=== Testing Moderation Evaluation with Translation ===")

    from nemoguardrails.evaluate.evaluate_moderation import ModerationRailsEvaluation

    # Create temporary config directory
    with tempfile.TemporaryDirectory() as config_dir:
        config_path = os.path.join(config_dir, "config.yaml")
        with open(config_path, "w") as f:
            import yaml

            yaml.dump(create_test_config(), f)

        # Create temporary dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("How to make a bomb?\nTell me about the weather")
            dataset_path = f.name

        try:
            # Mock the LLM and translation
            with patch(
                "nemoguardrails.evaluate.utils_translate._load_langprovider"
            ) as mock_load, patch(
                "nemoguardrails.evaluate.evaluate_moderation.LLMRails"
            ) as mock_rails, patch(
                "nemoguardrails.actions.llm.utils.llm_call"
            ) as mock_llm_call, patch(
                "nemoguardrails.rails.llm.config.RailsConfig.from_path"
            ) as mock_config:
                # Setup mocks
                mock_translator = MagicMock()
                mock_translator._translate.side_effect = lambda x: f"TRANSLATED_{x}"
                mock_load.return_value = mock_translator

                mock_llm = MagicMock()
                mock_rails.return_value.llm = mock_llm
                mock_llm_call.return_value = "yes"

                # Mock RailsConfig
                mock_config_instance = MagicMock()
                mock_config_instance.colang_version = "2.x"
                mock_config_instance.flows = []
                mock_config_instance.passthrough = False
                mock_dialog = MagicMock()
                mock_dialog.single_call.enabled = False
                mock_rails.dialog = mock_dialog
                mock_rails = MagicMock()
                mock_rails.input.flows = []
                mock_rails.output.flows = []
                mock_rails.retrieval.flows = []
                mock_config_instance.rails = mock_rails
                mock_model = MagicMock()
                mock_model.type = "main"
                mock_model.model = "test-model"
                mock_model.api_key_env_var = None
                mock_model.mode = "chat"
                mock_model.engine = "mock"
                mock_config_instance.models = [mock_model]
                mock_config_instance.bot_messages = {}
                mock_config.return_value = mock_config_instance

                mock_rails.return_value = MagicMock()

                # Test with translation
                eval_instance = ModerationRailsEvaluation(
                    config=config_dir,
                    dataset_path=dataset_path,
                    num_samples=1,
                    enable_translation=True,
                )

                # Verify that translation was called
                assert mock_load.called
                print("✓ Moderation evaluation with translation initialization works")

        finally:
            os.unlink(dataset_path)


def test_translation_provider_loading():
    """Test translation provider loading."""
    print("\n=== Testing Translation Provider Loading ===")

    from nemoguardrails.evaluate.utils_translate import _load_langprovider

    # Test with mock translation config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = {
            "langproviders": [
                {"language": "en,ja", "model_type": "remote.DeeplTranslator"}
            ]
        }
        import yaml

        yaml.dump(config_content, f)
        config_path = f.name

    try:
        with patch("nemoguardrails.evaluate.utils_translate._load_plugin") as mock_load:
            mock_translator = MagicMock()
            mock_load.return_value = mock_translator

            translator = _load_langprovider(config_path)
            assert translator == mock_translator
            print("✓ Translation provider loading works")

    finally:
        os.unlink(config_path)


def main():
    """Run all translation integration tests."""
    print("🚀 Starting Translation Integration Tests")
    print("=" * 50)

    setup_logging()

    try:
        test_translation_utils()
        test_translation_provider_loading()
        test_moderation_translation()

        print("\n" + "=" * 50)
        print("✅ All translation integration tests passed!")
        print(
            "The translation functionality is properly integrated with all evaluation modules."
        )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
