# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import ValidationError

from nemoguardrails import RailsConfig
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.library.huggingface_detector.actions import (
    _load_model_and_tokenizer,
    _standardize_blocked_classes_to_indices,
    huggingface_detector_check,
)
from nemoguardrails.rails.llm.config import HuggingfaceDetectorConfig, HuggingfaceModelConfig

# Check for optional dependencies
try:
    import transformers  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch  # noqa: F401

    torch.tensor([])
    TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    TORCH_AVAILABLE = False


# Combined check for tests that require both
PREREQUISITES_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


@pytest.mark.skipif(
    not PREREQUISITES_AVAILABLE,
    reason="Transformers library required for HuggingfaceModelConfig tests",
)
class TestHuggingfaceModelConfig:
    """Tests for HuggingfaceModelConfig validation."""

    @patch("transformers.AutoConfig")
    def test_config_with_string_labels(self, mock_config_cls):
        """Test that config accepts list of string labels."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = HuggingfaceModelConfig(
            model_repo="test/model",
            blocked_classes=["harmful", "violence", "hate"],
        )
        assert config.blocked_classes == ["harmful", "violence", "hate"]

    @patch("transformers.AutoConfig")
    def test_config_with_integer_indices(self, mock_config_cls):
        """Test that config accepts list of integer indices."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = HuggingfaceModelConfig(
            model_repo="test/model",
            blocked_classes=[0, 1, 2],
        )
        assert config.blocked_classes == [0, 1, 2]

    def test_config_rejects_mixed_types(self):
        """Test that config rejects mixed strings and integers."""
        with pytest.raises(ValidationError) as excinfo:
            HuggingfaceModelConfig(
                model_repo="test/model",
                blocked_classes=["harmful", 1, "violence"],
            )
        # Pydantic's Union validation rejects mixed types
        assert "validation error" in str(excinfo.value).lower()

    @patch("transformers.AutoConfig")
    def test_config_empty_blocked_classes(self, mock_config_cls):
        """Test that empty blocked_classes is allowed."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = HuggingfaceModelConfig(
            model_repo="test/model",
            blocked_classes=[],
        )
        assert config.blocked_classes == []

    def test_config_rejects_invalid_types(self):
        """Test that config rejects invalid types in blocked_classes."""
        with pytest.raises(ValidationError) as excinfo:
            HuggingfaceModelConfig(
                model_repo="test/model",
                blocked_classes=[1.5, 2.5],
            )
        # Pydantic's Union validation rejects floats with fractional parts
        assert "validation error" in str(excinfo.value).lower()

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for model validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_validation_rejects_invalid_model_repo(self, mock_config_cls):
        """Test that validation catches invalid/inaccessible model repos."""
        # Simulate model not found error
        mock_config_cls.from_pretrained.side_effect = OSError("Repository not found")

        with pytest.raises((InvalidRailsConfigurationError, ValidationError)) as excinfo:
            HuggingfaceModelConfig(
                model_repo="invalid/model",
                blocked_classes=["harmful"],
            )
        error_msg = str(excinfo.value)
        assert "Failed to load model 'invalid/model' from Huggingface Hub" in error_msg
        assert "Repository not found" in error_msg

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for model validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_validation_rejects_non_sequence_classification_model(self, mock_config_cls):
        """Test that validation catches models that aren't sequence classification."""
        # Mock a model config that's not for sequence classification
        mock_config = MagicMock()
        mock_config.architectures = ["BertForMaskedLM", "BertModel"]
        mock_config_cls.from_pretrained.return_value = mock_config

        with pytest.raises((InvalidRailsConfigurationError, ValidationError)) as excinfo:
            HuggingfaceModelConfig(
                model_repo="test/non-classifier-model",
                blocked_classes=["harmful"],
            )
        error_msg = str(excinfo.value)
        assert "does not appear to be a sequence classification model" in error_msg
        assert "BertForMaskedLM" in error_msg

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for model validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_validation_accepts_valid_sequence_classification_model(self, mock_config_cls):
        """Test that validation passes for valid sequence classification models."""
        # Mock a valid sequence classification model config
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        # This should not raise an exception
        config = HuggingfaceModelConfig(
            model_repo="test/valid-classifier",
            blocked_classes=["harmful"],
        )
        assert config.model_repo == "test/valid-classifier"

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for model validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_validation_accepts_different_architectures(self, mock_config_cls):
        """Test that validation accepts various sequence classification architectures."""
        valid_architectures = [
            ["RobertaForSequenceClassification"],
            ["DistilBertForSequenceClassification"],
            ["AlbertForSequenceClassification"],
            ["XLMRobertaForSequenceClassification"],
        ]

        for arch in valid_architectures:
            mock_config = MagicMock()
            mock_config.architectures = arch
            mock_config_cls.from_pretrained.return_value = mock_config

            # This should not raise an exception
            config = HuggingfaceModelConfig(
                model_repo=f"test/{arch[0].lower()}",
                blocked_classes=["harmful"],
            )
            assert config.model_repo == f"test/{arch[0].lower()}"


@pytest.mark.skipif(
    not PREREQUISITES_AVAILABLE,
    reason="Torch and transformers are required for Hugging Face detectors, install with: pip install torch transformers",
)
class TestHuggingfaceDetectorConfig:
    """Tests for HuggingfaceDetectorConfig with multiple models."""

    @patch("transformers.AutoConfig")
    def test_config_with_single_model(self, mock_config_cls):
        """Test config with a single model."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = HuggingfaceDetectorConfig(
            models=[
                HuggingfaceModelConfig(
                    model_repo="test/model",
                    blocked_classes=["harmful", "violence"],
                )
            ]
        )
        assert len(config.models) == 1
        assert config.models[0].model_repo == "test/model"
        assert config.models[0].blocked_classes == ["harmful", "violence"]

    @patch("transformers.AutoConfig")
    def test_config_with_multiple_models(self, mock_config_cls):
        """Test config with multiple models."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = HuggingfaceDetectorConfig(
            models=[
                HuggingfaceModelConfig(
                    model_repo="model1/repo",
                    blocked_classes=["harmful"],
                ),
                HuggingfaceModelConfig(
                    model_repo="model2/repo",
                    blocked_classes=[0, 1],
                ),
            ]
        )
        assert len(config.models) == 2
        assert config.models[0].model_repo == "model1/repo"
        assert config.models[1].model_repo == "model2/repo"

    def test_config_with_empty_models_list(self):
        """Test that empty models list is allowed."""
        config = HuggingfaceDetectorConfig(models=[])
        assert config.models == []


class TestConvertBlockedClassesToIndices:
    """Tests for _standardize_blocked_classes_to_indices helper function."""

    def test_standardize_string_labels_to_indices(self):
        """Test conversion of string labels to indices."""
        label2id = {"safe": 0, "harmful": 1, "violence": 2}

        result = _standardize_blocked_classes_to_indices(["harmful", "violence"], label2id)
        assert result == {1, 2}

    def test_standardize_labels_using_label2id(self):
        """Test conversion with label2id dict."""
        label2id = {"safe": 0, "harmful": 1, "violence": 2}

        result = _standardize_blocked_classes_to_indices(["harmful"], label2id)
        assert result == {1}

    def test_standardize_integer_indices_directly(self):
        """Test that integer indices are returned as-is."""
        result = _standardize_blocked_classes_to_indices([0, 2, 5], None)
        assert result == {0, 2, 5}

    def test_standardize_empty_list(self):
        """Test that empty list returns empty set."""
        result = _standardize_blocked_classes_to_indices([], None)
        assert result == set()

    def test_error_when_label_not_found(self):
        """Test error when label is not in model's mapping."""
        label2id = {"safe": 0, "harmful": 1}

        with pytest.raises(ValueError) as excinfo:
            _standardize_blocked_classes_to_indices(["unknown_label"], label2id)
        assert "Class label 'unknown_label' not found" in str(excinfo.value)
        assert "Available labels:" in str(excinfo.value)

    def test_error_when_no_label_mappings_available(self):
        """Test error when model has no label mappings and strings are provided."""
        with pytest.raises(ValueError) as excinfo:
            _standardize_blocked_classes_to_indices(["harmful"], None)
        assert "Model does not provide label mappings" in str(excinfo.value)
        assert "use class indices instead" in str(excinfo.value)


@pytest.mark.skipif(
    not PREREQUISITES_AVAILABLE,
    reason="Torch and transformers are required for Hugging Face detectors, install with: pip install torch transformers",
)
@pytest.mark.asyncio
class TestHuggingfaceDetectorActions:
    """Tests for the main detector action functions."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        import torch

        mock_tokenizer = MagicMock()
        # Return torch tensors instead of lists
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "safe", 1: "harmful", 2: "violence"}
        mock_model.config.label2id = {"safe": 0, "harmful": 1, "violence": 2}

        # Mock torch tensor operations
        logits = torch.tensor([[2.0, 5.0, 1.0]])  # Highest score for index 1 (harmful)

        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs

        # Mock parameters() to return an iterator with a device
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        return mock_model, mock_tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock RailsConfig."""
        with patch("transformers.AutoConfig") as mock_config_cls:
            # Mock the model config to have sequence classification architecture
            mock_config_obj = MagicMock()
            mock_config_obj.architectures = ["BertForSequenceClassification"]
            mock_config_cls.from_pretrained.return_value = mock_config_obj

            config = RailsConfig.from_content(
                yaml_content="""
                    models: []
                    rails:
                      config:
                        huggingface_detector:
                          models:
                            - model_repo: "test/model"
                              blocked_classes:
                                - "harmful"
                                - "violence"
                """,
                colang_content="",
            )
            return config

    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_check_input_blocks_harmful_content(self, mock_load, mock_model_and_tokenizer, mock_config):
        """Test that harmful input is blocked."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_load.return_value = (mock_model, mock_tokenizer)

        result = await huggingface_detector_check(
            text="harmful text",
            model_repo="test/model",
            config=mock_config,
        )

        assert result["allowed"] is False
        assert result["detected_class"] == "harmful"
        assert "all_scores" in result
        assert "score" in result

    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_check_input_allows_safe_content(self, mock_load, mock_config):
        """Test that safe input is allowed."""
        import torch

        # Create a mock that predicts "safe" (index 0)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "safe", 1: "harmful", 2: "violence"}
        mock_model.config.label2id = {"safe": 0, "harmful": 1, "violence": 2}

        logits = torch.tensor([[5.0, 1.0, 1.0]])  # Highest score for index 0 (safe)

        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs

        # Mock parameters() to return an iterator with a device
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_load.return_value = (mock_model, mock_tokenizer)

        result = await huggingface_detector_check(
            text="safe text",
            model_repo="test/model",
            config=mock_config,
        )

        assert result["allowed"] is True
        assert result["detected_class"] == "safe"

    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_check_output_blocks_harmful_content(self, mock_load, mock_model_and_tokenizer, mock_config):
        """Test that harmful output is blocked."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_load.return_value = (mock_model, mock_tokenizer)

        result = await huggingface_detector_check(
            text="harmful response",
            model_repo="test/model",
            config=mock_config,
        )

        assert result["allowed"] is False
        assert result["detected_class"] == "harmful"

    async def test_check_input_missing_config(self):
        """Test that missing config raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            await huggingface_detector_check(
                text="test",
                model_repo="test/model",
                config=None,
            )
        assert "configuration is required" in str(excinfo.value)

    @patch("transformers.AutoConfig")
    async def test_check_input_missing_text(self, mock_config_cls):
        """Test that missing text raises ValueError."""
        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: ["harmful"]
            """,
            colang_content="",
        )

        with pytest.raises(ValueError) as excinfo:
            await huggingface_detector_check(
                context_key="user_message",
                model_repo="test/model",
                config=config,
                context={},
            )
        assert "No text provided" in str(excinfo.value)

    @patch("transformers.AutoConfig")
    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_check_input_with_indices(self, mock_load, mock_config_cls):
        """Test detector with integer indices instead of labels."""
        import torch

        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        # Create config with integer indices
        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: [1, 2]
            """,
            colang_content="",
        )

        # Mock model without labels
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        # Remove id2label to simulate model without label mappings
        del mock_model.config.id2label

        logits = torch.tensor([[1.0, 5.0, 2.0]])  # Highest score for index 1

        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs

        # Mock parameters() to return an iterator with a device
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_load.return_value = (mock_model, mock_tokenizer)

        result = await huggingface_detector_check(
            text="test",
            model_repo="test/model",
            config=config,
        )

        # Should be blocked because index 1 is in blocked_classes
        assert result["allowed"] is False
        # Without id2label, detected_class should be string of index
        assert result["detected_class"] == "1"

    @patch("transformers.AutoConfig")
    async def test_missing_model_repo_parameter(self, mock_config_cls):
        """Test that missing model_repo parameter raises ValueError."""
        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: ["harmful"]
            """,
            colang_content="",
        )

        with pytest.raises(ValueError) as excinfo:
            await huggingface_detector_check(
                text="test",
                config=config,
            )
        assert "model_repo parameter is required" in str(excinfo.value)

    @patch("transformers.AutoConfig")
    async def test_model_not_in_config(self, mock_config_cls):
        """Test that specifying a model not in config raises ValueError."""
        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "model1/repo"
                          blocked_classes: ["harmful"]
            """,
            colang_content="",
        )

        with pytest.raises(ValueError) as excinfo:
            await huggingface_detector_check(
                text="test",
                model_repo="model2/repo",
                config=config,
            )
        assert "not present in the huggingface_detector configuration" in str(excinfo.value)
        assert "model1/repo" in str(excinfo.value)

    @patch("transformers.AutoConfig")
    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_device_passed_to_model_loader(self, mock_load, mock_config_cls):
        """Test that device from config is passed to model loader."""
        import torch

        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: ["harmful"]
                          device: "cuda"
            """,
            colang_content="",
        )

        # Mock model and tokenizer
        mock_tokenizer = MagicMock()
        # Create a mock tensor that handles .to() method
        mock_input_tensor = MagicMock()
        mock_input_tensor.to.return_value = mock_input_tensor
        mock_tokenizer.return_value = {"input_ids": mock_input_tensor}

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "safe", 1: "harmful"}
        mock_model.config.label2id = {"safe": 0, "harmful": 1}

        logits = torch.tensor([[5.0, 1.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs

        # Mock parameters() to return an iterator with a cuda device
        mock_param = MagicMock()
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_param.device = mock_device
        mock_model.parameters.return_value = iter([mock_param])

        mock_load.return_value = (mock_model, mock_tokenizer)

        await huggingface_detector_check(
            text="test",
            model_repo="test/model",
            config=config,
        )

        # Verify _load_model_and_tokenizer was called with device="cuda"
        mock_load.assert_called_once_with("test/model", device="cuda")

    @patch("transformers.AutoConfig")
    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_device_cpu_passed_to_model_loader(self, mock_load, mock_config_cls):
        """Test that CPU device from config is passed to model loader."""
        import torch

        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: ["harmful"]
                          device: "cpu"
            """,
            colang_content="",
        )

        # Mock model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "safe", 1: "harmful"}
        mock_model.config.label2id = {"safe": 0, "harmful": 1}

        logits = torch.tensor([[5.0, 1.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs

        # Mock parameters() to return an iterator with a cpu device
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_load.return_value = (mock_model, mock_tokenizer)

        await huggingface_detector_check(
            text="test",
            model_repo="test/model",
            config=config,
        )

        # Verify _load_model_and_tokenizer was called with device="cpu"
        mock_load.assert_called_once_with("test/model", device="cpu")

    @patch("transformers.AutoConfig")
    @patch("nemoguardrails.library.huggingface_detector.actions._load_model_and_tokenizer")
    async def test_no_device_defaults_to_none(self, mock_load, mock_config_cls):
        """Test that when device is not specified, None is passed to loader."""
        import torch

        # Mock the model config to have sequence classification architecture
        mock_config_obj = MagicMock()
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config_obj

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: ["harmful"]
            """,
            colang_content="",
        )

        # Mock model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "safe", 1: "harmful"}
        mock_model.config.label2id = {"safe": 0, "harmful": 1}

        logits = torch.tensor([[5.0, 1.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model.return_value = mock_outputs

        # Mock parameters() to return an iterator with a device
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        mock_load.return_value = (mock_model, mock_tokenizer)

        await huggingface_detector_check(
            text="test",
            model_repo="test/model",
            config=config,
        )

        # Verify _load_model_and_tokenizer was called with device=None
        mock_load.assert_called_once_with("test/model", device=None)


@pytest.mark.skipif(
    not PREREQUISITES_AVAILABLE,
    reason="Torch and transformers are required for Hugging Face detectors, install with: pip install torch transformers",
)
class TestLoadModelAndTokenizer:
    """Tests for _load_model_and_tokenizer function with device parameter."""

    @patch("transformers.AutoModelForSequenceClassification")
    @patch("transformers.AutoTokenizer")
    def test_model_moved_to_cuda_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test that model is moved to CUDA device when specified."""
        from nemoguardrails.library.huggingface_detector.actions import _model_cache

        # Clear cache before test
        _model_cache.clear()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_and_tokenizer("test/model", device="cuda")

        # Verify model.to() was called with "cuda"
        mock_model.to.assert_called_once_with("cuda")
        assert model == mock_model.to.return_value
        assert tokenizer == mock_tokenizer

    @patch("transformers.AutoModelForSequenceClassification")
    @patch("transformers.AutoTokenizer")
    def test_model_moved_to_cpu_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test that model is moved to CPU device when specified."""
        from nemoguardrails.library.huggingface_detector.actions import _model_cache

        _model_cache.clear()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_and_tokenizer("test/model", device="cpu")

        mock_model.to.assert_called_once_with("cpu")
        assert model == mock_model.to.return_value

    @patch("transformers.AutoModelForSequenceClassification")
    @patch("transformers.AutoTokenizer")
    def test_model_not_moved_when_device_none(self, mock_tokenizer_cls, mock_model_cls):
        """Test that model is not moved when device is None."""
        from nemoguardrails.library.huggingface_detector.actions import _model_cache

        _model_cache.clear()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_and_tokenizer("test/model", device=None)

        # Verify model.to() was NOT called
        mock_model.to.assert_not_called()
        assert model == mock_model

    @patch("transformers.AutoModelForSequenceClassification")
    @patch("transformers.AutoTokenizer")
    def test_model_cached_with_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test that models are cached separately for different devices."""
        from nemoguardrails.library.huggingface_detector.actions import _model_cache

        _model_cache.clear()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model_cuda = MagicMock()
        mock_model_cpu = MagicMock()

        # First call with CUDA
        mock_model_cls.from_pretrained.return_value = mock_model_cuda
        model1, _ = _load_model_and_tokenizer("test/model", device="cuda")

        # Second call with CPU
        mock_model_cls.from_pretrained.return_value = mock_model_cpu
        model2, _ = _load_model_and_tokenizer("test/model", device="cpu")

        # Third call with CUDA again (should use cache)
        model3, _ = _load_model_and_tokenizer("test/model", device="cuda")

        # Verify model was loaded twice (once for cuda, once for cpu)
        assert mock_model_cls.from_pretrained.call_count == 2

        # Verify third call returned cached model
        assert model3 == model1

    @patch("transformers.AutoModelForSequenceClassification")
    @patch("transformers.AutoTokenizer")
    def test_model_moved_to_specific_cuda_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test that model can be moved to specific CUDA device like cuda:1."""
        from nemoguardrails.library.huggingface_detector.actions import _model_cache

        _model_cache.clear()

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_and_tokenizer("test/model", device="cuda:1")

        mock_model.to.assert_called_once_with("cuda:1")


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="Torch library required for TestRailsConfigIntegration tests",
)
class TestRailsConfigIntegration:
    """Integration tests with RailsConfig."""

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_config_from_yaml_with_single_model(self, mock_config_cls):
        """Test loading config from YAML with a single model."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "ibm-granite/granite-guardian-hap-38m"
                          blocked_classes:
                            - "harmful"
                            - "violence"
            """,
            colang_content="",
        )

        assert len(config.rails.config.huggingface_detector.models) == 1
        assert config.rails.config.huggingface_detector.models[0].model_repo == "ibm-granite/granite-guardian-hap-38m"
        assert config.rails.config.huggingface_detector.models[0].blocked_classes == ["harmful", "violence"]

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_config_from_yaml_with_multiple_models(self, mock_config_cls):
        """Test loading config from YAML with multiple models."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "model1/repo"
                          blocked_classes: ["harmful"]
                        - model_repo: "model2/repo"
                          blocked_classes: [0, 1, 2]
            """,
            colang_content="",
        )

        assert len(config.rails.config.huggingface_detector.models) == 2
        assert config.rails.config.huggingface_detector.models[0].model_repo == "model1/repo"
        assert config.rails.config.huggingface_detector.models[0].blocked_classes == ["harmful"]
        assert config.rails.config.huggingface_detector.models[1].model_repo == "model2/repo"
        assert config.rails.config.huggingface_detector.models[1].blocked_classes == [0, 1, 2]

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    def test_config_from_yaml_mixed_types_fails(self):
        """Test that mixed types in YAML config fail validation."""
        with pytest.raises(ValidationError):
            RailsConfig.from_content(
                yaml_content="""
                    models: []
                    rails:
                      config:
                        huggingface_detector:
                          models:
                            - model_repo: "test/model"
                              blocked_classes: ["harmful", 1, 2]
                """,
                colang_content="",
            )

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_config_from_yaml_with_device(self, mock_config_cls):
        """Test loading config from YAML with device specified."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "test/model"
                          blocked_classes: ["harmful"]
                          device: "cuda"
            """,
            colang_content="",
        )

        assert len(config.rails.config.huggingface_detector.models) == 1
        assert config.rails.config.huggingface_detector.models[0].device == "cuda"

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_config_from_yaml_with_multiple_devices(self, mock_config_cls):
        """Test loading config with different devices for different models."""
        # Mock the model config to have sequence classification architecture
        mock_config = MagicMock()
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_config_cls.from_pretrained.return_value = mock_config

        config = RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    huggingface_detector:
                      models:
                        - model_repo: "model1/repo"
                          blocked_classes: ["harmful"]
                          device: "cuda"
                        - model_repo: "model2/repo"
                          blocked_classes: [0, 1]
                          device: "cpu"
                        - model_repo: "model3/repo"
                          blocked_classes: ["jailbreak"]
            """,
            colang_content="",
        )

        assert len(config.rails.config.huggingface_detector.models) == 3
        assert config.rails.config.huggingface_detector.models[0].device == "cuda"
        assert config.rails.config.huggingface_detector.models[1].device == "cpu"
        assert config.rails.config.huggingface_detector.models[2].device is None

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_config_validation_fails_for_invalid_model(self, mock_config_cls):
        """Test that config validation catches invalid models at config load time."""
        # Simulate model not found error
        mock_config_cls.from_pretrained.side_effect = OSError("Repository not found")

        with pytest.raises((InvalidRailsConfigurationError, ValidationError)) as excinfo:
            RailsConfig.from_content(
                yaml_content="""
                    models: []
                    rails:
                      config:
                        huggingface_detector:
                          models:
                            - model_repo: "invalid/model"
                              blocked_classes: ["harmful"]
                """,
                colang_content="",
            )
        error_msg = str(excinfo.value)
        assert "Failed to load model 'invalid/model' from Huggingface Hub" in error_msg

    @pytest.mark.skipif(
        not TRANSFORMERS_AVAILABLE,
        reason="Transformers library required for config validation tests",
    )
    @patch("transformers.AutoConfig")
    def test_config_validation_fails_for_non_classifier_model(self, mock_config_cls):
        """Test that config validation catches non-classifier models."""
        # Mock a model config that's not for sequence classification
        mock_config = MagicMock()
        mock_config.architectures = ["BertForMaskedLM"]
        mock_config_cls.from_pretrained.return_value = mock_config

        with pytest.raises((InvalidRailsConfigurationError, ValidationError)) as excinfo:
            RailsConfig.from_content(
                yaml_content="""
                    models: []
                    rails:
                      config:
                        huggingface_detector:
                          models:
                            - model_repo: "test/masked-lm-model"
                              blocked_classes: ["harmful"]
                """,
                colang_content="",
            )
        error_msg = str(excinfo.value)
        assert "does not appear to be a sequence classification model" in error_msg
