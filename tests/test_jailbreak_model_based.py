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

import sys
import types
from unittest import mock

import pytest

# Test 1: Lazy import behavior


def test_lazy_import_does_not_require_heavy_deps():
    """
    Importing the checks module should not require torch, transformers, or sklearn unless model-based classifier is used.
    """
    with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "sklearn": None}):
        import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

        # Just importing and calling unrelated functions should not raise ImportError
        assert hasattr(checks, "initialize_model")


# Test 2: Model-based classifier instantiation requires dependencies


def test_model_based_classifier_imports(monkeypatch):
    """
    Instantiating JailbreakClassifier should require sklearn and pickle, and use SnowflakeEmbed which requires torch/transformers.
    """
    # Mock dependencies
    fake_rf = mock.MagicMock()
    fake_embed = mock.MagicMock(return_value=[0.0])
    fake_pickle = types.SimpleNamespace(load=mock.MagicMock(return_value=fake_rf))
    fake_snowflake = mock.MagicMock(return_value=fake_embed)

    monkeypatch.setitem(
        sys.modules,
        "sklearn.ensemble",
        types.SimpleNamespace(RandomForestClassifier=mock.MagicMock()),
    )
    monkeypatch.setitem(sys.modules, "pickle", fake_pickle)
    monkeypatch.setitem(sys.modules, "torch", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", mock.MagicMock())

    # Patch SnowflakeEmbed to avoid real model loading
    import nemoguardrails.library.jailbreak_detection.model_based.models as models

    monkeypatch.setattr(models, "SnowflakeEmbed", fake_snowflake)

    # mocking file operations to avoid Windows permission issues
    mock_open = mock.mock_open()
    with mock.patch("builtins.open", mock_open):
        # Should not raise
        classifier = models.JailbreakClassifier("fake_model_path.pkl")
        assert classifier is not None
        # Should be callable
        result = classifier("test")
        assert isinstance(result, tuple)


# Test 3: Error if dependencies missing when instantiating model-based classifier


def test_model_based_classifier_missing_deps(monkeypatch):
    """
    If sklearn is missing, instantiating JailbreakClassifier should raise ImportError.
    """
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)

    import nemoguardrails.library.jailbreak_detection.model_based.models as models

    # to avoid Windows permission issues
    mock_open = mock.mock_open()
    with mock.patch("builtins.open", mock_open):
        with pytest.raises(ImportError):
            models.JailbreakClassifier("fake_model_path.pkl")


# Test 4: Return None when EMBEDDING_CLASSIFIER_PATH is not set


def test_initialize_model_with_none_classifier_path(monkeypatch):
    """
    initialize_model should return None when EMBEDDING_CLASSIFIER_PATH is not set.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    # Clear the LRU cache to ensure fresh test
    checks.initialize_model.cache_clear()

    # Mock environment variable to be None
    monkeypatch.setenv("EMBEDDING_CLASSIFIER_PATH", "")
    monkeypatch.delenv("EMBEDDING_CLASSIFIER_PATH", raising=False)

    result = checks.initialize_model()
    assert result is None


# Test 5: SnowflakeEmbed initialization and call with torch imports


def test_snowflake_embed_torch_imports(monkeypatch):
    """
    Test that SnowflakeEmbed properly imports torch and transformers when needed.
    """
    # Mock torch and transformers
    mock_torch = mock.MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_transformers = mock.MagicMock()

    mock_tokenizer = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    import nemoguardrails.library.jailbreak_detection.model_based.models as models

    embed = models.SnowflakeEmbed()
    assert embed.device == "cpu"  # as we mocked cuda.is_available() = False

    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock_tokens
    mock_tokenizer.return_value = mock_tokens

    import numpy as np

    fake_embedding = np.array([1.0, 2.0, 3.0])

    # the code does self.model(**tokens)[0][:, 0]
    # so we need to mock this properly
    mock_tensor_output = mock.MagicMock()
    mock_tensor_output.detach.return_value.cpu.return_value.squeeze.return_value.numpy.return_value = fake_embedding

    mock_first_index = mock.MagicMock()
    mock_first_index.__getitem__.return_value = mock_tensor_output  # for [:, 0]

    mock_model_output = mock.MagicMock()
    mock_model_output.__getitem__.return_value = mock_first_index  # for [0]

    mock_model.return_value = mock_model_output

    result = embed("test text")
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, fake_embedding)


# Test 6: Check jailbreak function with classifier parameter


def test_check_jailbreak_with_classifier():
    """
    Test check_jailbreak function when classifier is provided.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    mock_classifier = mock.MagicMock()
    # jailbreak detected with score 0.9
    mock_classifier.return_value = (True, 0.9)

    result = checks.check_jailbreak("test prompt", classifier=mock_classifier)

    assert result == {"jailbreak": True, "score": 0.9}
    mock_classifier.assert_called_once_with("test prompt")


# Test 7: Check jailbreak function without classifier parameter (uses initialize_model)


def test_check_jailbreak_without_classifier(monkeypatch):
    """
    Test check_jailbreak function when no classifier is provided, it should call initialize_model.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    # mock initialize_model to return a mock classifier
    mock_classifier = mock.MagicMock()
    # no jailbreak
    mock_classifier.return_value = (False, -0.5)
    mock_initialize_model = mock.MagicMock(return_value=mock_classifier)

    monkeypatch.setattr(checks, "initialize_model", mock_initialize_model)

    result = checks.check_jailbreak("safe prompt")

    assert result == {"jailbreak": False, "score": -0.5}
    mock_initialize_model.assert_called_once()
    mock_classifier.assert_called_once_with("safe prompt")


# Test 8: Check jailbreak raises RuntimeError when no classifier available


def test_check_jailbreak_no_classifier_available(monkeypatch):
    """
    Test check_jailbreak function raises RuntimeError when initialize_model returns None.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    # Mock initialize_model to return None (no classifier available)
    mock_initialize_model = mock.MagicMock(return_value=None)
    monkeypatch.setattr(checks, "initialize_model", mock_initialize_model)

    with pytest.raises(RuntimeError) as exc_info:
        checks.check_jailbreak("test prompt")

    assert "No jailbreak classifier available" in str(exc_info.value)
    assert "EMBEDDING_CLASSIFIER_PATH" in str(exc_info.value)
    mock_initialize_model.assert_called_once()


# Test 9: Test initialize_model with valid path


def test_initialize_model_with_valid_path(monkeypatch):
    """
    Test initialize_model with a valid classifier path.
    """
    from pathlib import Path

    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    checks.initialize_model.cache_clear()

    # mock environment variable
    test_path = "/fake/path/to/model"
    monkeypatch.setenv("EMBEDDING_CLASSIFIER_PATH", test_path)

    # mock JailbreakClassifier
    mock_classifier = mock.MagicMock()
    mock_jailbreak_classifier_class = mock.MagicMock(return_value=mock_classifier)
    monkeypatch.setattr(
        "nemoguardrails.library.jailbreak_detection.model_based.models.JailbreakClassifier",
        mock_jailbreak_classifier_class,
    )

    result = checks.initialize_model()

    assert result == mock_classifier

    expected_path = str(Path(test_path).joinpath("snowflake.pkl"))
    mock_jailbreak_classifier_class.assert_called_once_with(expected_path)


# Test 10: Test that NvEmbedE5 class no longer exists


def test_nv_embed_e5_removed():
    """
    Test that NvEmbedE5 class has been removed from the models module.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.models as models

    assert not hasattr(models, "NvEmbedE5")


# Test 11: Test SnowflakeEmbed still exists and works


def test_snowflake_embed_still_available():
    """
    Test that SnowflakeEmbed class is still available.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.models as models

    # This class should still exist
    assert hasattr(models, "SnowflakeEmbed")


# Test 12: Test initialize_model with logging


def test_initialize_model_logging(monkeypatch, caplog):
    """
    Test that initialize_model logs warning when path is not set.
    """
    import logging

    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    # clear the LRU cache to ensure fresh test
    checks.initialize_model.cache_clear()

    # set log level to capture warnings
    caplog.set_level(logging.WARNING)

    # mock environment variable to be None
    monkeypatch.delenv("EMBEDDING_CLASSIFIER_PATH", raising=False)

    result = checks.initialize_model()

    assert result is None
    assert "No embedding classifier path set" in caplog.text
    assert "Server /model endpoint will not work" in caplog.text


# Test 13: Test check_jailbreak with explicit None classifier


def test_check_jailbreak_explicit_none_classifier():
    """
    Test check_jailbreak when explicitly passed None as classifier.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    with pytest.raises(RuntimeError) as exc_info:
        checks.check_jailbreak("test prompt", classifier=None)

    assert "No jailbreak classifier available" in str(exc_info.value)


# Test 14: Test check_jailbreak preserves original behavior with valid classifier


def test_check_jailbreak_valid_classifier_preserved():
    """
    Test that check_jailbreak still works normally with a valid classifier.
    """
    import nemoguardrails.library.jailbreak_detection.model_based.checks as checks

    mock_classifier = mock.MagicMock()
    mock_classifier.return_value = (True, 0.95)

    result = checks.check_jailbreak("malicious prompt", classifier=mock_classifier)

    assert result == {"jailbreak": True, "score": 0.95}
    mock_classifier.assert_called_once_with("malicious prompt")
