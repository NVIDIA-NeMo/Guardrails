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

import re
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.benchmark.mock_llm_server.config import ModelSettings
from nemoguardrails.benchmark.mock_llm_server.response_data import (
    calculate_tokens,
    generate_id,
    get_latency_seconds,
    get_response,
    is_unsafe,
)


class TestGenerateId:
    """Test the generate_id function."""

    def test_generate_id_default_prefix(self):
        """Test generating ID with default prefix."""
        id1 = generate_id()
        assert id1.startswith("chatcmpl-")
        # ID should be in format: prefix-{8 hex chars}
        assert len(id1) == len("chatcmpl-") + 8

    def test_generate_id_custom_prefix(self):
        """Test generating ID with custom prefix."""
        id1 = generate_id("cmpl")
        assert id1.startswith("cmpl-")
        assert len(id1) == len("cmpl-") + 8

    def test_generate_id_format(self):
        """Test that generated IDs have correct format."""
        id1 = generate_id("test")
        # Should match pattern: prefix-{8 hex chars}
        pattern = r"test-[0-9a-f]{8}"
        assert re.match(pattern, id1)


class TestCalculateTokens:
    """Test the calculate_tokens function."""

    def test_calculate_tokens_empty_string(self):
        """Test calculating tokens for empty string."""
        tokens = calculate_tokens("")
        assert tokens == 1  # Returns at least 1

    def test_calculate_tokens_short_text(self):
        """Test calculating tokens for short text."""
        tokens = calculate_tokens("Hi")
        # 2 chars / 4 = 0, but max(1, 0) = 1
        assert tokens == 1

    def test_calculate_tokens_exact_division(self):
        """Test calculating tokens for text divisible by 4."""
        text = "a" * 20  # 20 chars / 4 = 5 tokens
        tokens = calculate_tokens(text)
        assert tokens == 5

    def test_calculate_tokens_with_remainder(self):
        """Test calculating tokens for text with remainder."""
        text = "a" * 19  # 19 chars / 4 = 4 (integer division)
        tokens = calculate_tokens(text)
        assert tokens == 4

    def test_calculate_tokens_long_text(self):
        """Test calculating tokens for long text."""
        text = "This is a longer text that should have multiple tokens." * 10
        tokens = calculate_tokens(text)
        expected = max(1, len(text) // 4)
        assert tokens == expected

    def test_calculate_tokens_unicode(self):
        """Test calculating tokens with unicode characters."""
        text = "Hello 世界 🌍"
        tokens = calculate_tokens(text)
        assert tokens >= 1
        assert tokens == max(1, len(text) // 4)


@pytest.fixture
def model_settings() -> ModelSettings:
    """Generate config data for use in response generation"""
    settings = ModelSettings(
        model="gpt-4o",
        unsafe_probability=0.5,
        unsafe_text="Sorry Dave, I'm afraid I can't do that.",
        safe_text="I'm an AI assistant and am happy to help",
        latency_min_seconds=0.2,
        latency_max_seconds=1.0,
        latency_mean_seconds=0.5,
        latency_std_seconds=0.1,
    )
    return settings


@pytest.fixture
def random_seed() -> int:
    """Return a fixed seed number for all tests"""
    return 12345


@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.seed")
@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.binomial")
def test_is_unsafe_mocks_no_seed(mock_binomial: MagicMock, mock_seed: MagicMock, model_settings: ModelSettings):
    """Check `is_unsafe()` calls the correct numpy functions"""
    mock_binomial.return_value = [True]

    response = is_unsafe(model_settings)

    assert response
    assert mock_seed.call_count == 0
    assert mock_binomial.call_count == 1
    mock_binomial.assert_called_once_with(n=1, p=model_settings.unsafe_probability, size=1)


@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.seed")
@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.binomial")
def test_is_unsafe_mocks_with_seed(mock_binomial, mock_seed, model_settings: ModelSettings, random_seed: int):
    """Check `is_unsafe()` calls the correct numpy functions"""
    mock_binomial.return_value = [False]

    response = is_unsafe(model_settings, random_seed)

    assert not response
    assert mock_seed.call_count == 1
    assert mock_binomial.call_count == 1
    mock_binomial.assert_called_once_with(n=1, p=model_settings.unsafe_probability, size=1)


def test_is_unsafe_prob_one(model_settings: ModelSettings):
    """Check `is_unsafe()` with probability of 1 returns True"""

    model_settings.unsafe_probability = 1.0
    response = is_unsafe(model_settings)
    assert response


def test_is_unsafe_prob_zero(model_settings: ModelSettings):
    """Check `is_unsafe()` with probability of 0 returns False"""

    model_settings.unsafe_probability = 0.0
    response = is_unsafe(model_settings)
    assert not response


def test_get_response_safe(model_settings: ModelSettings):
    """Check we get the safe response with is_unsafe returns False"""
    with patch("nemoguardrails.benchmark.mock_llm_server.response_data.is_unsafe") as mock_is_unsafe:
        mock_is_unsafe.return_value = False
        response = get_response(model_settings)
        assert response == model_settings.safe_text


def test_get_response_unsafe(model_settings: ModelSettings):
    """Check we get the safe response with is_unsafe returns False"""
    with patch("nemoguardrails.benchmark.mock_llm_server.response_data.is_unsafe") as mock_is_unsafe:
        mock_is_unsafe.return_value = True
        response = get_response(model_settings)
        assert response == model_settings.unsafe_text


@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.seed")
@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.normal")
@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.clip")
def test_get_latency_seconds_mocks_no_seed(mock_clip, mock_normal, mock_seed, model_settings: ModelSettings):
    """Check we call the correct numpy functions (not including seed)"""

    mock_normal.return_value = model_settings.latency_mean_seconds
    mock_clip.return_value = model_settings.latency_max_seconds

    result = get_latency_seconds(model_settings)

    assert result == mock_clip.return_value
    assert mock_seed.call_count == 0
    mock_normal.assert_called_once_with(
        loc=model_settings.latency_mean_seconds,
        scale=model_settings.latency_std_seconds,
        size=1,
    )
    mock_clip.assert_called_once_with(
        mock_normal.return_value,
        a_min=model_settings.latency_min_seconds,
        a_max=model_settings.latency_max_seconds,
    )


@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.seed")
@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.random.normal")
@patch("nemoguardrails.benchmark.mock_llm_server.response_data.np.clip")
def test_get_latency_seconds_mocks_with_seed(
    mock_clip, mock_normal, mock_seed, model_settings: ModelSettings, random_seed: int
):
    """Check we call the correct numpy functions (not including seed)"""

    mock_normal.return_value = model_settings.latency_mean_seconds
    mock_clip.return_value = model_settings.latency_max_seconds

    result = get_latency_seconds(model_settings, seed=random_seed)

    assert result == mock_clip.return_value
    mock_seed.assert_called_once_with(random_seed)
    mock_normal.assert_called_once_with(
        loc=model_settings.latency_mean_seconds,
        scale=model_settings.latency_std_seconds,
        size=1,
    )
    mock_clip.assert_called_once_with(
        mock_normal.return_value,
        a_min=model_settings.latency_min_seconds,
        a_max=model_settings.latency_max_seconds,
    )
