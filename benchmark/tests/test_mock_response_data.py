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

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmark.mock_llm_server.config import ModelSettings
from benchmark.mock_llm_server.response_data import (
    calculate_tokens,
    generate_chunk_latencies,
    generate_id,
    get_latency_seconds,
    get_response,
    is_unsafe,
    split_response_into_chunks,
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
        e2e_latency_min_seconds=0.2,
        e2e_latency_max_seconds=1.0,
        e2e_latency_mean_seconds=0.5,
        e2e_latency_std_seconds=0.1,
    )
    return settings


@pytest.fixture
def random_seed() -> int:
    """Return a fixed seed number for all tests"""
    return 12345


@patch("benchmark.mock_llm_server.response_data.np.random.seed")
@patch("benchmark.mock_llm_server.response_data.np.random.binomial")
def test_is_unsafe_mocks_no_seed(mock_binomial: MagicMock, mock_seed: MagicMock, model_settings: ModelSettings):
    """Check `is_unsafe()` calls the correct numpy functions"""
    mock_binomial.return_value = [True]

    response = is_unsafe(model_settings)

    assert response
    assert mock_seed.call_count == 0
    assert mock_binomial.call_count == 1
    mock_binomial.assert_called_once_with(n=1, p=model_settings.unsafe_probability, size=1)


@patch("benchmark.mock_llm_server.response_data.np.random.seed")
@patch("benchmark.mock_llm_server.response_data.np.random.binomial")
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
    with patch("benchmark.mock_llm_server.response_data.is_unsafe") as mock_is_unsafe:
        mock_is_unsafe.return_value = False
        response = get_response(model_settings)
        assert response == model_settings.safe_text


def test_get_response_unsafe(model_settings: ModelSettings):
    """Check we get the safe response with is_unsafe returns False"""
    with patch("benchmark.mock_llm_server.response_data.is_unsafe") as mock_is_unsafe:
        mock_is_unsafe.return_value = True
        response = get_response(model_settings)
        assert response == model_settings.unsafe_text


@patch("benchmark.mock_llm_server.response_data.np.random.seed")
@patch("benchmark.mock_llm_server.response_data.np.random.normal")
@patch("benchmark.mock_llm_server.response_data.np.clip")
def test_get_latency_seconds_mocks_no_seed(mock_clip, mock_normal, mock_seed, model_settings: ModelSettings):
    """Check we call the correct numpy functions (not including seed)"""

    mock_normal.return_value = np.array([model_settings.e2e_latency_mean_seconds])
    mock_clip.return_value = np.array([model_settings.e2e_latency_max_seconds])

    result = get_latency_seconds(model_settings)

    assert result == mock_clip.return_value
    assert mock_seed.call_count == 0
    mock_normal.assert_called_once_with(
        loc=model_settings.e2e_latency_mean_seconds,
        scale=model_settings.e2e_latency_std_seconds,
        size=1,
    )
    mock_clip.assert_called_once_with(
        mock_normal.return_value,
        a_min=model_settings.e2e_latency_min_seconds,
        a_max=model_settings.e2e_latency_max_seconds,
    )


@patch("benchmark.mock_llm_server.response_data.np.random.seed")
@patch("benchmark.mock_llm_server.response_data.np.random.normal")
@patch("benchmark.mock_llm_server.response_data.np.clip")
def test_get_latency_seconds_mocks_with_seed(
    mock_clip, mock_normal, mock_seed, model_settings: ModelSettings, random_seed: int
):
    """Check we call the correct numpy functions (not including seed)"""

    mock_normal.return_value = np.array([model_settings.e2e_latency_mean_seconds])
    mock_clip.return_value = np.array([model_settings.e2e_latency_max_seconds])

    result = get_latency_seconds(model_settings, seed=random_seed)

    assert result == mock_clip.return_value
    mock_seed.assert_called_once_with(random_seed)
    mock_normal.assert_called_once_with(
        loc=model_settings.e2e_latency_mean_seconds,
        scale=model_settings.e2e_latency_std_seconds,
        size=1,
    )
    mock_clip.assert_called_once_with(
        mock_normal.return_value,
        a_min=model_settings.e2e_latency_min_seconds,
        a_max=model_settings.e2e_latency_max_seconds,
    )


class TestSplitResponseIntoChunks:
    """Test the split_response_into_chunks function."""

    def test_split_simple_sentence(self):
        """Test splitting a simple sentence into word chunks."""
        text = "Hello world"
        chunks = split_response_into_chunks(text)
        assert chunks == ["Hello ", "world"]

    def test_split_multiple_words(self):
        """Test splitting multiple words preserves spacing."""
        text = "I can help you"
        chunks = split_response_into_chunks(text)
        assert chunks == ["I ", "can ", "help ", "you"]

    def test_split_empty_string(self):
        """Test splitting an empty string returns empty list."""
        text = ""
        chunks = split_response_into_chunks(text)
        assert chunks == []

    def test_split_single_word(self):
        """Test splitting a single word returns it without trailing space."""
        text = "Hello"
        chunks = split_response_into_chunks(text)
        assert chunks == ["Hello"]

    def test_split_preserves_punctuation(self):
        """Test that punctuation stays attached to words."""
        text = "Hello, world!"
        chunks = split_response_into_chunks(text)
        assert chunks == ["Hello, ", "world!"]

    def test_split_reconstructs_original(self):
        """Test that joining chunks reconstructs the original text."""
        text = "I can provide information and help with a wide range of topics."
        chunks = split_response_into_chunks(text)
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_split_whitespace_only(self):
        """Test splitting whitespace-only string returns empty list."""
        text = "   "
        chunks = split_response_into_chunks(text)
        assert chunks == []


class TestGenerateChunkLatencies:
    """Test the generate_chunk_latencies function."""

    @pytest.fixture
    def streaming_settings(self) -> ModelSettings:
        """Generate config data with streaming latency settings.
        Each value should be unique to make sure we pass the correct configs to numpy functions
        """
        return ModelSettings(
            model="test-model",
            unsafe_text="Unsafe",
            safe_text="Safe",
            ttft_min_seconds=0.1,
            ttft_max_seconds=4.0,
            ttft_mean_seconds=1.3,
            ttft_std_seconds=0.96,
            chunk_latency_min_seconds=0.01,
            chunk_latency_max_seconds=0.12,
            chunk_latency_mean_seconds=0.05,
            chunk_latency_std_seconds=0.02,
        )

    def test_generate_latencies_zero_chunks(self, streaming_settings: ModelSettings):
        """Test generating latencies for zero chunks returns empty array."""
        latencies = generate_chunk_latencies(streaming_settings, 0)
        assert len(latencies) == 0
        assert isinstance(latencies, np.ndarray)

    def test_generate_latencies_negative_chunks(self, streaming_settings: ModelSettings):
        """Test generating latencies for negative chunks returns empty array."""
        latencies = generate_chunk_latencies(streaming_settings, -1)
        assert len(latencies) == 0

    @patch("benchmark.mock_llm_server.response_data.np.random.seed")
    @patch("benchmark.mock_llm_server.response_data.np.random.normal")
    @patch("benchmark.mock_llm_server.response_data.np.clip")
    def test_generate_latencies_single_chunk_no_seed(
        self,
        mock_clip: MagicMock,
        mock_normal: MagicMock,
        mock_seed: MagicMock,
        streaming_settings: ModelSettings,
    ):
        """Test single chunk calls TTFT normal and clip, no seed call."""
        mock_normal.return_value = np.array([streaming_settings.ttft_mean_seconds])
        mock_clip.return_value = np.array([streaming_settings.ttft_mean_seconds])

        latencies = generate_chunk_latencies(streaming_settings, 1)

        assert len(latencies) == 1
        assert mock_seed.call_count == 0
        mock_normal.assert_called_once_with(
            loc=streaming_settings.ttft_mean_seconds,
            scale=streaming_settings.ttft_std_seconds,
            size=1,
        )
        mock_clip.assert_called_once_with(
            mock_normal.return_value,
            a_min=streaming_settings.ttft_min_seconds,
            a_max=streaming_settings.ttft_max_seconds,
        )

    @patch("benchmark.mock_llm_server.response_data.np.random.seed")
    @patch("benchmark.mock_llm_server.response_data.np.random.normal")
    @patch("benchmark.mock_llm_server.response_data.np.clip")
    def test_generate_latencies_single_chunk_with_seed(
        self,
        mock_clip: MagicMock,
        mock_normal: MagicMock,
        mock_seed: MagicMock,
        streaming_settings: ModelSettings,
    ):
        """Test single chunk with seed calls np.random.seed."""
        mock_normal.return_value = np.array([streaming_settings.ttft_mean_seconds])
        mock_clip.return_value = np.array([streaming_settings.ttft_min_seconds])
        seed_value = 42

        latencies = generate_chunk_latencies(streaming_settings, 1, seed=seed_value)

        assert len(latencies) == 1
        mock_seed.assert_called_once_with(seed_value)
        mock_normal.assert_called_once_with(
            loc=streaming_settings.ttft_mean_seconds,
            scale=streaming_settings.ttft_std_seconds,
            size=1,
        )
        mock_clip.assert_called_once_with(
            mock_normal.return_value,
            a_min=streaming_settings.ttft_min_seconds,
            a_max=streaming_settings.ttft_max_seconds,
        )

    @patch("benchmark.mock_llm_server.response_data.np.random.seed")
    @patch("benchmark.mock_llm_server.response_data.np.random.normal")
    @patch("benchmark.mock_llm_server.response_data.np.clip")
    def test_generate_latencies_multiple_chunks_no_seed(
        self,
        mock_clip: MagicMock,
        mock_normal: MagicMock,
        mock_seed: MagicMock,
        streaming_settings: ModelSettings,
    ):
        """Test multiple chunks calls TTFT then ITL normal and clip."""
        num_chunks = 5
        ttft_value = np.array([streaming_settings.ttft_mean_seconds])
        chunk_values = np.array([0.01, 0.02, 0.03, 0.04])

        # First call returns TTFT, second call returns ITL values
        mock_normal.side_effect = [ttft_value, chunk_values]
        mock_clip.side_effect = [ttft_value, chunk_values]

        latencies = generate_chunk_latencies(streaming_settings, num_chunks)

        assert len(latencies) == num_chunks
        assert mock_seed.call_count == 0
        assert mock_normal.call_count == 2
        assert mock_clip.call_count == 2

        # Check the TTFT Normal distribution
        ttft_normal_call_args, ttft_normal_call_kwargs = mock_normal.call_args_list[0]
        assert ttft_normal_call_args == ()  # All arguments are passed as kwargs, so args list is empty
        assert ttft_normal_call_kwargs["loc"] == streaming_settings.ttft_mean_seconds
        assert ttft_normal_call_kwargs["scale"] == streaming_settings.ttft_std_seconds
        assert ttft_normal_call_kwargs["size"] == 1

        # Check the ITL Normal distribution call (for all but the first chunk)
        chunk_normal_call_args, chunk_normal_call_kwargs = mock_normal.call_args_list[1]
        assert chunk_normal_call_args == ()  # All arguments are passed as kwargs, so args list is empty
        assert chunk_normal_call_kwargs["loc"] == streaming_settings.chunk_latency_mean_seconds
        assert chunk_normal_call_kwargs["scale"] == streaming_settings.chunk_latency_std_seconds
        assert chunk_normal_call_kwargs["size"] == num_chunks - 1

        # Check TTFT clip calls
        ttft_clip_call_args, ttft_clip_call_kwargs = mock_clip.call_args_list[0]
        assert ttft_clip_call_args[0] == ttft_value
        assert ttft_clip_call_kwargs["a_max"] == streaming_settings.ttft_max_seconds
        assert ttft_clip_call_kwargs["a_min"] == streaming_settings.ttft_min_seconds

        # Check ITL clip calls
        chunk_clip_call_args, chunk_clip_call_kwargs = mock_clip.call_args_list[1]
        np.testing.assert_array_equal(chunk_clip_call_args[0], chunk_values)
        assert chunk_clip_call_kwargs["a_max"] == streaming_settings.chunk_latency_max_seconds
        assert chunk_clip_call_kwargs["a_min"] == streaming_settings.chunk_latency_min_seconds

    @patch("benchmark.mock_llm_server.response_data.np.random.seed")
    @patch("benchmark.mock_llm_server.response_data.np.random.normal")
    @patch("benchmark.mock_llm_server.response_data.np.clip")
    def test_generate_latencies_multiple_chunks_with_seed(
        self,
        mock_clip: MagicMock,
        mock_normal: MagicMock,
        mock_seed: MagicMock,
        streaming_settings: ModelSettings,
    ):
        """Test multiple chunks with seed calls np.random.seed once."""
        num_chunks = 3
        seed_value = 12345
        ttft_value = np.array([streaming_settings.ttft_mean_seconds])
        chunk_values = np.array([streaming_settings.chunk_latency_mean_seconds] * (num_chunks - 1))

        mock_normal.side_effect = [ttft_value, chunk_values]
        mock_clip.side_effect = [ttft_value, chunk_values]

        latencies = generate_chunk_latencies(streaming_settings, num_chunks, seed=seed_value)

        assert len(latencies) == num_chunks
        mock_seed.assert_called_once_with(seed_value)

        # Exact call arguments are tested in test_generate_latencies_multiple_chunks_no_seed, no need to retest
        assert mock_normal.call_count == 2
        assert mock_clip.call_count == 2

    @patch("benchmark.mock_llm_server.response_data.np.random.seed")
    @patch("benchmark.mock_llm_server.response_data.np.random.normal")
    @patch("benchmark.mock_llm_server.response_data.np.clip")
    def test_generate_latencies_returns_correct_values(
        self,
        mock_clip: MagicMock,
        mock_normal: MagicMock,
        streaming_settings: ModelSettings,
    ):
        """Test that returned latencies contain the clipped values."""
        num_chunks = 3
        ttft_clipped = np.array([0.25])
        chunk_clipped = np.array([0.04, 0.06])

        mock_normal.side_effect = [np.array([0.3]), np.array([0.05, 0.07])]
        mock_clip.side_effect = [ttft_clipped, chunk_clipped]

        latencies = generate_chunk_latencies(streaming_settings, num_chunks)

        assert len(latencies) == num_chunks
        assert latencies[0] == ttft_clipped[0]
        np.testing.assert_array_equal(latencies[1:], chunk_clipped)
