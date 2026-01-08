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


import uuid
from typing import Optional

import numpy as np

from benchmark.mock_llm_server.config import ModelSettings


def generate_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique ID for completions."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def calculate_tokens(text: str) -> int:
    """Rough token calculation (approximately 4 characters per token)."""
    return max(1, len(text) // 4)


def get_response(config: ModelSettings, seed: Optional[int] = None) -> str:
    """Get a dummy /completion or /chat/completion response."""

    if is_unsafe(config, seed):
        return config.unsafe_text
    return config.safe_text


def get_latency_seconds(config: ModelSettings, seed: Optional[int] = None) -> float:
    """Sample end-to-end latency for this request using the model's config.
    Very inefficient to generate each sample singly rather than in batch.
    """
    if seed:
        np.random.seed(seed)

    # Sample from the normal distribution using model config
    latency_seconds = np.random.normal(
        loc=config.e2e_latency_mean_seconds,
        scale=config.e2e_latency_std_seconds,
        size=1,
    )

    # Truncate distribution's support using min and max config values
    latency_seconds = np.clip(
        latency_seconds,
        a_min=config.e2e_latency_min_seconds,
        a_max=config.e2e_latency_max_seconds,
    )
    return float(latency_seconds[0])


def is_unsafe(config: ModelSettings, seed: Optional[int] = None) -> bool:
    """Check if the model should return a refusal
    Very inefficient to generate each sample singly rather than in batch
    """
    if seed:
        np.random.seed(seed)

    refusal = np.random.binomial(n=1, p=config.unsafe_probability, size=1)
    return bool(refusal[0])


def split_response_into_chunks(response_text: str) -> list[str]:
    """Split response text by whitespace into chunks for streaming.

    Each word (and any attached punctuation) becomes a separate chunk.
    Whitespace is preserved by appending a space after each chunk except the last.

    Args:
        response_text: The full response text to split

    Returns:
        List of text chunks to stream back
    """
    words = response_text.split()
    if not words:
        return []

    # Add space after each word except the last to preserve original spacing
    chunks = [word + " " for word in words[:-1]]
    chunks.append(words[-1])  # Last word without trailing space
    return chunks


def generate_chunk_latencies(
    config: ModelSettings,
    num_chunks: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate latencies for each streaming chunk.

    Uses TTFT (Time to First Token) for the first chunk and ITL (Inter-Token Latency)
    for subsequent chunks. Both are sampled from truncated normal distributions.

    Args:
        config: Model settings containing TTFT and ITL parameters
        num_chunks: Number of chunks to generate latencies for
        seed: Optional random seed for reproducibility

    Returns:
        Numpy array of latencies in seconds, one for each chunk
    """
    if num_chunks <= 0:
        return np.array([])

    if seed:
        np.random.seed(seed)

    latencies = np.zeros(num_chunks)

    # First chunk uses TTFT
    ttft = np.random.normal(loc=config.ttft_mean_seconds, scale=config.ttft_std_seconds, size=1)
    ttft = np.clip(ttft, a_min=config.ttft_min_seconds, a_max=config.ttft_max_seconds)
    latencies[0] = ttft[0]

    # Remaining chunks use Inter Token Latencies
    if num_chunks > 1:
        inter_token_latencies = np.random.normal(
            loc=config.chunk_latency_mean_seconds,
            scale=config.chunk_latency_std_seconds,
            size=num_chunks - 1,
        )
        inter_token_latencies = np.clip(
            inter_token_latencies,
            a_min=config.chunk_latency_min_seconds,
            a_max=config.chunk_latency_max_seconds,
        )
        latencies[1:] = inter_token_latencies

    return latencies
