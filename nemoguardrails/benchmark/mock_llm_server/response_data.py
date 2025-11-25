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


import uuid
from typing import Optional

import numpy as np

from nemoguardrails.benchmark.mock_llm_server.config import ModelSettings


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
    """Sample latency for this request using the model's config
    Very inefficient to generate each sample singly rather than in batch
    """
    if seed:
        np.random.seed(seed)

    # Sample from the normal distribution using model config
    latency_seconds = np.random.normal(loc=config.latency_mean_seconds, scale=config.latency_std_seconds, size=1)

    # Truncate distribution's support using min and max config values
    latency_seconds = np.clip(
        latency_seconds,
        a_min=config.latency_min_seconds,
        a_max=config.latency_max_seconds,
    )
    return float(latency_seconds)


def is_unsafe(config: ModelSettings, seed: Optional[int] = None) -> bool:
    """Check if the model should return a refusal
    Very inefficient to generate each sample singly rather than in batch
    """
    if seed:
        np.random.seed(seed)

    refusal = np.random.binomial(n=1, p=config.unsafe_probability, size=1)
    return bool(refusal[0])
