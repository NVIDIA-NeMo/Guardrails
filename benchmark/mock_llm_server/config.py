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

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILE_ENV_VAR = "MOCK_LLM_CONFIG_FILE"
config_file_path = os.getenv(CONFIG_FILE_ENV_VAR, "model_settings.yml")
CONFIG_FILE = Path(config_file_path)


class ModelSettings(BaseSettings):
    """Pydantic model to configure the Mock LLM Server."""

    # Mandatory fields
    model: str = Field(..., description="Model name served by mock server")
    unsafe_probability: float = Field(default=0.1, description="Probability of unsafe response (between 0 and 1)")
    unsafe_text: str = Field(..., description="Refusal response to unsafe prompt")
    safe_text: str = Field(..., description="Safe response")

    # Config with default values
    # Latency sampled from a truncated-normal distribution.
    # Plain Normal distributions have infinite support, and can be negative
    e2e_latency_min_seconds: float = Field(default=0.1, description="Minimum latency in seconds")
    e2e_latency_max_seconds: float = Field(default=5, description="Maximum latency in seconds")
    e2e_latency_mean_seconds: float = Field(default=0.5, description="The average response time in seconds")
    e2e_latency_std_seconds: float = Field(default=0.1, description="Standard deviation of response time")

    # Streaming latency: Time to First Token (TTFT)
    # https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html#time-to-first-token-ttft
    ttft_min_seconds: float = Field(default=0.05, description="Minimum TTFT in seconds")
    ttft_max_seconds: float = Field(default=0.5, description="Maximum TTFT in seconds")
    ttft_mean_seconds: float = Field(default=0.1, description="Average TTFT in seconds")
    ttft_std_seconds: float = Field(default=0.02, description="Standard deviation of TTFT")

    # Streaming latency: Chunk Latency
    # https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html#inter-token-latency-itl
    chunk_latency_min_seconds: float = Field(default=0.01, description="Minimum chunk latency in seconds")
    chunk_latency_max_seconds: float = Field(default=0.1, description="Maximum chunk latency in seconds")
    chunk_latency_mean_seconds: float = Field(default=0.03, description="Average chunk latency in seconds")
    chunk_latency_std_seconds: float = Field(default=0.01, description="Standard deviation of chunk latency")

    model_config = SettingsConfigDict(env_file=CONFIG_FILE)


@lru_cache()
def get_settings() -> ModelSettings:
    """Singleton-pattern to get settings once via lru_cache"""
    settings = ModelSettings()  # type: ignore (These are filled in by loading from CONFIG_FILE)
    return settings
