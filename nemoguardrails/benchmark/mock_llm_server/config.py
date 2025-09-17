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

import os
from functools import lru_cache
from typing import Any, Optional, Union

import yaml
from openai._utils import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppModelConfig(BaseModel):
    """Pydantic model to configure the Mock LLM Server."""

    # Mandatory fields
    model: str = Field(..., description="Model name served by mock server")
    refusal_text: str = Field(..., description="Refusal response text")

    # Config with default values
    refusal_probability: float = Field(
        default=0.1, description="Probability of refusal (between 0 and 1)"
    )
    # Latency sampled from a truncated-normal distribution.
    # Plain Normal distributions have infinite support, and can be negative
    latency_min_seconds: float = Field(
        default=0.1, description="Minimum latency in seconds"
    )
    latency_max_seconds: float = Field(
        default=5, description="Maximum latency in seconds"
    )
    latency_mean_seconds: float = Field(
        default=0.5, description="The average response time in seconds"
    )
    latency_std_seconds: float = Field(
        default=0.1, description="Standard deviation of response time"
    )


settings: Optional[AppModelConfig] = None


def load_config(yaml_file: str) -> None:
    """Load the Model configuration from YAML file, store in global `settings` var"""
    global settings
    with open(yaml_file, "r") as f:
        config_data = yaml.safe_load(f)
    settings = AppModelConfig(**config_data)


@lru_cache
def get_config() -> AppModelConfig:
    """FastAPI Dependency to inject model configuration"""
    print(f"get_config called, settings = {settings}")
    print(f"GET_CONFIG CALLED IN PROCESS ID: {os.getpid()}")

    if settings is None:
        raise RuntimeError("No configuration loaded")
    return settings
