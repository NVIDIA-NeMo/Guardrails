# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Pydantic models for AIPerf configuration validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseConfig(BaseModel):
    """Base configuration for AIPerf benchmark runs."""

    # Model details
    model: str = Field(..., description="Model name")
    tokenizer: Optional[str] = Field(
        default=None,
        description="Optional tokenizer Huggingface name, or local directory",
    )
    url: str = Field(..., description="Model base URL")
    endpoint: str = Field(default="/v1/chat/completions", description="API endpoint path")
    endpoint_type: Literal["chat", "completions"] = Field(
        default="chat",
        description="Type of endpoint (chat or completions)",
    )
    api_key_env_var: Optional[str] = Field(default=None, description="API key environment variable")
    streaming: Optional[bool] = Field(default=False, description="Streaming mode")

    # Load generation settings
    warmup_request_count: int = Field(description="Requests to send before beginning performance-test")
    benchmark_duration: int = Field(description="Benchmark duration in seconds")
    concurrency: int = Field(description="Number of concurrent requests")
    request_rate: Optional[float] = Field(
        default=None,
        description="Request rate (requests per second, auto-calculated if not provided)",
    )
    request_rate_mode: Optional[Literal["constant", "poisson"]] = Field(
        default="constant",
        description="Request rate mode (constant, poisson, etc.)",
    )

    # Synthetic data generation
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    prompt_input_tokens_mean: Optional[int] = Field(
        default=None,
        description="Mean number of input tokens",
    )
    prompt_input_tokens_stddev: Optional[int] = Field(
        default=None,
        description="Standard deviation of input tokens",
    )
    prompt_output_tokens_mean: Optional[int] = Field(
        default=None,
        description="Mean number of output tokens",
    )
    prompt_output_tokens_stddev: Optional[int] = Field(
        default=None,
        description="Standard deviation of output tokens",
    )


class AIPerfConfig(BaseModel):
    """Main configuration model for AIPerf benchmark runner."""

    batch_name: str = Field(default="benchmark", description="Name for this batch of benchmarks")
    output_base_dir: str = Field(
        default="aiperf_results",
        description="Base directory for benchmark results",
    )
    base_config: BaseConfig = Field(..., description="Base configuration applied to all benchmark runs")
    sweeps: Optional[Dict[str, List[Union[int, str]]]] = Field(
        default=None,
        description="Parameter sweeps. Key is the parameter to change, value is a list of values to use",
    )

    @field_validator("sweeps")
    @classmethod
    def validate_sweeps(cls, v: Optional[Dict[str, List[Any]]]) -> Optional[Dict[str, List[Any]]]:
        """Validate that sweep values are lists of ints or strings."""
        if v is None:
            return v

        for param_name, values in v.items():
            if len(values) == 0:
                raise ValueError(f"Sweep parameter '{param_name}' cannot be empty")

        return v

    @model_validator(mode="after")
    def validate_sweep_keys(self):
        """Validate that sweep keys exist in base_config."""
        sweeps = self.sweeps
        if sweeps is None:
            return self

        # Get all valid field names from BaseConfig
        valid_keys = set(BaseConfig.model_fields.keys())

        # Check each sweep parameter
        for param_name in sweeps:
            if param_name not in valid_keys:
                valid_fields = sorted(valid_keys)
                raise ValueError(
                    f"Sweep parameter '{param_name}' is not a valid BaseConfig field. "
                    f"Valid fields are: {', '.join(valid_fields)}"
                )

        return self

    def get_output_base_path(self) -> Path:
        """Get the base output directory as a Path object."""
        return Path(self.output_base_dir)
