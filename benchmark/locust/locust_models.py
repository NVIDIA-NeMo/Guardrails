#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Pydantic models for Locust load test configuration validation.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LocustConfig(BaseModel):
    """Configuration for a Locust load-test run"""

    model_config = ConfigDict(extra="forbid")

    # Server details
    host: str = Field(
        default="http://localhost:8000",
        description="Base URL of the NeMo Guardrails server to test",
    )
    config_id: str = Field(..., description="Guardrails configuration ID to use")
    model: str = Field(..., description="Model name to use in requests")

    # Load test parameters
    users: int = Field(
        default=256,
        ge=1,
        description="Maximum number of concurrent users",
    )
    spawn_rate: float = Field(
        default=10,
        ge=0.1,
        description="Rate at which users are spawned (users/second)",
    )
    run_time: int = Field(
        default=60,
        ge=1,
        description="Test duration in seconds",
    )

    # Request configuration
    message: str = Field(
        default="Hello, what can you do?",
        description="Message content to send in chat completion requests",
    )

    # Output configuration
    headless: bool = Field(
        default=True,
        description="Run in headless mode without web UI",
    )

    output_base_dir: str = Field(
        default="locust_results",
        description="Base directory for load test results",
    )

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Ensure host starts with http:// or https://"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Host must start with http:// or https://")
        # Remove trailing slash if present
        return v.rstrip("/")
