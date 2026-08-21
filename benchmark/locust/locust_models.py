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

import re
from itertools import product
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# Keys that configure the batch itself rather than an individual Locust run.
BATCH_KEYS = ("batch_name", "output_base_dir", "sweeps")

SweepValue = Union[int, float, str]

# Anything outside this set is replaced when a swept value becomes a directory name.
UNSAFE_LABEL_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def label_segment(value: SweepValue) -> str:
    """Render a swept value as one safe path segment.

    Sweep values reach the filesystem as directory names, and legitimate values
    contain separators: ``meta/llama-3.3-70b-instruct`` would otherwise nest
    directories, and a value containing ``..`` would climb out of the batch
    directory entirely.
    """
    segment = UNSAFE_LABEL_CHARS.sub("-", str(value))

    # "", "." and ".." are not usable directory names.
    if set(segment) <= {"."}:
        return "value"

    return segment


class LocustSweepConfig(BaseModel):
    """A batch of Locust runs sharing one base configuration.

    Accepts the nested form, which mirrors ``benchmark/aiperf``::

        batch_name: sweep_concurrency
        output_base_dir: locust_results
        base_config:
          host: "http://localhost:9000"
          config_id: content_safety_local
          model: meta/llama-3.3-70b-instruct
        sweeps:
          users: [1, 2, 4]

    It also accepts a flat single-run config, which is the format that
    predates sweeps, so existing configuration files keep working::

        host: "http://localhost:9000"
        config_id: content_safety_local
        model: meta/llama-3.3-70b-instruct

    A flat config may carry a ``sweeps`` block too; the remaining keys are
    read as the base configuration.
    """

    model_config = ConfigDict(extra="forbid")

    batch_name: str = Field(
        default="benchmark",
        description="Name for this batch of runs, used as a directory under output_base_dir",
    )
    output_base_dir: Optional[str] = Field(
        default=None,
        description="Base directory for results. Defaults to the base config's output_base_dir",
    )
    base_config: LocustConfig = Field(
        ...,
        description="Configuration applied to every run, before sweep values override it",
    )
    sweeps: Optional[Dict[str, List[SweepValue]]] = Field(
        default=None,
        description="Parameters to sweep. Key is a LocustConfig field, value is the list of values to run",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_flat_config(cls, data):
        """Read a flat single-run config as a batch with one base configuration."""
        if not isinstance(data, dict) or "base_config" in data:
            return data

        batch = {key: value for key, value in data.items() if key in BATCH_KEYS}
        # output_base_dir stays in the base config as well, so LocustConfig keeps its own default.
        base = {key: value for key, value in data.items() if key not in ("batch_name", "sweeps")}
        return {**batch, "base_config": base}

    @model_validator(mode="after")
    def validate_sweeps(self):
        """Default the output directory and reject sweeps that cannot be applied."""
        if self.output_base_dir is None:
            self.output_base_dir = self.base_config.output_base_dir

        if not self.sweeps:
            return self

        unknown = sorted(set(self.sweeps) - set(LocustConfig.model_fields))
        if unknown:
            raise ValueError(f"Sweep parameters are not LocustConfig fields: {unknown}")

        empty = sorted(key for key, values in self.sweeps.items() if not values)
        if empty:
            raise ValueError(f"Sweep parameters have no values: {empty}")

        return self

    def expand(self) -> List[tuple[str, LocustConfig]]:
        """Return a ``(label, config)`` pair for every run in this batch.

        Sweeping several parameters runs their Cartesian product, matching the
        AIPerf runner. Without a sweep this is the base configuration alone,
        labelled with the empty string.
        """
        if not self.sweeps:
            return [("", self.base_config)]

        # Sorted so the run order and directory names do not depend on YAML key order.
        keys = sorted(self.sweeps)

        runs = []
        used: set[str] = set()
        for combination in product(*(self.sweeps[key] for key in keys)):
            overrides = dict(zip(keys, combination, strict=True))
            # Rebuild rather than model_copy so field constraints still apply to swept values.
            config = LocustConfig(**{**self.base_config.model_dump(), **overrides})
            label = self._unique_label(overrides, used)
            used.add(label)
            runs.append((label, config))

        return runs

    @staticmethod
    def _unique_label(overrides: dict, used: set) -> str:
        """Build a label for one combination that no earlier combination took.

        Sanitising can map distinct values onto the same segment: ``a/b`` and
        ``a-b`` both become ``a-b``. Without a suffix the two levels would share
        a directory and the later one would overwrite the earlier one's results.
        Suffixes are assigned in expansion order, which is deterministic, so
        --resume still matches a level to its directory.
        """
        label = "_".join(f"{key}-{label_segment(value)}" for key, value in overrides.items())
        if label not in used:
            return label

        suffix = 2
        while f"{label}-{suffix}" in used:
            suffix += 1
        return f"{label}-{suffix}"
