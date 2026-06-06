# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration management for domain hallucination guard (Layer 1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Layer1Config:
    """Layer 1 (fast LLM judgment) settings."""

    enabled: bool = True
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class Layer2Config:
    """Layer 2 (advanced network verification) settings."""

    enabled: bool = False
    temperature: float = 0.1
    max_tokens: int = 1024
    dns_timeout: float = 5.0


@dataclass
class DomainHallucinationGuardConfig:
    """Domain hallucination guard configuration."""

    layer1: Optional[Layer1Config] = None
    layer2: Optional[Layer2Config] = None
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        if self.layer1 is None:
            self.layer1 = Layer1Config()
        if self.layer2 is None:
            self.layer2 = Layer2Config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer1": asdict(self.layer1),
            "layer2": asdict(self.layer2),
            "debug": self.debug,
            "log_level": self.log_level,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def load_config(config_path: str) -> DomainHallucinationGuardConfig:
    """Load configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Loaded configuration object
    """
    with open(config_path) as f:
        data = json.load(f)

    layer1_data = data.get("layer1", {})
    layer1 = Layer1Config(**layer1_data) if layer1_data else Layer1Config()

    layer2_data = data.get("layer2", {})
    layer2 = Layer2Config(**layer2_data) if layer2_data else Layer2Config()

    return DomainHallucinationGuardConfig(
        layer1=layer1,
        layer2=layer2,
        debug=data.get("debug", False),
        log_level=data.get("log_level", "INFO"),
    )


# Default config instance
_DEFAULT_CONFIG = DomainHallucinationGuardConfig()


def get_config() -> DomainHallucinationGuardConfig:
    """Get default configuration."""
    return _DEFAULT_CONFIG


def set_config(config: DomainHallucinationGuardConfig) -> None:
    """Set default configuration."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = config
