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

"""Colang runtime selection."""

from typing import Dict, Type

from nemoguardrails.colang.v1_0.runtime.runtime import Runtime, RuntimeV1_0
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm.config import RailsConfig

__all__ = ["runtime_for_colang_version"]


def runtime_for_colang_version(config: RailsConfig, verbose: bool) -> Runtime:
    """Create the Colang runtime selected by the config version."""
    colang_version_to_runtime: Dict[str, Type[Runtime]] = {
        "1.0": RuntimeV1_0,
        "2.x": RuntimeV2_x,
    }
    if config.colang_version not in colang_version_to_runtime:
        raise InvalidRailsConfigurationError(
            f"Unsupported colang version: {config.colang_version}. "
            f"Supported versions: {list(colang_version_to_runtime.keys())}"
        )

    return colang_version_to_runtime[config.colang_version](config=config, verbose=verbose)
