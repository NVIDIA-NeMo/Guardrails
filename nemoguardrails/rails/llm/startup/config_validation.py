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

"""LLMRails startup config validation."""

from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.startup.colang_flows import validate_rail_flow_names

__all__ = ["validate_llmrails_config"]


def validate_llmrails_config(config: RailsConfig) -> None:
    validate_rail_flow_names(config)

    if config.passthrough and config.rails.dialog.single_call.enabled:
        raise InvalidRailsConfigurationError(
            "The passthrough mode and the single call dialog rails mode can't be used at the same time. "
            "The single call mode needs to use an altered prompt when prompting the LLM. "
        )
