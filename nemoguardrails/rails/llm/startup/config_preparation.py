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

"""LLMRails config preparation."""

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.startup.colang_flows import (
    load_default_colang_1_flows,
    load_guardrails_library_flows_and_bot_messages,
    mark_rail_flows_as_system_subflows,
)

__all__ = ["prepare_llmrails_config"]


def prepare_llmrails_config(
    *,
    config: RailsConfig,
    in_place: bool = True,
) -> RailsConfig:
    """Prepare a RailsConfig for the standard LLMRails runtime."""
    prepared_config = config if in_place else config.model_copy(deep=True)

    load_default_colang_1_flows(prepared_config)
    load_guardrails_library_flows_and_bot_messages(prepared_config)

    mark_rail_flows_as_system_subflows(prepared_config)

    return prepared_config
