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

import os

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.providers import get_llm_provider_names
from nemoguardrails.llm.providers.providers import get_community_chat_provider_names

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_custom_llm_registration():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_custom_llm"))

    _ = LLMRails(config)
    supported_llms = get_llm_provider_names()

    assert "custom_llm" in supported_llms


def test_custom_chat_model_registration():
    config = RailsConfig.from_path(
        os.path.join(CONFIGS_FOLDER, "with_custom_chat_model")
    )
    _ = LLMRails(config)

    assert "custom_chat_model" in get_community_chat_provider_names()
