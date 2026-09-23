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

from __future__ import annotations

import pytest

from nemoguardrails import LLMRails
from tests.recorded.assertions import assert_stream_contract
from tests.recorded.rails.public_api.configs import INPUT_RAILS_CONFIG
from tests.recorded.rails_config import load_config

pytestmark = [pytest.mark.recorded]


def test_load_config_returns_fresh_copy_when_llmrails_mutates_config():
    config = load_config(INPUT_RAILS_CONFIG)
    original_flow_count = len(config.flows)

    LLMRails(config, verbose=False)

    assert len(config.flows) > original_flow_count
    assert len(load_config(INPUT_RAILS_CONFIG).flows) == original_flow_count


def test_assert_stream_contract_accepts_metadata_text_chunks():
    assert assert_stream_contract([{"text": "Hello", "metadata": {"usage": {}}}], expect_multiple=False) == "Hello"
