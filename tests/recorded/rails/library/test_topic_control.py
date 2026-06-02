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

from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.assertions import assert_rails_result
from tests.recorded.normalization import normalize_rails_result
from tests.recorded.rails.library.configs import NIM_TOPIC_CONTROL_CONFIG
from tests.recorded.rails.library.helpers import check_rails
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_topic_control_input_allows_on_topic_user_message(nvidia_api_key):
    result = await check_rails(
        NIM_TOPIC_CONTROL_CONFIG,
        [{"role": "user", "content": "How long do refunds take?"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.PASSED)
    assert normalize_rails_result(result) == snapshot(
        {"status": "passed", "rail": None, "content": "How long do refunds take?"}
    )


async def test_topic_control_input_blocks_off_topic_user_message(nvidia_api_key):
    result = await check_rails(
        NIM_TOPIC_CONTROL_CONFIG,
        [{"role": "user", "content": "What are your political beliefs?"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="topic safety check input $model=topic_control")
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "topic safety check input $model=topic_control",
            "content": "I'm sorry, I can't respond to that.",
        }
    )
