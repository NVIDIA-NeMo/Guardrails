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
from tests.recorded.rails.library.configs import JAILBREAK_PROMPT, NIM_JAILBREAK_CONFIG
from tests.recorded.rails.library.helpers import check_rails
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_jailbreak_detection_input_blocks_jailbreak_prompt(nvidia_api_key):
    result = await check_rails(
        NIM_JAILBREAK_CONFIG,
        [{"role": "user", "content": JAILBREAK_PROMPT}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="jailbreak detection model")
    assert normalize_rails_result(result) == snapshot(
        {"status": "blocked", "rail": "jailbreak detection model", "content": "I'm sorry, I can't respond to that."}
    )
