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
from nemoguardrails.rails.llm.options import RailStatus
from tests.recorded.assertions import assert_generated_message, assert_rails_result
from tests.recorded.normalization import normalize_rails_result
from tests.recorded.rails.public_api.configs import DIALOG_CONFIG, SINGLE_CALL_CONFIG
from tests.recorded.rails_config import load_config
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


@pytest.mark.vcr
async def test_dialog_generate_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(DIALOG_CONFIG), verbose=False)

    result = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
        }
    )


@pytest.mark.vcr
async def test_single_call_generate_async_public_contract(openai_api_key):
    rails = LLMRails(load_config(SINGLE_CALL_CONFIG), verbose=False)

    result = await rails.generate_async(messages=[{"role": "user", "content": "hello"}])

    assert_generated_message(result)
    assert result == snapshot(
        {
            "role": "assistant",
            "content": """\
bot express greeting
"Hello again! 😊 How can I assist you today?"\
""",
        }
    )


async def test_single_call_check_async_without_io_rails():
    rails = LLMRails(load_config(SINGLE_CALL_CONFIG), verbose=False)

    result = await rails.check_async([{"role": "user", "content": "hello"}])

    assert_rails_result(result, status=RailStatus.PASSED, content="hello")
    assert normalize_rails_result(result) == snapshot({"status": "passed", "rail": None, "content": "hello"})
