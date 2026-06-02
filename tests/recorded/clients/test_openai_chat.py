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

import httpx
import pytest

from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient
from nemoguardrails.llm.models.openai_chat import OpenAIChatModel
from tests.recorded.utils import DUMMY_OPENAI_API_KEY, api_key_for_record_mode

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_openai_chat_generate_text(record_mode):
    api_key = api_key_for_record_mode("OPENAI_API_KEY", DUMMY_OPENAI_API_KEY, record_mode)

    async with httpx.AsyncClient() as http_client:
        client = OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        model = OpenAIChatModel(client=client, model="gpt-4o-mini")

        result = await model.generate_async("Say hello in one word")

    assert isinstance(result.content, str)
    assert result.content
    assert result.finish_reason in {"stop", "length", "tool_calls", "content_filter", "other"}
    assert result.request_id
    assert result.usage is not None
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.usage.total_tokens >= result.usage.input_tokens + result.usage.output_tokens
