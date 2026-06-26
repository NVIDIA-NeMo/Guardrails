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
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.vcr, pytest.mark.asyncio]


async def test_openai_chat_generate_text(openai_api_key):
    async with httpx.AsyncClient() as http_client:
        client = OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=openai_api_key,
            http_client=http_client,
            max_retries=0,
        )
        model = OpenAIChatModel(client=client, model="gpt-4o-mini")

        result = await model.generate_async("Say hello in one word")

    assert result.usage is not None
    assert {
        "content": result.content,
        "finish_reason": result.finish_reason,
        "model": result.model,
        "request_id": result.request_id,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
            "cached_tokens": result.usage.cached_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
        },
    } == snapshot(
        {
            "content": "Hello!",
            "finish_reason": "stop",
            "model": "gpt-4o-mini-2024-07-18",
            "request_id": "[RECORDED_RESPONSE_ID]",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 2,
                "total_tokens": 14,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            },
        }
    )
