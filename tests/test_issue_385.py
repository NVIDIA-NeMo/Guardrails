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

import pytest

from nemoguardrails import LLMRails, RailsConfig


@pytest.mark.asyncio
async def test_1():
    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello World!"
        """,
        config={
            "models": [],
            "rails": {"dialog": {"user_messages": {"embeddings_only": True}}},
        },
    )
    rails = LLMRails(config)

    # This should not raise an exception
    await rails.generate_async(
        messages=[
            {"role": "user", "content": "hi!"},
            {"role": "assistant", "content": "hi!"},
            {"role": "user", "content": "hi!"},
        ]
    )
