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

from typing import Any

from nemoguardrails.rails.llm.options import RailType
from tests.recorded.rails.helpers import async_chunks, build_rails
from tests.recorded.rails_config import RailsConfigSource
from tests.utils import FakeLLMModel


async def check_rails(
    config: RailsConfigSource,
    messages: list[dict[str, Any]],
    *,
    rail_types: tuple[RailType, ...] | None = None,
):
    rails = build_rails(config)
    return await rails.check_async(messages, rail_types=list(rail_types) if rail_types is not None else None)


async def generate_with_fake_main(
    config: RailsConfigSource,
    main_output: str,
    messages: list[dict[str, Any]],
):
    rails = build_rails(config, llm=FakeLLMModel(responses=[main_output]))
    return await rails.generate_async(
        messages=messages,
        options={"log": {"activated_rails": True, "llm_calls": True}},
    )


async def stream_with_fake_main(
    config: RailsConfigSource,
    main_output: str,
    messages: list[dict[str, Any]],
):
    rails = build_rails(config, streaming=True)
    chunks = []
    async for chunk in rails.stream_async(
        messages=messages,
        generator=async_chunks([main_output]),
        options={"rails": ["output"]},
    ):
        chunks.append(chunk)
    return chunks
