# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.rails.llm.buffer import RollingBuffer as BufferStrategy


async def fake_streaming_handler():
    # Fake streaming handler that yields chunks
    for i in range(15):
        yield f"chunk{i}"


@pytest.mark.asyncio
async def test_buffer_strategy():
    buffer_strategy = BufferStrategy(buffer_context_size=5, buffer_chunk_size=10)
    streaming_handler = fake_streaming_handler()

    expected_buffers = [
        [
            "chunk0",
            "chunk1",
            "chunk2",
            "chunk3",
            "chunk4",
            "chunk5",
            "chunk6",
            "chunk7",
            "chunk8",
            "chunk9",
        ],
        [
            "chunk5",
            "chunk6",
            "chunk7",
            "chunk8",
            "chunk9",
            "chunk10",
            "chunk11",
            "chunk12",
            "chunk13",
            "chunk14",
        ],
        ["chunk10", "chunk11", "chunk12", "chunk13", "chunk14"],
    ]

    async for idx, (buffer, _) in async_enumerate(buffer_strategy(streaming_handler)):
        assert buffer == expected_buffers[idx]


async def async_enumerate(aiterable, start=0):
    idx = start
    async for item in aiterable:
        yield idx, item
        idx += 1


def test_generate_chunk_str():
    buffer_strategy = BufferStrategy(buffer_context_size=5, buffer_chunk_size=10)
    buffer = ["chunk0", "chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    current_index = 6

    # we've already processed chunks 0 to 4 by setting last_index to 5
    buffer_strategy.last_index = 5

    result = buffer_strategy.generate_chunk_str(buffer, current_index)
    assert result == "chunk5"
