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

import asyncio

from transformers.generation.streamers import TextStreamer


class AsyncTextIteratorStreamer(TextStreamer):
    """
    Simple async implementation for HuggingFace Transformers streamers.

    This follows closely how transformers.generation.streamers.TextIteratorStreamer works,
    with minor modifications to make it async.
    """

    def __init__(
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.loop = None

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if len(text) > 0:
            asyncio.run_coroutine_threadsafe(self.text_queue.put(text), self.loop)

        if stream_end:
            asyncio.run_coroutine_threadsafe(self.text_queue.put(text), self.loop)

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.text_queue.get()
        if value == self.stop_signal:
            raise StopAsyncIteration()
        else:
            return value
