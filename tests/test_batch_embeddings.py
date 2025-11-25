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
from time import time

import pytest

from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
from nemoguardrails.embeddings.index import IndexItem


@pytest.mark.skip(reason="Run manually.")
@pytest.mark.asyncio
async def test_search_speed():
    embeddings_index = BasicEmbeddingsIndex(embedding_model="all-MiniLM-L6-v2", embedding_engine="SentenceTransformers")

    # We compute an initial embedding, to warm up the model.
    await embeddings_index._get_embeddings(["warm up"])

    items = []
    for i in range(100):
        items.append(IndexItem(text=str(i), meta={"i": i}))

    t0 = time()
    await embeddings_index.add_items(items)
    took = time() - t0

    # Should take less than 2 seconds
    assert took < 2

    await embeddings_index.build()

    # Now, do a 100 individual requests

    # Statistics
    total_time = 0
    completed_requests = 0
    req_counter = 0
    concurrency = 300
    requests = 300

    async def _search(text):
        nonlocal total_time, completed_requests, req_counter

        async with semaphore:
            req_counter += 1
            # req_id = req_counter
            # delay = random.random()
            # print(f"Starting reqeust {req_id} with {delay:.2f}s delay.")
            # await asyncio.sleep(delay)

            start_time = time()

            await embeddings_index.search(text)

            delay = time() - start_time
            total_time += delay
            completed_requests += 1

    tasks = []
    t0 = time()
    semaphore = asyncio.Semaphore(concurrency)
    for i in range(requests):
        task = asyncio.ensure_future(_search(f"This is a long sentence meant to mimic a user request {i}." * 5))
        tasks.append(task)

    await asyncio.gather(*tasks)
    took = time() - t0

    print(f"Processing {completed_requests} took {took:0.2f}.")

    print(f"Completed {completed_requests} requests in {total_time:.2f} seconds.")
    print(f"Average latency: {total_time / completed_requests if completed_requests else 0:.2f} seconds.")
    print(f"Maximum concurrency: {concurrency}")
