# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@pytest.mark.asyncio
async def test_batch_get_embeddings_raises_for_all_waiters_if_provider_fails(monkeypatch):
    embeddings_index = BasicEmbeddingsIndex(use_batching=True, max_batch_size=2, max_batch_hold=1.0)

    async def _fail(_texts):
        raise RuntimeError("provider down")

    monkeypatch.setattr(embeddings_index, "_get_embeddings", _fail)

    results = await asyncio.wait_for(
        asyncio.gather(
            embeddings_index._batch_get_embeddings("first"),
            embeddings_index._batch_get_embeddings("second"),
            return_exceptions=True,
        ),
        timeout=0.2,
    )

    assert len(results) == 2
    for result in results:
        assert isinstance(result, RuntimeError)
        assert str(result) == "Failed to compute embeddings for batched request."
        assert isinstance(result.__cause__, RuntimeError)
        assert str(result.__cause__) == "provider down"


@pytest.mark.asyncio
async def test_batch_get_embeddings_recovers_after_failed_batch(monkeypatch):
    embeddings_index = BasicEmbeddingsIndex(use_batching=True, max_batch_size=2, max_batch_hold=1.0)
    call_count = 0

    async def _maybe_fail(texts):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise RuntimeError("provider down")

        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(embeddings_index, "_get_embeddings", _maybe_fail)

    failed_results = await asyncio.wait_for(
        asyncio.gather(
            embeddings_index._batch_get_embeddings("first"),
            embeddings_index._batch_get_embeddings("second"),
            return_exceptions=True,
        ),
        timeout=0.2,
    )

    assert all(isinstance(result, RuntimeError) for result in failed_results)

    recovered_results = await asyncio.wait_for(
        asyncio.gather(
            embeddings_index._batch_get_embeddings("ok"),
            embeddings_index._batch_get_embeddings("great"),
        ),
        timeout=0.2,
    )

    assert recovered_results == [[2.0], [5.0]]
