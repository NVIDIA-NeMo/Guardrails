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
from random import random
from time import time

import pytest

from nemoguardrails import LLMRails, RailsConfig


@pytest.mark.skip(reason="Run manually.")
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

    response = await rails.generate_async(prompt="hi!" * 100)

    assert response == "Hello World!"

    # Statistics
    total_time = 0
    completed_requests = 0
    req_counter = 0
    concurrency = 100
    requests = 1000
    latencies = []

    async def _generate(text, idx):
        nonlocal total_time, completed_requests, req_counter

        if idx < concurrency:
            # For the first `concurrency` calls, we introduce an artificial delay
            # to avoid everything being triggered at once, "in waves".
            delay = 10 * random()
            print(f"Initial delay of {delay}")
            await asyncio.sleep(delay)

        async with semaphore:
            req_counter += 1
            start_time = time()
            _response = await rails.generate_async(prompt=text)
            assert _response == "Hello World!"

            duration = time() - start_time
            latencies.append(duration)
            total_time += duration

            completed_requests += 1

    tasks = []
    t0 = time()
    semaphore = asyncio.Semaphore(concurrency)
    for i in range(requests):
        task = asyncio.ensure_future(_generate(f"hi {i}" * 100, i))
        tasks.append(task)

    await asyncio.gather(*tasks)
    took = time() - t0

    print(f"Processing {completed_requests} took {took:0.2f}.")

    print(f"Completed {completed_requests} requests in {total_time:.2f} seconds.")
    average_latency = total_time / completed_requests if completed_requests else 0

    print(f"Average latency: {average_latency:.2f} seconds.")
    print(f"Maximum concurrency: {concurrency}")
    print(f"Throughput: {completed_requests / took}")

    print(latencies)
