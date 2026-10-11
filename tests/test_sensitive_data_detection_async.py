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

import asyncio
from threading import Barrier, Event

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.library.sensitive_data_detection import actions

has_sdd = (
    check_optional_dependency("presidio_analyzer")
    and check_optional_dependency("presidio_anonymizer")
    and check_optional_dependency("spacy")
    and check_optional_dependency("en_core_web_lg")
)
requires_sdd = pytest.mark.skipif(not has_sdd, reason="Requires the sdd extra and the en_core_web_lg model")


@requires_sdd
@pytest.mark.asyncio
@pytest.mark.parametrize("mask", [False, True], ids=["detection", "masking"])
async def test_allows_progress_during_analysis(mask):
    """Allow async progress and overlapping calls to a real Presidio recognizer."""
    from presidio_analyzer import Pattern, PatternRecognizer

    started, release, finished = Event(), Event(), Event()
    barrier = Barrier(2, action=started.set)

    class GatedRecognizer(PatternRecognizer):
        """Keep actual pattern recognition active until the async task releases it."""

        def analyze(self, *args, **kwargs):
            """Expose the active analysis interval without replacing Presidio."""
            barrier.wait(timeout=10)
            try:
                release.wait(timeout=10)
                return super().analyze(*args, **kwargs)
            finally:
                finished.set()

    recognizer = GatedRecognizer(
        supported_entity="ASYNC_TEST",
        name="Async test recognizer",
        patterns=[Pattern(name="secret", regex="secret", score=1.0)],
    )
    analyzer = actions._get_analyzer(score_threshold=0.4)
    analyzer.registry.add_recognizer(recognizer)
    config = RailsConfig.from_content(
        yaml_content="""
rails:
  config:
    sensitive_data_detection:
      input:
        entities: [ASYNC_TEST]
        score_threshold: 0.4
"""
    )
    action = actions.mask_sensitive_data if mask else actions.detect_sensitive_data
    tasks = [asyncio.create_task(action("input", text, config)) for text in ["secret", "harmless"]]
    try:
        assert await asyncio.to_thread(started.wait, 10), "Analysis did not start"
        assert not finished.is_set(), "Analysis blocked the event loop until it finished"
        assert all(not task.done() for task in tasks)
        release.set()
        expected = (
            [
                RailOutcome.transform(
                    [(TransformTarget.USER_MESSAGE, "<ASYNC_TEST>")],
                    metadata={"source": "input", "text": "secret", "masked_text": "<ASYNC_TEST>"},
                ),
                RailOutcome.allow(metadata={"source": "input", "text": "harmless", "masked_text": "harmless"}),
            ]
            if mask
            else [
                RailOutcome.block(metadata={"has_sensitive_data": True}),
                RailOutcome.allow(metadata={"has_sensitive_data": False}),
            ]
        )
        assert await asyncio.gather(*tasks) == expected
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        analyzer.registry.remove_recognizer(recognizer.name)


@requires_sdd
@pytest.mark.asyncio
async def test_concurrent_cold_analyzer_initialization():
    """Concurrent cache misses must share a single real model initialization."""
    actions._create_analyzer.cache_clear()
    barrier = Barrier(2)

    def get_analyzer():
        barrier.wait(timeout=10)
        return actions._get_analyzer()

    first, second = await asyncio.gather(
        asyncio.to_thread(get_analyzer),
        asyncio.to_thread(get_analyzer),
    )
    assert first is second
    assert actions._create_analyzer.cache_info().misses == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_running", [False, True])
async def test_cancelled_detection_work(cancel_running):
    """Skip cancelled queued work and isolate running work from the shared pool."""
    started = [Event(), Event()]
    release, finished, queued_started = Event(), Event(), Event()

    def occupy_worker(index):
        """Hold both workers while testing cancellation and executor isolation."""
        started[index].set()
        try:
            release.wait(timeout=10)
        finally:
            finished.set()

    with actions._DetectionExecutor(max_workers=2) as executor:
        running = [asyncio.create_task(executor.run(lambda i=i: occupy_worker(i))) for i in range(2)]
        queued = []
        try:
            for event in started:
                assert await asyncio.to_thread(event.wait, 10)
            queued = [asyncio.create_task(executor.run(queued_started.set)) for _ in range(2)]
            await asyncio.sleep(0)
            for task in running if cancel_running else queued:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not finished.is_set()
            assert not queued_started.is_set()
            assert await asyncio.to_thread(lambda: "available") == "available"
        finally:
            release.set()
            await asyncio.gather(*running, *queued, return_exceptions=True)
    assert finished.is_set()
    assert queued_started.is_set() == cancel_running
