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
from contextvars import ContextVar
from threading import Event

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.sensitive_data_detection import actions


@pytest.fixture(scope="module")
def require_presidio():
    """Use the installed real integration without downloading models in tests."""
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("presidio_anonymizer")
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_lg"):
        pytest.skip("The en_core_web_lg model must already be installed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("My name is John Smith.", RailOutcome.block(metadata={"has_sensitive_data": True})),
        ("Discuss data security.", RailOutcome.allow(metadata={"has_sensitive_data": False})),
    ],
)
async def test_detection_verdicts(require_presidio, text, expected):
    """Preserve real Presidio verdicts on both initial and repeated calls."""
    config = RailsConfig.from_content(
        yaml_content="""
rails:
  config:
    sensitive_data_detection:
      input:
        entities: [PERSON]
"""
    )
    for _ in range(2):
        assert await actions.detect_sensitive_data("input", text, config) == expected


@pytest.mark.asyncio
async def test_detection_allows_progress_during_analysis(require_presidio):
    """Run another task while a real custom Presidio recognizer is analyzing."""
    from presidio_analyzer import Pattern, PatternRecognizer

    started, release, finished = Event(), Event(), Event()

    class GatedRecognizer(PatternRecognizer):
        """Keep actual pattern recognition active until the async task releases it."""

        def analyze(self, *args, **kwargs):
            """Expose the active analysis interval without replacing Presidio."""
            started.set()
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
    task = asyncio.create_task(actions.detect_sensitive_data("input", "secret", config))
    try:
        assert await asyncio.to_thread(started.wait, 10), "Analysis did not start"
        assert not finished.is_set(), "Analysis blocked the event loop until it finished"
        assert not task.done()
        release.set()
        assert await task == RailOutcome.block(metadata={"has_sensitive_data": True})
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        analyzer.registry.remove_recognizer(recognizer.name)


@pytest.mark.asyncio
async def test_concurrent_cold_analyzer_initialization(require_presidio):
    """Concurrent cache misses must share a single real model initialization."""
    actions._create_analyzer.cache_clear()
    first, second = await asyncio.gather(
        asyncio.to_thread(actions._get_analyzer),
        asyncio.to_thread(actions._get_analyzer),
    )
    assert first is second
    assert actions._create_analyzer.cache_info().misses == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_running", [False, True])
async def test_cancelled_detection_work(cancel_running):
    """Cancellation skips queued work and leaves at most one running analysis."""
    started, release, finished, queued_started = Event(), Event(), Event(), Event()

    def occupy_worker():
        """Hold the worker while testing cancellation and executor isolation."""
        started.set()
        try:
            release.wait(timeout=10)
        finally:
            finished.set()

    running = asyncio.create_task(actions._run_detection(occupy_worker))
    queued = None
    try:
        assert await asyncio.to_thread(started.wait, 10)
        queued = asyncio.create_task(actions._run_detection(queued_started.set))
        await asyncio.sleep(0)
        cancelled = running if cancel_running else queued
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert not finished.is_set()
        assert not queued_started.is_set()
        # The shared executor remains available even when detection is occupied.
        assert await asyncio.to_thread(lambda: "available") == "available"
    finally:
        release.set()
        await asyncio.gather(running, *([queued] if queued else []), return_exceptions=True)
        # Drain the dedicated worker, including work whose awaiter was cancelled.
        await actions._run_detection(lambda: None)
    assert finished.is_set()
    assert queued_started.is_set() == cancel_running


@pytest.mark.asyncio
async def test_detection_worker_preserves_context_and_errors():
    """Preserve request context and propagate analysis errors to the caller."""
    request_id = ContextVar("presidio_test_request_id")
    token = request_id.set("request-123")
    try:
        assert await actions._run_detection(request_id.get) == "request-123"
    finally:
        request_id.reset(token)
    with pytest.raises(LookupError):
        await actions._run_detection(request_id.get)
