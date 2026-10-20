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

"""Regression tests across the real IORails lifecycle, with controlled resources."""

import asyncio
import logging
import sys
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails import _lifecycle
from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.http.transport import HttpxHTTPClient


class ProbeEngine(BaseEngine):
    def __init__(self):
        super().__init__()
        self.start_entered = asyncio.Event()
        self.stop_entered = asyncio.Event()
        self.start_gate = asyncio.Event()
        self.stop_gate = asyncio.Event()
        self.start_gate.set()
        self.stop_gate.set()
        self.starts = 0
        self.stops = 0
        self.owned = False
        self.stop_error = None

    async def start(self):
        self.starts += 1
        self.owned = True
        self.start_entered.set()
        await self.start_gate.wait()

    async def stop(self):
        self.stops += 1
        self.stop_entered.set()
        await self.stop_gate.wait()
        if self.stop_error is not None:
            raise self.stop_error
        self.owned = False


@pytest.fixture
def rails(monkeypatch):
    monkeypatch.setattr("nemoguardrails.telemetry.report_usage", lambda *args, **kwargs: None)
    return Guardrails(RailsConfig.from_content(yaml_content="models: []"), require_iorails=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["task", "scope"])
async def test_cancelled_core_startup_rolls_back_partial_engine(rails, cancellation):
    first, second = ProbeEngine(), ProbeEngine()
    second.start_gate.clear()
    rails.rails_engine.engine_registry._engines = {"first": first, "second": second}
    if cancellation == "task":
        startup = asyncio.create_task(rails.startup())
        await second.start_entered.wait()
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
    else:
        with anyio.move_on_after(0.02) as scope:
            await rails.startup()
        assert scope.cancelled_caught
    assert not first.owned
    assert not second.owned
    assert first.stops == second.stops == 1
    assert not rails.rails_engine._running
    await rails.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_real_core_cleanup_retries_only_unfinished_engines(rails, failure):
    first, second = ProbeEngine(), ProbeEngine()
    rails.rails_engine.engine_registry._engines = {"first": first, "second": second}
    await rails.startup()
    if failure == "timeout":
        first.stop_gate.clear()
        with anyio.move_on_after(0.02) as scope:
            await rails.shutdown()
        assert scope.cancelled_caught
    else:
        first.stop_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            await rails.shutdown()
    assert first.owned
    previous_second_stops = second.stops
    first.stop_gate.set()
    first.stop_error = None
    await rails.shutdown()
    assert first.stops == 2
    assert not first.owned
    assert not second.owned
    assert second.stops == (previous_second_stops if previous_second_stops else 1)
    await rails.shutdown()
    assert first.stops == 2
    assert not rails.rails_engine._generate_async_queue._workers


@pytest.mark.asyncio
async def test_failed_startup_retains_failed_rollback_for_facade_shutdown(rails):
    first, second = ProbeEngine(), ProbeEngine()
    first.stop_error = RuntimeError("rollback failed")
    startup_error = ValueError("startup failed")
    second.start = AsyncMock(side_effect=startup_error)
    rails.rails_engine.engine_registry._engines = {"first": first, "second": second}
    with pytest.raises(RuntimeError, match="startup failed") as caught:
        await rails.startup()
    assert caught.value.__cause__ is startup_error
    assert first.owned
    previous_stops = first.stops
    first.stop_error = None
    await rails.shutdown()
    assert first.stops == previous_stops + 1
    assert not first.owned


@pytest.mark.asyncio
async def test_registry_restart_cannot_overwrite_failed_cleanup():
    registry = EngineRegistry([])
    engine = ProbeEngine()
    registry._engines = {"one": engine}
    await registry.start()
    engine.stop_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError):
        await registry.stop()
    with pytest.raises(RuntimeError):
        await registry.start()
    assert engine.starts == 1
    engine.stop_error = None
    await registry.stop()
    await registry.start()
    assert engine.starts == 2
    await registry.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_base_engine_retains_client_until_close_completes(monkeypatch, failure):
    engine = BaseEngine()
    await engine.start()
    client = engine._client
    actual_close = client.close
    gate = asyncio.Event()

    async def interrupted_close():
        if failure == "exception":
            raise RuntimeError("close failed")
        await gate.wait()

    closer = AsyncMock(side_effect=interrupted_close)
    monkeypatch.setattr(client, "close", closer)
    try:
        if failure == "exception":
            with pytest.raises(RuntimeError):
                await engine.stop()
        else:
            with anyio.move_on_after(0.02) as scope:
                await engine.stop()
            assert scope.cancelled_caught
        closer.side_effect = actual_close
        await engine.stop()
        assert closer.await_count == 2
        assert engine._client is None
    finally:
        await actual_close()


@pytest.mark.asyncio
async def test_interrupted_queue_drain_can_be_retried():
    queue = AsyncWorkQueue("lifecycle_probe", max_queue_size=2, max_concurrency=1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def work():
        entered.set()
        await release.wait()

    task = asyncio.create_task(queue.submit(work))
    await entered.wait()
    workers = tuple(queue._workers)
    try:
        with anyio.move_on_after(0.02) as scope:
            await queue.stop()
        assert scope.cancelled_caught
        release.set()
        await task
        await queue.stop()
        assert all(worker.done() for worker in workers)
        assert queue._workers == []
    finally:
        release.set()
        await task
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_starts_share_one_lifecycle(rails):
    engine = ProbeEngine()
    engine.start_gate.clear()
    rails.rails_engine.engine_registry._engines = {"one": engine}
    first = asyncio.create_task(rails.startup())
    await engine.start_entered.wait()
    second = asyncio.create_task(rails.startup())
    await asyncio.sleep(0)
    assert engine.starts == 1
    assert engine.stops == 0
    engine.start_gate.set()
    await asyncio.gather(first, second)
    await asyncio.gather(rails.shutdown(), rails.shutdown())
    assert engine.stops == 1


@pytest.mark.asyncio
async def test_rollback_timeout_preserves_startup_error_and_cleanup_ownership(rails, monkeypatch):
    monkeypatch.setattr(_lifecycle, "_ROLLBACK_TIMEOUT_SECONDS", 0.01)
    first, second = ProbeEngine(), ProbeEngine()
    first.stop_gate.clear()
    original = ValueError("startup failure")
    second.start = AsyncMock(side_effect=original)
    rails.rails_engine.engine_registry._engines = {"first": first, "second": second}
    with pytest.raises(RuntimeError) as caught:
        await rails.startup()
    assert caught.value.__cause__ is original
    assert first.owned
    first.stop_gate.set()
    await rails.shutdown()
    assert not first.owned
    assert not rails.rails_engine.engine_registry._owned_engines


@pytest.mark.asyncio
async def test_cancellation_survives_a_secondary_cleanup_error(rails):
    first, second = ProbeEngine(), ProbeEngine()
    first.stop_gate.clear()
    second.stop_error = ValueError("secondary cleanup failure")
    rails.rails_engine.engine_registry._engines = {"first": first, "second": second}
    await rails.startup()
    closing = asyncio.create_task(rails.shutdown())
    await first.stop_entered.wait()
    closing.cancel("original cancellation")
    with pytest.raises(
        asyncio.CancelledError,
        match="original cancellation" if sys.version_info >= (3, 11) else None,
    ):
        await closing
    assert second.stops == 1
    first.stop_gate.set()
    second.stop_error = None
    await rails.shutdown()
    assert not first.owned
    assert not second.owned


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_shutdown_logs_component_failures_without_changing_error_precedence(
    rails, monkeypatch, caplog, cancelled
):
    """Report each ordinary failure while retaining ownership and cancellation."""

    await rails.startup()
    core = rails.rails_engine
    first = asyncio.CancelledError("queue cleanup interrupted") if cancelled else ValueError("queue cleanup failed")
    second = RuntimeError("rails cleanup failed")
    queue_stop = AsyncMock(side_effect=first)
    rails_stop = AsyncMock(side_effect=second)
    registry_stop = AsyncMock(wraps=core.engine_registry.stop)
    caplog.clear()
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(core._generate_async_queue, "stop", queue_stop)
            patcher.setattr(core.rails_manager, "stop", rails_stop)
            patcher.setattr(core.engine_registry, "stop", registry_stop)
            with caplog.at_level(logging.WARNING, logger=IORails.__module__):
                with pytest.raises(type(first)) as caught:
                    await rails.shutdown()

        assert caught.value is first
        queue_stop.assert_awaited_once()
        rails_stop.assert_awaited_once()
        registry_stop.assert_awaited_once()
        assert not core.is_running
        assert core._owned_components == {"queue", "rails"}
        records = [record for record in caplog.records if record.name == IORails.__module__]
        components = ["rails"] if cancelled else ["queue", "rails"]
        errors = [second] if cancelled else [first, second]
        assert [record.getMessage() for record in records] == [
            f"Failed to stop IORails component {component}" for component in components
        ]
        assert all(record.levelno == logging.WARNING for record in records)
        assert all(record.exc_info is not None and record.exc_info[2] is not None for record in records)
        assert [record.exc_info[1] for record in records] == errors
    finally:
        await rails.shutdown()

    assert not core._owned_components


@pytest.mark.asyncio
async def test_failed_startup_can_retry_without_closing_unused_rail_http_client(rails):
    engine = ProbeEngine()
    actual_start = engine.start
    engine.start = AsyncMock(side_effect=ValueError("startup failed"))
    rails.rails_engine.engine_registry._engines = {"one": engine}
    with pytest.raises(RuntimeError):
        await rails.startup()
    engine.start = actual_start
    await rails.startup()
    assert not rails.rails_engine.rails_manager._http_client._closed
    await rails.shutdown()


@pytest.mark.asyncio
async def test_base_engine_restart_does_not_replace_client_with_pending_cleanup(monkeypatch):
    engine = BaseEngine()
    await engine.start()
    client = engine._client
    actual_close = client.close
    closer = AsyncMock(side_effect=RuntimeError("close failed"))
    monkeypatch.setattr(client, "close", closer)
    try:
        with pytest.raises(RuntimeError):
            await engine.stop()
        with pytest.raises(RuntimeError):
            await engine.start()
        assert engine._client is client
        closer.side_effect = actual_close
        await engine.stop()
    finally:
        await actual_close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_startup_and_clears_facade_readiness(rails):
    engine = ProbeEngine()
    engine.start_gate.clear()
    rails.rails_engine.engine_registry._engines = {"one": engine}
    startup = asyncio.create_task(rails.startup())
    await engine.start_entered.wait()
    shutdown = asyncio.create_task(rails.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    engine.start_gate.set()
    await asyncio.gather(startup, shutdown)
    assert engine.stops == 1
    assert not engine.owned
    assert not rails._started
    assert not rails.rails_engine._running


@pytest.mark.asyncio
async def test_repeated_task_cancellation_preserves_original_and_cleanup_ownership(rails):
    first, second = ProbeEngine(), ProbeEngine()
    first.stop_gate.clear()
    second.start_gate.clear()
    rails.rails_engine.engine_registry._engines = {"first": first, "second": second}
    startup = asyncio.create_task(rails.startup())
    await second.start_entered.wait()
    startup.cancel("original cancellation")
    await first.stop_entered.wait()
    startup.cancel("rollback cancellation")
    first.stop_gate.set()
    with pytest.raises(
        asyncio.CancelledError,
        match="original cancellation" if sys.version_info >= (3, 11) else None,
    ):
        await startup
    await rails.shutdown()
    assert not first.owned
    assert not second.owned


@pytest.mark.asyncio
async def test_facade_restarts_after_its_engine_is_stopped(rails):
    """Use engine readiness when deciding whether facade startup is needed."""

    engine = ProbeEngine()
    rails.rails_engine.engine_registry._engines = {"one": engine}
    await rails.startup()
    try:
        await rails.rails_engine.stop()
        await rails.startup()
        assert engine.starts == 2
        assert engine.owned
        assert rails._started
    finally:
        await rails.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_real_core_retains_http_owner_after_interrupted_or_failed_close(monkeypatch, failure):
    """Retry the real RailsManager owner after an unfinished HTTPX close."""

    entered = asyncio.Event()
    release = asyncio.Event()
    attempts = []
    completed = []

    async def close_transport():
        attempts.append(None)
        entered.set()
        await release.wait()
        if failure == "exception" and len(attempts) == 1:
            raise RuntimeError("transport cleanup failed")
        completed.append(None)

    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    monkeypatch.setattr(transport, "aclose", close_transport)
    owned = httpx.AsyncClient(transport=transport, trust_env=False)
    monkeypatch.setattr("nemoguardrails.http.transport.httpx.AsyncClient", lambda **kwargs: owned)
    rails = Guardrails(RailsConfig.from_content(yaml_content="models: []"), require_iorails=True)
    core = rails.rails_engine
    assert isinstance(core, IORails)
    owner = core.rails_manager._http_client
    assert isinstance(owner, HttpxHTTPClient)
    await rails.startup()
    try:
        if failure == "timeout":
            with anyio.move_on_after(0.01) as scope:
                await rails.shutdown()
            assert scope.cancelled_caught
        else:
            release.set()
            with pytest.raises(RuntimeError, match="transport cleanup failed"):
                await rails.shutdown()
        assert entered.is_set()
        assert owned.is_closed
        assert not completed
        assert "rails" in core._owned_components
        assert not owner._closed
        release.set()
        with anyio.fail_after(1):
            if failure == "exception":
                with pytest.raises(RuntimeError, match="transport cleanup failed"):
                    await rails.shutdown()
                assert not completed
                assert "rails" in core._owned_components
                assert not owner._closed
            else:
                await rails.shutdown()
                assert completed == [None]
                assert not core._owned_components
                assert owner._closed
        assert len(attempts) == 1
    finally:
        release.set()
        if failure == "timeout":
            await rails.shutdown()
        await owned.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("transport cleanup failed"), asyncio.CancelledError("cleanup interrupted")]
)
async def test_rollback_logs_cleanup_exception_and_traceback(caplog, error):
    """Report why rollback failed without replacing its caller's error."""

    original = ValueError("startup failed")

    async def fail_cleanup():
        raise error

    with caplog.at_level(logging.WARNING, logger=_lifecycle.__name__):
        with pytest.raises(ValueError) as caught:
            try:
                raise original
            except BaseException:
                await _lifecycle.rollback_startup(fail_cleanup)
                raise
    assert caught.value is original
    record = next(record for record in caplog.records if record.name == _lifecycle.__name__)
    assert record.exc_info is not None
    assert record.exc_info[1] is error
    assert record.exc_info[2] is not None
    assert str(error) in caplog.text
    assert "Traceback" in caplog.text
