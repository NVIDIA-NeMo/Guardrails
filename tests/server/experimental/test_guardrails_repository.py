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
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.http.transport import HttpxHTTPClient
from nemoguardrails.server.experimental import _runtime, rails
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._runtime import ServerRuntime
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry, _repository
from nemoguardrails.server.experimental.rails._adapter import LegacyGuardrailsChecker
from nemoguardrails.server.experimental.rails._repository import GuardrailsRepository
from nemoguardrails.server.experimental.rails.config import validate_config_entries


@pytest.mark.asyncio
async def test_application_preloads_even_unselected_configs_before_readiness(tmp_path, monkeypatch):
    events = []
    path = config_path(tmp_path)
    engines = []

    def create_engine(*, config, require_iorails):
        assert isinstance(config, RailsConfig)
        assert require_iorails
        index = len(engines)

        async def start():
            events.append(f"start-{index}")

        instance = MagicMock(config=config, startup=AsyncMock(side_effect=start), shutdown=AsyncMock())
        engines.append(instance)
        return instance

    monkeypatch.setattr(_repository, "Guardrails", create_engine)
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(ConfigEntry("selected", path), ConfigEntry("unselected", path)),
        config_resolver=lambda _context: "selected",
        http_transport_factory=lambda: httpx.MockTransport(lambda _request: pytest.fail("no dispatch expected")),
    )
    assert not hasattr(application.state, "experimental_proxy_runtime")
    async with application.router.lifespan_context(application):
        events.append("ready")
        runtime = _runtime.get_runtime(application)
        assert runtime.content_checkers.get("selected") is runtime.content_checkers.get("selected")
        assert runtime.content_checkers.get("unselected") is not None
        assert events == ["start-0", "start-1", "ready"]
        assert not runtime.http_client.is_closed
    assert runtime.http_client.is_closed
    assert not hasattr(application.state, "experimental_proxy_runtime")
    for engine in engines:
        engine.startup.assert_awaited_once()
        engine.shutdown.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", ["parse", "construct", "startup", "cancel"])
async def test_application_startup_failure_rolls_back_catalog_and_http(tmp_path, monkeypatch, failure_at):
    path = config_path(tmp_path)
    first = MagicMock(startup=AsyncMock(), shutdown=AsyncMock())
    failure = asyncio.CancelledError() if failure_at == "cancel" else ValueError("invalid config")
    second = MagicMock(startup=AsyncMock(side_effect=failure), shutdown=AsyncMock())
    factory = MagicMock(side_effect=[first, failure if failure_at == "construct" else second])
    monkeypatch.setattr(_repository, "Guardrails", factory)
    if failure_at == "parse":
        monkeypatch.setattr(
            _repository.RailsConfig,
            "from_path",
            MagicMock(side_effect=[RailsConfig.from_content(yaml_content="models: []"), failure]),
        )
    transport = httpx.MockTransport(lambda _request: pytest.fail("no dispatch expected"))
    transport.aclose = AsyncMock()
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(ConfigEntry("first", path), ConfigEntry("second", path)),
        config_resolver=lambda _context: "first",
        http_transport_factory=lambda: transport,
    )
    with pytest.raises(type(failure)) as caught:
        async with application.router.lifespan_context(application):
            pytest.fail("partial startup must never become ready")
    assert caught.value is failure
    first.shutdown.assert_awaited_once()
    transport.aclose.assert_awaited_once()
    assert not hasattr(application.state, "experimental_proxy_runtime")


@pytest.mark.asyncio
async def test_file_loading_does_not_block_the_event_loop(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    config = RailsConfig.from_content(yaml_content="models: []")

    def read_config(_path):
        entered.set()
        release.wait(2)
        finished.set()
        return config

    instance = MagicMock(startup=AsyncMock(), shutdown=AsyncMock())
    monkeypatch.setattr(_repository.RailsConfig, "from_path", read_config)
    monkeypatch.setattr(_repository, "Guardrails", MagicMock(return_value=instance))
    repository = GuardrailsRepository((ConfigEntry("policy", config_path(tmp_path)),))
    load = asyncio.create_task(repository.startup())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not finished.is_set()
    finally:
        release.set()
        await load
        await repository.close()


@pytest.mark.asyncio
async def test_lookup_requires_the_whole_catalog_to_finish_startup(tmp_path, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def startup():
        entered.set()
        await release.wait()

    first = MagicMock(startup=AsyncMock(), shutdown=AsyncMock())
    second = MagicMock(startup=AsyncMock(side_effect=startup), shutdown=AsyncMock())
    monkeypatch.setattr(_repository, "Guardrails", MagicMock(side_effect=[first, second]))
    path = config_path(tmp_path)
    repository = GuardrailsRepository((ConfigEntry("first", path), ConfigEntry("second", path)))
    loading = asyncio.create_task(repository.startup())
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="completed startup"):
            repository.get("first")
    finally:
        release.set()
        await loading
    assert repository.get("first") is not repository.get("second")
    await repository.startup()
    await repository.close()
    for instance in (first, second):
        instance.startup.assert_awaited_once()
        instance.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_startup_rolls_back_completed_engines_and_rejects_lookup(tmp_path, monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def startup():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    first = MagicMock(startup=AsyncMock(), shutdown=AsyncMock())
    second = MagicMock(startup=AsyncMock(side_effect=startup), shutdown=AsyncMock())
    monkeypatch.setattr(_repository, "Guardrails", MagicMock(side_effect=[first, second]))
    path = config_path(tmp_path)
    repository = GuardrailsRepository((ConfigEntry("first", path), ConfigEntry("second", path)))
    waiting = asyncio.create_task(repository.startup())
    await entered.wait()
    waiting.cancel()
    try:
        with anyio.fail_after(0.5):
            with pytest.raises(asyncio.CancelledError):
                await waiting
        assert cancelled.is_set()
        first.shutdown.assert_awaited_once()
        with pytest.raises(RuntimeError, match="closed"):
            repository.get("first")
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["hang", "raise"])
async def test_shutdown_attempts_all_checkers_and_allows_failed_cleanup_retry(tmp_path, monkeypatch, caplog, failure):
    monkeypatch.setattr(_repository, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01, raising=False)

    async def fail():
        if failure == "hang":
            await asyncio.Event().wait()
        raise RuntimeError("private cleanup detail")

    first = MagicMock(startup=AsyncMock(), shutdown=AsyncMock(side_effect=fail))
    second = MagicMock(startup=AsyncMock(), shutdown=AsyncMock())
    monkeypatch.setattr(_repository, "Guardrails", MagicMock(side_effect=[first, second]))
    path = config_path(tmp_path)
    repository = GuardrailsRepository((ConfigEntry("first", path), ConfigEntry("second", path)))
    await repository.startup()
    try:
        with anyio.fail_after(0.5):
            await repository.close()
        first.shutdown.assert_awaited_once()
        second.shutdown.assert_awaited_once()
        assert "cleanup" in caplog.text.lower()
        assert "private cleanup detail" not in caplog.text
        first.shutdown.side_effect = None
        await repository.close()
        assert first.shutdown.await_count == 2
        second.shutdown.assert_awaited_once()
    finally:
        first.shutdown.side_effect = None


@pytest.mark.asyncio
async def test_shutdown_is_shielded_and_concurrent_closes_do_not_duplicate_cleanup(tmp_path, monkeypatch):
    async def shutdown():
        await anyio.lowlevel.checkpoint()

    instance = MagicMock(startup=AsyncMock(), shutdown=AsyncMock(side_effect=shutdown))
    monkeypatch.setattr(_repository, "Guardrails", MagicMock(return_value=instance))
    repository = GuardrailsRepository((ConfigEntry("policy", config_path(tmp_path)),))
    await repository.startup()
    with anyio.CancelScope() as scope:
        scope.cancel()
        await repository.close()
    await asyncio.gather(repository.close(), repository.close())
    instance.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_retries_real_core_shutdown_after_queue_timeout(tmp_path, monkeypatch):
    """Retain the real engine until interrupted worker cleanup completes."""

    monkeypatch.setattr(_repository, "_SHUTDOWN_TIMEOUT_SECONDS", 0.02)
    repository = GuardrailsRepository((ConfigEntry("policy", config_path(tmp_path)),))
    await repository.startup()
    facade = repository._checkers["policy"]._rails
    assert isinstance(facade, Guardrails)
    core = facade.rails_engine
    assert isinstance(core, IORails)
    queue = core._generate_async_queue
    workers = tuple(queue._workers)
    http_owner = core.rails_manager._http_client
    assert isinstance(http_owner, HttpxHTTPClient)
    http_client = http_owner._client
    entered = asyncio.Event()
    release = asyncio.Event()

    async def unfinished_check():
        entered.set()
        await release.wait()

    pending = asyncio.create_task(queue.submit(unfinished_check))
    try:
        with anyio.fail_after(1):
            await entered.wait()
            await repository.close()
        assert "policy" in repository._checkers
        assert any(not worker.done() for worker in workers)

        release.set()
        with anyio.fail_after(1):
            await pending
            await repository.close()
        assert not repository._checkers
        assert all(worker.done() for worker in workers)
        assert http_client.is_closed
    finally:
        release.set()
        pending.cancel()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(pending, *workers, return_exceptions=True)
        await http_client.aclose()
        await repository.close()


@pytest.mark.asyncio
async def test_http_client_closes_even_when_checker_shutdown_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(_repository, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01)

    async def shutdown():
        await asyncio.Event().wait()

    instance = MagicMock(startup=AsyncMock(), shutdown=AsyncMock(side_effect=shutdown))
    monkeypatch.setattr(_repository, "Guardrails", MagicMock(return_value=instance))
    runtime = ServerRuntime.create((ConfigEntry("policy", config_path(tmp_path)),), None)
    await runtime.content_checkers.startup()
    with anyio.fail_after(0.5):
        await runtime.close()
    assert runtime.http_client.is_closed
    instance.shutdown.side_effect = None
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["hang", "raise"])
async def test_http_shutdown_is_bounded_and_preserves_primary_failure(monkeypatch, caplog, failure):
    monkeypatch.setattr(_runtime, "_HTTP_SHUTDOWN_TIMEOUT_SECONDS", 0.01)

    async def close_http():
        if failure == "hang":
            await asyncio.Event().wait()
        raise RuntimeError("private HTTP cleanup detail")

    client = MagicMock(aclose=AsyncMock(side_effect=close_http))
    repository = MagicMock(close=AsyncMock())
    runtime = ServerRuntime(client, repository, None)
    with anyio.fail_after(0.5):
        with pytest.raises(ValueError, match="primary failure"):
            try:
                raise ValueError("primary failure")
            finally:
                await runtime.close()
    repository.close.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert "HTTP client cleanup" in caplog.text
    assert "private HTTP cleanup detail" not in caplog.text


def config_path(tmp_path):
    path = tmp_path / "rails"
    path.mkdir(exist_ok=True)
    (path / "config.yml").write_text("models: []\n", encoding="utf-8")
    return path


def test_rails_api_exports_only_config_resolution_types():
    expected = {"ConfigContext", "ConfigEntry", "ConfigResolver"}

    assert set(rails.__all__) == expected
    assert all(getattr(rails, name) is not None for name in expected)


@pytest.mark.asyncio
async def test_repository_caches_and_closes_guardrails_by_config_id(tmp_path):
    guardrails = MagicMock()
    guardrails.startup = AsyncMock()
    guardrails.shutdown = AsyncMock()
    repository = GuardrailsRepository((ConfigEntry("content-safety", config_path(tmp_path)),))

    with (
        patch("nemoguardrails.server.experimental.rails._repository.RailsConfig.from_path", return_value=MagicMock()),
        patch(
            "nemoguardrails.server.experimental.rails._repository.Guardrails",
            return_value=guardrails,
        ) as guardrails_class,
    ):
        await repository.startup()
        first = repository.get("content-safety")
        second = repository.get("content-safety")
        await repository.close()

    assert isinstance(first, LegacyGuardrailsChecker)
    assert second is first
    guardrails_class.assert_called_once()
    guardrails.startup.assert_awaited_once()
    guardrails.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_does_not_share_cores_between_config_ids(tmp_path):
    path = config_path(tmp_path)
    guardrails = [MagicMock(), MagicMock()]
    for instance in guardrails:
        instance.startup = AsyncMock()
        instance.shutdown = AsyncMock()
    repository = GuardrailsRepository((ConfigEntry("tenant-a", path), ConfigEntry("tenant-b", path)))

    with (
        patch("nemoguardrails.server.experimental.rails._repository.RailsConfig.from_path", return_value=MagicMock()),
        patch("nemoguardrails.server.experimental.rails._repository.Guardrails", side_effect=guardrails),
    ):
        await repository.startup()
        first = repository.get("tenant-a")
        second = repository.get("tenant-b")
        await repository.close()

    assert first is not second
    for instance in guardrails:
        instance.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_cannot_restart_after_startup_failure(tmp_path):
    guardrails = MagicMock()
    guardrails.startup = AsyncMock(side_effect=RuntimeError("broken config"))
    repository = GuardrailsRepository((ConfigEntry("broken", config_path(tmp_path)),))

    with (
        patch("nemoguardrails.server.experimental.rails._repository.RailsConfig.from_path", return_value=MagicMock()),
        patch("nemoguardrails.server.experimental.rails._repository.Guardrails", return_value=guardrails),
    ):
        with pytest.raises(RuntimeError, match="broken config"):
            await repository.startup()
        with pytest.raises(RuntimeError, match="closed"):
            await repository.startup()
        with pytest.raises(RuntimeError, match="closed"):
            repository.get("broken")
        await repository.close()

    guardrails.startup.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_rejects_unknown_config(tmp_path):
    repository = GuardrailsRepository(())
    await repository.startup()

    with pytest.raises(ValueError, match="Unknown Rails config"):
        repository.get("unknown")
    await repository.close()


def test_configs_are_validated_at_application_construction(tmp_path):
    path = config_path(tmp_path)

    with pytest.raises(ValueError, match="Duplicate Rails config"):
        validate_config_entries((ConfigEntry("duplicate", path), ConfigEntry("duplicate", path)))
    with pytest.raises(ValueError, match="configuration directory"):
        validate_config_entries((ConfigEntry("missing", tmp_path / "missing"),))


def test_multiple_configs_require_a_resolver(tmp_path):
    path = config_path(tmp_path)

    with pytest.raises(ValueError, match="require a resolver"):
        create_proxy_application(
            providers=(create_openai_chat_integration(),),
            rails_configs=(ConfigEntry("first", path), ConfigEntry("second", path)),
        )


def test_resolver_requires_a_config_entry():
    with pytest.raises(ValueError, match="requires at least one config entry"):
        create_proxy_application(
            providers=(create_openai_chat_integration(),),
            config_resolver=lambda _context: "missing",
        )
