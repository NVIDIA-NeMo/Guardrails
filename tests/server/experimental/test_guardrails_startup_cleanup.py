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

"""Retain real core owners when startup rollback cannot finish."""

import asyncio

import anyio
import httpx
import pytest
import pytest_asyncio

import nemoguardrails
from nemoguardrails import Guardrails
from nemoguardrails.guardrails import _lifecycle
from nemoguardrails.guardrails.base_engine import BaseEngine
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._runtime import get_runtime
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry, _repository
from nemoguardrails.server.experimental.rails._repository import GuardrailsRepository


class StartupEngine(BaseEngine):
    """Acquire an HTTPX owner before startup fails or is cancelled."""

    def __init__(self, failure, cleanup):
        """Control startup interruption and whether cleanup can complete."""
        super().__init__()
        self.failure = failure
        self.cleanup = cleanup
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.client = None
        self.stop_attempts = 0
        self.start_error = ValueError("startup marker")

    async def start(self):
        """Acquire the HTTP client before the injected startup interruption."""
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
        self.started.set()
        if self.failure == "cancel":
            await asyncio.Event().wait()
        raise self.start_error

    async def stop(self):
        """Keep the client open until cleanup is permitted to finish."""
        self.stop_attempts += 1
        if not self.release.is_set():
            if self.cleanup == "timeout":
                await self.release.wait()
            else:
                raise RuntimeError("cleanup marker")
        if self.client is not None:
            await self.client.aclose()


@pytest_asyncio.fixture
async def startup_owner(monkeypatch, failure, cleanup):
    """Keep real Guardrails and HTTPX lifecycle owners behind a controlled engine."""

    monkeypatch.setattr(nemoguardrails, "LLMRails", Guardrails)
    monkeypatch.setattr(_lifecycle, "_ROLLBACK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(_repository, "_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    probe = StartupEngine(failure, cleanup)
    facades = []

    def create_facade(**kwargs):
        facade = Guardrails(**kwargs)
        if not facades:
            facade.rails_engine.engine_registry._engines = {"startup-probe": probe}
        facades.append(facade)
        return facade

    monkeypatch.setattr(_repository, "Guardrails", create_facade)
    try:
        yield probe, facades
    finally:
        probe.release.set()
        for facade in facades:
            await facade.shutdown()
            await facade.rails_engine.rails_manager.stop()
        if probe.client is not None:
            await probe.client.aclose()


async def failed_startup(start, probe, failure):
    """Interrupt startup after acquisition and preserve its original error."""

    task = asyncio.create_task(start())
    with anyio.fail_after(1):
        await probe.started.wait()
        if failure == "cancel":
            task.cancel("startup interrupted")
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError) as caught:
            await task
    if failure == "cancel":
        assert task.cancelled()
    else:
        assert caught.value.__cause__ is probe.start_error


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "cancel"])
@pytest.mark.parametrize("cleanup", ["raise", "timeout"])
async def test_failed_startup_keeps_real_engine_for_cleanup_retry(tmp_path, startup_owner, failure, cleanup):
    """A failed core stays owned without becoming available for requests."""

    probe, facades = startup_owner
    directory = tmp_path / "rails"
    directory.mkdir()
    (directory / "config.yml").write_text("models: []\n")
    repository = GuardrailsRepository((ConfigEntry("policy", directory),))

    await failed_startup(repository.startup, probe, failure)

    assert repository._checkers["policy"]._rails is facades[0]
    assert not probe.client.is_closed
    assert probe.stop_attempts >= 2
    with pytest.raises(RuntimeError, match="closed"):
        repository.get("policy")
    with pytest.raises(RuntimeError, match="closed"):
        await repository.startup()

    probe.release.set()
    await repository.close()
    assert not repository._checkers
    assert probe.client.is_closed
    attempts = probe.stop_attempts
    await repository.close()
    assert probe.stop_attempts == attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "cancel"])
@pytest.mark.parametrize("cleanup", ["raise", "timeout"])
async def test_application_retains_failed_startup_until_cleanup_finishes(tmp_path, startup_owner, failure, cleanup):
    """Application restart waits for the previous core's cleanup to succeed."""

    probe, facades = startup_owner
    directory = tmp_path / "rails"
    directory.mkdir()
    (directory / "config.yml").write_text("models: []\n")
    application = create_proxy_application(
        providers=(create_openai_chat_integration(),),
        rails_configs=(ConfigEntry("policy", directory),),
    )

    async def start():
        async with application.router.lifespan_context(application):
            pytest.fail("failed startup must never become ready")

    await failed_startup(start, probe, failure)
    runtime = application.state.proxy_cleanup_runtime
    assert runtime.content_checkers._checkers["policy"]._rails is facades[0]
    assert runtime.http_client.is_closed
    assert not probe.client.is_closed
    with pytest.raises(RuntimeError, match="not available"):
        get_runtime(application)
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        async with application.router.lifespan_context(application):
            pytest.fail("pending cleanup must prevent a replacement runtime")
    assert len(facades) == 1

    probe.release.set()
    async with application.router.lifespan_context(application):
        assert probe.client.is_closed
        assert len(facades) == 2
        assert get_runtime(application) is not runtime
        assert not hasattr(application.state, "proxy_cleanup_runtime")
    assert not hasattr(application.state, "proxy_runtime")
