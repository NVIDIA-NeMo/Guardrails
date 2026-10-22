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

"""Load and own Guardrails content-checking adapters."""

import asyncio
import logging
from collections.abc import Sequence

import anyio

import nemoguardrails
from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.server.experimental._content_checker import ContentChecker, validate_content_checker
from nemoguardrails.server.experimental.rails._adapter import LegacyGuardrailsChecker
from nemoguardrails.server.experimental.rails.config import ConfigEntry, validate_config_entries

log = logging.getLogger(__name__)
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class GuardrailsRepository:
    """Preload the trusted catalog before serving and own its checkers.

    File parsing runs in a worker thread; engine construction and startup stay
    on the application loop. Requests only look up already-started checkers.
    Startup failure attempts cleanup of all acquired engines and prevents
    readiness. Shutdown gives each owner a shielded, cooperative budget;
    failed cleanup is retryable.
    """

    def __init__(self, entries: Sequence[ConfigEntry]):
        resolved_entries = validate_config_entries(entries)
        self._config_paths = {entry.id: entry.config_path for entry in resolved_entries}
        self._checkers: dict[str, LegacyGuardrailsChecker] = {}
        self._ready = False
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()

    async def startup(self) -> None:
        """Initialize every declared config once, or roll back the whole catalog.

        Retain engines before startup so incomplete core rollback can be retried.
        Failed or cancelled startup attempts cleanup of all acquired engines.
        A failed repository is closed, not reusable.
        """

        try:
            async with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("The Guardrails repository is closed.")
                if self._ready:
                    return
                for config_id, path in self._config_paths.items():
                    config = await asyncio.to_thread(RailsConfig.from_path, str(path))
                    rails = Guardrails(
                        config=config,
                        use_iorails=nemoguardrails.LLMRails is nemoguardrails.Guardrails,
                    )
                    self._checkers[config_id] = LegacyGuardrailsChecker(rails)
                    validate_content_checker(self._checkers[config_id])
                    await rails.startup()
                self._ready = True
        except BaseException:
            await self.close()
            raise

    @property
    def has_pending_cleanup(self) -> bool:
        """Report retained engines after an incomplete close attempt."""

        return self._closed and bool(self._checkers)

    def get(self, config_id: str) -> ContentChecker:
        """Look up a preloaded checker without I/O or request-time initialization."""

        if self._closed:
            raise RuntimeError("The Guardrails repository is closed.")
        if not self._ready:
            raise RuntimeError("The Guardrails repository has not completed startup.")
        if config_id not in self._checkers:
            raise ValueError(f"Unknown Rails config: {config_id}")
        return self._checkers[config_id]

    async def _close_checker(self, config_id: str, checker: LegacyGuardrailsChecker) -> None:
        """Close one initialized checker without blocking its peers."""

        try:
            with anyio.move_on_after(_SHUTDOWN_TIMEOUT_SECONDS, shield=True) as scope:
                await checker.close()
            if scope.cancelled_caught:
                log.warning("Guardrails cleanup timed out; resources may remain open.")
            else:
                self._checkers.pop(config_id, None)
        except Exception:
            log.warning("Guardrails cleanup failed; resources may remain open.")

    async def close(self) -> None:
        """Attempt every owner independently; never restart a closed repository.

        Five seconds is a cooperative per-owner budget, not forced termination
        of blocking code or cancellation-suppressing engines. Successful owners
        are forgotten; another close call may retry incomplete cleanup.
        """

        self._closed = True
        with anyio.CancelScope(shield=True):
            async with self._lifecycle_lock:
                async with anyio.create_task_group() as group:
                    for config_id, checker in tuple(self._checkers.items()):
                        group.start_soon(self._close_checker, config_id, checker)
