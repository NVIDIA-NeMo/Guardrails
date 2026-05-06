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
import logging
import os
from types import MappingProxyType
from typing import Any, Callable, Mapping

from nemoguardrails.registry import Registry
from nemoguardrails.types import LLMFramework

log = logging.getLogger(__name__)


def _default_factory() -> LLMFramework:
    from nemoguardrails.llm.frameworks.default import DefaultFramework

    return DefaultFramework()


def _langchain_factory() -> LLMFramework:
    from nemoguardrails.integrations.langchain.llm_adapter import LangChainFramework

    return LangChainFramework()


class LLMFrameworkRegistry(Registry):
    _factories: Mapping[str, Callable[[], LLMFramework]] = MappingProxyType(
        {
            "default": _default_factory,
            "langchain": _langchain_factory,
        }
    )
    _active_name: str = os.environ.get("NEMOGUARDRAILS_LLM_FRAMEWORK", "default")

    def validate(self, name: str, item: Any) -> None:
        if not isinstance(item, LLMFramework):
            raise TypeError(f"{name!r} does not implement LLMFramework")
        if not asyncio.iscoroutinefunction(getattr(item, "reset", None)):
            raise TypeError(f"{name!r}.reset must be an async coroutine function")

    def get(self, name: str) -> LLMFramework:
        if name not in self.items and name in self._factories:
            self.add(name, self._factories[name]())
        return super().get(name)

    @property
    def active(self) -> str:
        return type(self)._active_name

    @active.setter
    def active(self, name: str) -> None:
        if name not in self.items and name not in self._factories:
            known = sorted(set(self.list()) | set(self._factories))
            raise KeyError(f"Unknown framework {name!r}. Available: {known}")
        type(self)._active_name = name


def register_framework(name: str, framework: LLMFramework) -> None:
    LLMFrameworkRegistry().add(name, framework)


def get_framework(name: str) -> LLMFramework:
    return LLMFrameworkRegistry().get(name)


def set_default_framework(name: str) -> None:
    LLMFrameworkRegistry().active = name


def get_default_framework() -> str:
    return LLMFrameworkRegistry().active


async def _areset_frameworks() -> None:
    registry = LLMFrameworkRegistry()
    frameworks_to_close = list(registry.items.values())
    try:
        for fw in frameworks_to_close:
            reset = getattr(fw, "reset", None)
            if reset is None:
                continue
            try:
                await reset()
            except Exception as exc:
                log.warning("Error resetting framework %r: %s", fw, exc)
    finally:
        registry.reset()
        type(registry)._active_name = os.environ.get("NEMOGUARDRAILS_LLM_FRAMEWORK", "default")


def _reset_frameworks() -> None:
    """Synchronous teardown helper. Must NOT be called from a running event loop.

    Wraps :func:`_areset_frameworks` in :func:`asyncio.run`, which raises
    ``RuntimeError`` if a loop is already running on the current thread.
    Async callers should ``await _areset_frameworks()`` directly.
    """
    asyncio.run(_areset_frameworks())
