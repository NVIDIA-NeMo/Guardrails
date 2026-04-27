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
import os
import threading
from typing import Dict

from nemoguardrails.types import LLMFramework

_frameworks: Dict[str, LLMFramework] = {}
_default_framework: str = os.environ.get("NEMOGUARDRAILS_LLM_FRAMEWORK", "default")


def register_framework(name: str, framework: LLMFramework) -> None:
    if name in _frameworks:
        raise ValueError(f"Framework '{name}' is already registered.")
    _frameworks[name] = framework


def get_framework(name: str) -> LLMFramework:
    if name not in _frameworks:
        if name == "langchain":
            from nemoguardrails.integrations.langchain.llm_adapter import LangChainFramework

            _frameworks["langchain"] = LangChainFramework()
        elif name == "default":
            from nemoguardrails.llm.default_framework import DefaultFramework

            _frameworks["default"] = DefaultFramework()
        else:
            available = list(_frameworks.keys())
            raise KeyError(f"Unknown framework '{name}'. Available frameworks: {available}")
    return _frameworks[name]


_LAZY_FRAMEWORKS = {"langchain", "default"}


def set_default_framework(name: str) -> None:
    if name not in _frameworks and name not in _LAZY_FRAMEWORKS:
        raise KeyError(f"Unknown framework '{name}'. Register it first or use one of: {sorted(_LAZY_FRAMEWORKS)}")
    global _default_framework
    _default_framework = name


def get_default_framework() -> str:
    return _default_framework


def _reset_frameworks() -> None:
    global _default_framework

    frameworks_to_close = list(_frameworks.values())
    if frameworks_to_close:
        _run_async_from_sync(_close_frameworks(frameworks_to_close))

    _frameworks.clear()
    _default_framework = os.environ.get("NEMOGUARDRAILS_LLM_FRAMEWORK", "default")


async def _close_frameworks(frameworks) -> None:
    for fw in frameworks:
        await fw.reset()


def _run_async_from_sync(coro) -> None:
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        asyncio.run(coro)
        return

    error_holder = []

    def _runner():
        try:
            asyncio.run(coro)
        except BaseException as exc:
            error_holder.append(exc)

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join()
    if error_holder:
        raise error_holder[0]
