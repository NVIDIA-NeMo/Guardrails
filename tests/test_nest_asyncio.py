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
import importlib

import pytest

import nemoguardrails
from nemoguardrails import RailsConfig
from tests.utils import TestChat

config = RailsConfig.from_content(yaml_content="""models: []""")

chat = TestChat(
    config,
    llm_completions=[
        "Hello there!",
        "Hello there!",
        "Hello there!",
    ],
)


def test_sync_api():
    chat >> "Hi!"
    chat << "Hello there!"


@pytest.mark.asyncio
async def test_async_api(monkeypatch):
    monkeypatch.setenv("DISABLE_NEST_ASYNCIO", "False")

    importlib.reload(nemoguardrails)

    chat >> "Hi!"
    chat << "Hello there!"


@pytest.mark.asyncio
async def test_async_api_error(monkeypatch):
    monkeypatch.setenv("DISABLE_NEST_ASYNCIO", "True")

    # Reload the module to re-run its top-level code with the new env var
    importlib.reload(nemoguardrails)
    importlib.reload(nemoguardrails.patch_asyncio)
    importlib.reload(asyncio)

    # Remove the patching marker
    delattr(asyncio, "_nest_patched")

    assert nemoguardrails.patch_asyncio.nest_asyncio_patch_applied is False
    assert not hasattr(asyncio, "_nest_patched")

    with pytest.raises(
        RuntimeError,
        match=r"await generate_async",
    ):
        chat >> "Hi!"
        chat << "Hello there!"
