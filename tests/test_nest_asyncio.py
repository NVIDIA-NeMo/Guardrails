# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

BLOCK_NEST_ASYNCIO = """
import sys
from importlib.abc import MetaPathFinder

assert "nest_asyncio" not in sys.modules
nest_asyncio_imports = []

class BlockNestAsyncio(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nest_asyncio":
            nest_asyncio_imports.append(fullname)
            raise ModuleNotFoundError("nest_asyncio is unavailable", name=fullname)

sys.meta_path.insert(0, BlockNestAsyncio())
"""

CHAT_SETUP = """
from nemoguardrails import RailsConfig
from tests.utils import TestChat

config = RailsConfig.from_content(yaml_content="models: []")
chat = TestChat(config, llm_completions=["Hello there!"])
"""


def _run_in_subprocess(source, disable_nest_asyncio):
    env = os.environ.copy()
    if disable_nest_asyncio is None:
        env.pop("DISABLE_NEST_ASYNCIO", None)
    else:
        env["DISABLE_NEST_ASYNCIO"] = "true" if disable_nest_asyncio else "false"
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_sync_api():
    _run_in_subprocess(
        CHAT_SETUP
        + """
chat >> "Hi!"
chat << "Hello there!"
""",
        True,
    )


def test_async_api():
    _run_in_subprocess(
        CHAT_SETUP
        + """
import asyncio
import nemoguardrails.patch_asyncio

assert nemoguardrails.patch_asyncio.nest_asyncio_patch_applied is True
assert hasattr(asyncio, "_nest_patched")

async def main():
    chat >> "Hi!"
    chat << "Hello there!"

asyncio.run(main())
""",
        False,
    )


def test_async_api_error():
    _run_in_subprocess(
        CHAT_SETUP
        + """
import asyncio
import nemoguardrails.patch_asyncio

assert nemoguardrails.patch_asyncio.nest_asyncio_patch_applied is False
assert not hasattr(asyncio, "_nest_patched")

async def main():
    try:
        chat >> "Hi!"
        chat << "Hello there!"
    except RuntimeError as exc:
        assert "await generate_async" in str(exc)
    else:
        raise AssertionError("The synchronous API did not reject a running event loop")

asyncio.run(main())
""",
        True,
    )


@pytest.mark.parametrize("disable_nest_asyncio", [None, True])
def test_sync_and_async_apis_without_nest_asyncio(disable_nest_asyncio):
    _run_in_subprocess(
        BLOCK_NEST_ASYNCIO
        + """
import warnings
warnings.simplefilter("error", FutureWarning)
"""
        + CHAT_SETUP
        + """
import asyncio
import nemoguardrails.patch_asyncio

assert nest_asyncio_imports == []
assert "nest_asyncio" not in sys.modules
assert not nemoguardrails.patch_asyncio.nest_asyncio_patch_applied
assert not hasattr(asyncio, "_nest_patched")

chat >> "Hi!"
chat << "Hello there!"

async def main():
    async_chat = TestChat(config, llm_completions=["Hello there!"])
    assert await async_chat.app.generate_async(prompt="Hi!") == "Hello there!"

asyncio.run(main())
assert nest_asyncio_imports == []
""",
        disable_nest_asyncio,
    )


def test_opt_in_without_nest_asyncio_has_installation_hint():
    _run_in_subprocess(
        BLOCK_NEST_ASYNCIO
        + """
try:
    import nemoguardrails
except ImportError as exc:
    assert "pip install 'nemoguardrails[nest-asyncio]'" in str(exc)
    assert "DISABLE_NEST_ASYNCIO=true" in str(exc)
    assert isinstance(exc.__cause__, ModuleNotFoundError)
else:
    raise AssertionError("Patching succeeded without nest_asyncio")
""",
        False,
    )


def test_opt_in_warns_about_future_deprecation():
    _run_in_subprocess(
        """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", FutureWarning)
    import nemoguardrails

future_warnings = [warning for warning in caught if warning.category is FutureWarning]
assert len(future_warnings) == 1, future_warnings
message = str(future_warnings[0].message)
assert "patching" in message
assert "deprecated in" in message
assert "0.26" in message
assert "generate_async" in message
assert "Synchronous APIs" in message
assert nemoguardrails.patch_asyncio.nest_asyncio_patch_applied
""",
        False,
    )


def test_apply_without_optional_dependency_raises(monkeypatch):
    import nemoguardrails.patch_asyncio as patch_asyncio

    monkeypatch.setenv("DISABLE_NEST_ASYNCIO", "false")
    monkeypatch.setattr(patch_asyncio, "nest_asyncio_patch_applied", False)
    monkeypatch.setitem(sys.modules, "nest_asyncio", None)

    with pytest.raises(ImportError) as exc_info:
        patch_asyncio.apply()

    assert "pip install 'nemoguardrails[nest-asyncio]'" in str(exc_info.value)
    assert "DISABLE_NEST_ASYNCIO=true" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
    assert patch_asyncio.nest_asyncio_patch_applied is False


def test_apply_enabled_warns_and_patches(monkeypatch):
    import nemoguardrails.patch_asyncio as patch_asyncio

    nest_asyncio = ModuleType("nest_asyncio")
    apply_patch = Mock()
    monkeypatch.setattr(nest_asyncio, "apply", apply_patch, raising=False)
    monkeypatch.setitem(sys.modules, "nest_asyncio", nest_asyncio)
    monkeypatch.setenv("DISABLE_NEST_ASYNCIO", "false")
    monkeypatch.setattr(patch_asyncio, "nest_asyncio_patch_applied", False)

    with pytest.warns(FutureWarning, match="patching will be deprecated in NeMo Guardrails 0.26") as caught:
        patch_asyncio.apply()

    assert "generate_async" in str(caught[0].message)
    assert "Synchronous APIs" in str(caught[0].message)
    assert caught[0].filename == __file__
    apply_patch.assert_called_once_with()
    assert patch_asyncio.nest_asyncio_patch_applied is True
