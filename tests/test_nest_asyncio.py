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

CHAT_SETUP = """
from nemoguardrails import RailsConfig
from tests.utils import TestChat

config = RailsConfig.from_content(yaml_content="models: []")
chat = TestChat(config, llm_completions=["Hello there!"])
"""


def _run_in_subprocess(source, disable_nest_asyncio):
    env = os.environ.copy()
    env["DISABLE_NEST_ASYNCIO"] = "true" if disable_nest_asyncio else "false"
    result = subprocess.run(
        [sys.executable, "-c", CHAT_SETUP + source],
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sync_api():
    _run_in_subprocess(
        """
chat >> "Hi!"
chat << "Hello there!"
""",
        True,
    )


def test_async_api():
    _run_in_subprocess(
        """
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
        """
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
