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

"""Public runtime and server paths remain usable with removed model modules blocked."""

import subprocess
import sys
import textwrap


def test_public_paths_cannot_import_legacy_model_modules():
    source = textwrap.dedent(
        """
        import asyncio
        import importlib.abc
        import socket
        import sys

        blocked = {
            "nemoguardrails.guardrails.model_engine",
            "nemoguardrails.guardrails.base_engine",
            "nemoguardrails.guardrails.engine_registry",
            "nemoguardrails.guardrails._http",
        }

        class BlockLegacy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise AssertionError(f"Legacy model import: {fullname}")
                return None

        def deny_network(*args, **kwargs):
            raise AssertionError("The boundary test must not contact a provider")

        sys.meta_path.insert(0, BlockLegacy())
        socket.socket.connect = deny_network
        socket.socket.connect_ex = deny_network

        from nemoguardrails import Guardrails, RailsConfig
        from nemoguardrails.llm.frameworks import registry
        from nemoguardrails.llm.frameworks.default import DefaultFramework
        from nemoguardrails.testing.fake_model import FakeLLMModel
        import nemoguardrails.server.api
        import nemoguardrails.server.hooks.app
        import nemoguardrails.server.exception_handlers
        import nemoguardrails.telemetry

        nemoguardrails.telemetry.report_usage = lambda *args, **kwargs: None
        framework = DefaultFramework()
        framework.register_provider("boundary", lambda model, **kwargs: FakeLLMModel(responses=["configured"]))
        registry._frameworks = {"default": framework}
        registry._default_framework = "default"

        async def exercise():
            injected = Guardrails(
                RailsConfig(models=[]), llm=FakeLLMModel(responses=["buffered", "streamed"]), require_iorails=True
            )
            configured = Guardrails(
                RailsConfig.from_content(config={"models": [{"type": "main", "engine": "boundary", "model": "main"}]}),
                require_iorails=True,
            )
            try:
                assert (await injected.generate_async(messages=[{"role": "user", "content": "hi"}]))["content"] == "buffered"
                chunks = [chunk async for chunk in injected.stream_async(messages=[{"role": "user", "content": "hi"}])]
                assert "streamed" in "".join(chunk for chunk in chunks if isinstance(chunk, str))
                assert (await configured.generate_async(messages=[{"role": "user", "content": "hi"}]))["content"] == "configured"
            finally:
                await injected.shutdown()
                await configured.shutdown()

        asyncio.run(exercise())
        assert not blocked.intersection(sys.modules)
        """
    )
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
