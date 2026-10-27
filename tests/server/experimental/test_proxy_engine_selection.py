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

"""Compare the proxy checking path with the server's selectable Rails engines."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import httpx
import pytest

from nemoguardrails import Guardrails
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._content_checker import ContentAllowed, InputContentCheck
from nemoguardrails.server.experimental._runtime import get_runtime
from nemoguardrails.server.experimental.provider.types import GuardedMessage
from nemoguardrails.server.experimental.providers.openai.integration import create_openai_chat_integration
from nemoguardrails.server.experimental.rails import ConfigEntry, _repository

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("value", [None, "0", "false", "1", "true", "yes", "TRUE"])
def test_proxy_uses_the_server_import_time_engine_switch(value):
    """Select the same constructor as the server in a fresh interpreter."""
    env = os.environ.copy()
    env.pop("NEMO_GUARDRAILS_IORAILS_ENGINE", None)
    if value is not None:
        env["NEMO_GUARDRAILS_IORAILS_ENGINE"] = value
    source = dedent("""\
        import nemoguardrails
        from nemoguardrails.server.experimental.rails import _repository
        assert _repository.LLMRails is nemoguardrails.LLMRails
        print(_repository.LLMRails.__name__)
    """)
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    expected = "Guardrails" if value is not None and value.lower() in {"1", "true", "yes"} else "LLMRails"
    assert result.stdout.strip() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("constructor", [LLMRails, Guardrails], ids=["llmrails", "iorails"])
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("scenario", ["allow", "input-block", "output-block"])
async def test_real_engines_enforce_the_same_proxy_checks(tmp_path, monkeypatch, caplog, constructor, stream, scenario):
    """Keep forwarding bytes and check-before-dispatch/release behavior identical."""
    monkeypatch.setattr(_repository, "LLMRails", constructor)
    directory = tmp_path / "rails"
    directory.mkdir()
    (directory / "config.yml").write_text(
        dedent("""\
            models: []
            rails:
              config:
                regex_detection:
                  input:
                    patterns: [BLOCK_INPUT]
                  output:
                    patterns: [BLOCK_OUTPUT]
              input:
                flows: [regex check input]
              output:
                flows: [regex check output]
                streaming:
                  enabled: true
                  chunk_size: 8
                  context_size: 0
                  stream_first: false
        """),
        encoding="utf-8",
    )
    text = "BLOCK_OUTPUT" if scenario == "output-block" else "allowed answer"
    if stream:
        chunk = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}}]}
        body = b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"
    else:
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
    received = []
    sources = []

    async def upstream(request):
        received.append(await request.aread())
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if stream else "application/json"},
            stream=httpx.ByteStream(body),
        )
        sources.append(response)
        return response

    application = create_proxy_application(
        providers=(create_openai_chat_integration(base_url="https://provider.test/v1"),),
        rails_configs=(ConfigEntry("policy", directory),),
        http_transport_factory=lambda: httpx.MockTransport(upstream),
    )
    request_body = json.dumps(
        {
            "model": "synthetic-model",
            "messages": [{"role": "user", "content": "BLOCK_INPUT" if scenario == "input-block" else "question"}],
            "stream": stream,
        },
        indent=2,
    ).encode()
    with caplog.at_level("INFO", logger=_repository.__name__):
        async with application.router.lifespan_context(application):
            runtime = get_runtime(application)
            rails = runtime.content_checkers.get("policy")._rails
            engine = rails.rails_engine if isinstance(rails, Guardrails) else rails
            assert isinstance(engine, IORails if constructor is Guardrails else LLMRails)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://proxy.test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions", content=request_body, headers={"content-type": "application/json"}
                )
    assert engine.__class__.__name__ in caplog.text
    assert runtime.http_client.is_closed
    assert all(source.is_closed for source in sources)
    if constructor is Guardrails:
        assert not engine.is_running
    if scenario == "input-block":
        assert response.status_code == 400
        assert received == []
    else:
        assert received == [request_body]
        if scenario == "allow":
            assert response.status_code == 200
            assert response.content == body
        else:
            assert response.status_code == (200 if stream else 400)
            assert b"BLOCK_OUTPUT" not in response.content
            assert "guardrails_violation" in response.text
    with pytest.raises(RuntimeError, match="closed"):
        runtime.content_checkers.get("policy")


@pytest.mark.asyncio
async def test_proxy_keeps_the_servers_llmrails_fallback(tmp_path, monkeypatch, caplog):
    """Use and report the actual legacy engine when a custom flow needs Colang."""
    monkeypatch.setattr(_repository, "LLMRails", Guardrails)
    directory = tmp_path / "rails"
    directory.mkdir()
    (directory / "config.yml").write_text("models: []\nrails:\n  input:\n    flows: [custom input]\n", encoding="utf-8")
    (directory / "rails.co").write_text("define flow custom input\n  $allowed = True\n", encoding="utf-8")
    repository = _repository.GuardrailsRepository((ConfigEntry("custom", directory),))
    try:
        with caplog.at_level("INFO", logger=_repository.__name__):
            await repository.startup()
        checker = repository.get("custom")
        assert isinstance(checker._rails.rails_engine, LLMRails)
        assert "Initialized proxy checker using LLMRails." in caplog.text
        assert isinstance(await checker.check_input(InputContentCheck(GuardedMessage("user", "hello"))), ContentAllowed)
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_proxy_rejects_colang_two_before_constructing_an_engine(tmp_path, monkeypatch):
    """Engine selection must not admit configurations unsupported by check_async."""
    directory = tmp_path / "rails"
    directory.mkdir()
    (directory / "config.yml").write_text("models: []\ncolang_version: 2.x\n", encoding="utf-8")

    def unexpected_engine(**kwargs):
        pytest.fail("unsupported checking configuration must be rejected before construction")

    monkeypatch.setattr(_repository, "LLMRails", unexpected_engine)
    repository = _repository.GuardrailsRepository((ConfigEntry("unsupported", directory),))
    with pytest.raises(ValueError, match="Colang 1.0"):
        await repository.startup()
    with pytest.raises(RuntimeError, match="closed"):
        repository.get("unsupported")
