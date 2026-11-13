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

from unittest import mock

import httpx
import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.http import HTTPConnectionError, HTTPResponse
from nemoguardrails.library.jailbreak_detection.actions import jailbreak_detection_model
from nemoguardrails.library.jailbreak_detection.errors import JailbreakDetectionError
from nemoguardrails.library.jailbreak_detection.model_based import checks
from nemoguardrails.llm.cache.lfu import LFUCache
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.testing import RecordingHTTPClient
from tests.utils import TestChat


@pytest.fixture(autouse=True)
def restore_call_context():
    info_token = llm_call_info_var.set(None)
    log_token = processing_log_var.set(None)
    yield
    processing_log_var.reset(log_token)
    llm_call_info_var.reset(info_token)


def _config(backend, *, colang2=False):
    flow = "jailbreak detection heuristics" if backend == "heuristics" else "jailbreak detection model"
    detector_config = {}
    if backend == "nim":
        detector_config = {"nim_base_url": "https://detector.example", "nim_server_endpoint": "/check"}
    elif backend != "local":
        detector_config = {"server_endpoint": "https://detector.example/check"}
    config = {
        "models": [],
        "rails": {"input": {"flows": [flow]}, "config": {"jailbreak_detection": detector_config}},
    }
    colang = None
    if colang2:
        config["colang_version"] = "2.x"
        config["rails"].pop("input")
        colang = f"""
import core
import llm
import guardrails
import nemoguardrails.library.jailbreak_detection

flow input rails $input_text
  {flow}

flow main
  activate llm continuation
  user said something
  bot say "Unchecked input reached generation."
"""
    return RailsConfig.from_content(config=config, colang_content=colang)


def _queue_failure(httpx_mock, failure):
    if failure == "timeout":
        httpx_mock.add_exception(httpx.ReadTimeout("detector unavailable"))
    elif failure == "connection":
        httpx_mock.add_exception(httpx.ConnectError("detector unavailable"))
    elif failure == "malformed":
        httpx_mock.add_response(json={"jailbreak": None})
    else:
        httpx_mock.add_response(status_code=503, content=b"private response body")


@pytest.mark.parametrize("engine", ["colang1", "iorails"])
@pytest.mark.parametrize("backend", ["server", "nim"])
@pytest.mark.parametrize("failure", ["timeout", "connection", "status", "malformed"])
@pytest.mark.asyncio
async def test_detector_outage_stops_generation(engine, backend, failure, httpx_mock, caplog, monkeypatch):
    config = _config(backend)
    _queue_failure(httpx_mock, failure)
    messages = [{"role": "user", "content": "private checked input"}]
    if engine == "iorails":
        async with IORails(config) as rails:
            main_call = mock.AsyncMock(side_effect=AssertionError("Unchecked input reached generation."))
            monkeypatch.setattr(rails.engine_registry, "model_call", main_call)
            response = await rails.generate_async(messages=messages)
            main_call.assert_not_awaited()
    else:
        chat = TestChat(config, llm_completions=["Unchecked input reached generation."])
        response = await chat.app.generate_async(messages=messages)
        assert chat.llm.inference_count == 0
    assert response["content"] == "I'm sorry, an internal error has occurred."
    assert len(httpx_mock.get_requests()) == 1
    assert "private response body" not in caplog.text
    assert "private checked input" not in caplog.text


@pytest.mark.parametrize("backend", ["server", "nim", "heuristics"])
@pytest.mark.parametrize("failure", ["timeout", "connection", "status", "malformed"])
def test_detector_outage_stops_generation_colang2(backend, failure, httpx_mock, caplog):
    chat = TestChat(_config(backend, colang2=True), llm_completions=["Unchecked input reached generation."])
    _queue_failure(httpx_mock, failure)
    chat >> "private checked input"
    chat << "I'm sorry, an internal error has occurred."
    assert len(httpx_mock.get_requests()) == 1
    assert chat.llm.inference_count == 0
    assert "private checked input" not in caplog.text
    assert "private response body" not in caplog.text


@pytest.mark.parametrize("failure", ["timeout", "connection", "status", "malformed"])
@pytest.mark.asyncio
async def test_heuristics_outage_stops_generation(failure, httpx_mock, caplog):
    chat = TestChat(_config("heuristics"), llm_completions=["Unchecked input reached generation."])
    _queue_failure(httpx_mock, failure)
    response = await chat.app.generate_async(messages=[{"role": "user", "content": "private checked input"}])
    assert response["content"] == "I'm sorry, an internal error has occurred."
    assert chat.llm.inference_count == 0
    assert "private checked input" not in caplog.text
    assert "private response body" not in caplog.text


@pytest.mark.parametrize("backend", ["server", "nim", "local"])
@pytest.mark.asyncio
async def test_outage_is_not_cached_and_recovery_runs_detector(backend, monkeypatch):
    cache = LFUCache(maxsize=10)
    task_manager = LLMTaskManager(_config(backend))
    if backend == "local":
        detector = mock.Mock(side_effect=[RuntimeError("unavailable"), {"jailbreak": False}])
        monkeypatch.setattr(checks, "check_jailbreak", detector)
        client = RecordingHTTPClient()
        error_type = RuntimeError
    else:
        client = RecordingHTTPClient(
            [
                HTTPConnectionError("unavailable"),
                HTTPResponse(status_code=200, headers={}, content=b'{"jailbreak": false}'),
            ]
        )
        error_type = JailbreakDetectionError
    with pytest.raises(error_type):
        await jailbreak_detection_model(
            task_manager, user_message="hello", model_caches={"jailbreak_detection": cache}, http_client=client
        )
    outcome = await jailbreak_detection_model(
        task_manager, user_message="hello", model_caches={"jailbreak_detection": cache}, http_client=client
    )
    assert not outcome.is_blocked
    assert llm_call_info_var.get().from_cache is False
    if backend == "local":
        assert detector.call_count == 2
    else:
        assert len(client.requests) == 2


@pytest.mark.parametrize("engine", ["colang1", "iorails"])
@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
@pytest.mark.asyncio
async def test_local_detector_failure_stops_generation(engine, error_type, monkeypatch):
    monkeypatch.setattr(checks, "check_jailbreak", mock.Mock(side_effect=error_type("unavailable")))
    monkeypatch.setattr("nemoguardrails.guardrails.compiled_rail._is_installed", lambda _: True)
    config = _config("local")
    messages = [{"role": "user", "content": "hello"}]
    if engine == "iorails":
        async with IORails(config) as rails:
            main_call = mock.AsyncMock(side_effect=AssertionError("Unchecked input reached generation."))
            monkeypatch.setattr(rails.engine_registry, "model_call", main_call)
            response = await rails.generate_async(messages=messages)
            main_call.assert_not_awaited()
    else:
        chat = TestChat(config, llm_completions=["Unchecked input reached generation."])
        response = await chat.app.generate_async(messages=messages)
        assert chat.llm.inference_count == 0
    assert response["content"] == "I'm sorry, an internal error has occurred."
