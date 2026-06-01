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

from types import SimpleNamespace

import pytest

from nemoguardrails.rails.llm.generation import generation_tracing
from nemoguardrails.rails.llm.generation.generation_tracing import (
    export_generation_trace,
    prepare_generation_tracing,
    restore_generation_trace_log,
)
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationLogOptions,
    GenerationOptions,
    GenerationResponse,
)


def test_generation_tracing_exports():
    assert generation_tracing.__all__ == [
        "GenerationTracingState",
        "export_generation_trace",
        "prepare_generation_tracing",
        "restore_generation_trace_log",
    ]


def test_prepare_generation_tracing_returns_original_options_when_disabled():
    options = GenerationOptions(output_vars=["answer"])

    state = prepare_generation_tracing(
        tracing_enabled=False,
        gen_options=options,
    )

    assert state.gen_options is options
    assert state.original_log_options is None


def test_prepare_generation_tracing_copies_options_and_forces_trace_logs():
    options = GenerationOptions(log=GenerationLogOptions(colang_history=True))

    state = prepare_generation_tracing(
        tracing_enabled=True,
        gen_options=options,
    )

    assert state.gen_options is not options
    assert state.gen_options is not None
    assert state.gen_options.log.activated_rails is True
    assert state.gen_options.log.llm_calls is True
    assert state.gen_options.log.internal_events is True
    assert state.gen_options.log.colang_history is True
    assert state.original_log_options == GenerationLogOptions(colang_history=True)
    assert options.log.activated_rails is False
    assert options.log.llm_calls is False
    assert options.log.internal_events is False


def test_prepare_generation_tracing_creates_options_when_absent():
    state = prepare_generation_tracing(
        tracing_enabled=True,
        gen_options=None,
    )

    assert state.gen_options is not None
    assert state.gen_options.log.activated_rails is True
    assert state.gen_options.log.llm_calls is True
    assert state.gen_options.log.internal_events is True
    assert state.original_log_options == GenerationLogOptions()


def test_restore_generation_trace_log_removes_trace_added_log_when_original_requested_none():
    response = GenerationResponse(
        response="hello",
        log=GenerationLog(
            activated_rails=[ActivatedRail(type="input", name="rail")],
            internal_events=[{"type": "event"}],
            llm_calls=[],
        ),
    )

    restore_generation_trace_log(
        response=response,
        original_log_options=GenerationLogOptions(),
    )

    assert response.log is None


def test_restore_generation_trace_log_keeps_only_originally_requested_log_fields():
    response = GenerationResponse(
        response="hello",
        log=GenerationLog(
            activated_rails=[ActivatedRail(type="input", name="rail")],
            internal_events=[{"type": "event"}],
            llm_calls=[],
        ),
    )

    restore_generation_trace_log(
        response=response,
        original_log_options=GenerationLogOptions(internal_events=True),
    )

    assert response.log is not None
    assert response.log.internal_events == [{"type": "event"}]
    assert response.log.activated_rails == []
    assert response.log.llm_calls == []


@pytest.mark.asyncio
async def test_export_generation_trace_uses_tracing_config(monkeypatch):
    calls = {}

    class FakeTracer:
        def __init__(
            self,
            *,
            input,
            response,
            adapters,
            span_format,
            enable_content_capture,
        ):
            calls["init"] = {
                "input": input,
                "response": response,
                "adapters": adapters,
                "span_format": span_format,
                "enable_content_capture": enable_content_capture,
            }

        async def export_async(self):
            calls["exported"] = True

    monkeypatch.setattr("nemoguardrails.tracing.Tracer", FakeTracer)

    response = GenerationResponse(response="hello", log=GenerationLog())
    adapters = [object()]
    messages = [{"role": "user", "content": "hi"}]
    await export_generation_trace(
        tracing_config=SimpleNamespace(
            span_format="legacy",
            enable_content_capture=True,
        ),
        log_adapters=adapters,
        messages=messages,
        response=response,
    )

    assert calls == {
        "init": {
            "input": messages,
            "response": response,
            "adapters": adapters,
            "span_format": "legacy",
            "enable_content_capture": True,
        },
        "exported": True,
    }
