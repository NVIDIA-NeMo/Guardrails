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

"""Shared calls isolate state while retaining complete streaming responses."""

import asyncio

import pytest

from nemoguardrails.context import (
    explain_info_var,
    llm_call_info_var,
    llm_response_metadata_var,
    llm_stats_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.llm.call import llm_call
from nemoguardrails.logging.explain import ExplainInfo, LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.testing.fake_model import FakeLLMModel
from nemoguardrails.types import LLMResponse, UsageInfo


@pytest.fixture(autouse=True)
def invocation_context():
    values = {
        explain_info_var: ExplainInfo(),
        llm_call_info_var: None,
        llm_response_metadata_var: None,
        llm_stats_var: LLMStats(),
        reasoning_trace_var: None,
        tool_calls_var: None,
        processing_log_var: [],
    }
    tokens = [(variable, variable.set(value)) for variable, value in values.items()]
    yield
    for variable, token in reversed(tokens):
        variable.reset(token)


@pytest.mark.asyncio
async def test_sequential_calls_have_independent_records_without_stale_response_fields():
    llm_call_info_var.set(LLMCallInfo(task="check"))
    model = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="first",
                request_id="request-1",
                usage=UsageInfo(input_tokens=2, output_tokens=1, total_tokens=3),
            ),
            LLMResponse(content="second"),
        ]
    )

    await llm_call(model, "first prompt")
    first = llm_call_info_var.get()
    await llm_call(model, "second prompt")
    second = llm_call_info_var.get()

    assert first is not second
    assert first.id != second.id
    assert first.completion == "first"
    assert first.request_id == "request-1"
    assert second.completion == "second"
    assert second.request_id is None
    assert second.total_tokens == 0
    assert first.task == second.task == "check"
    recorded = [entry["data"] for entry in processing_log_var.get()]
    assert [info.completion for info in recorded] == ["first", "second"]
    assert recorded[0] is not recorded[1]
    assert [info.completion for info in explain_info_var.get().llm_calls] == ["first", "second"]


@pytest.mark.asyncio
async def test_nested_calls_restore_outer_record_and_response_context(monkeypatch):
    inner = FakeLLMModel(
        llm_responses=[LLMResponse(content="inner", reasoning="inner reasoning", usage=UsageInfo(2, 1, 3))]
    )
    outer = FakeLLMModel(responses=[])

    async def generate(prompt, *, stop=None, **kwargs):
        info = llm_call_info_var.get()
        reasoning_trace_var.set("pending outer reasoning")
        tool_calls_var.set([{"id": "pending outer tool"}])
        llm_response_metadata_var.set({"pending_outer": True})
        await llm_call(inner, "inner prompt")
        assert llm_call_info_var.get() is info
        assert reasoning_trace_var.get() == "pending outer reasoning"
        assert tool_calls_var.get() == [{"id": "pending outer tool"}]
        assert llm_response_metadata_var.get() == {"pending_outer": True}
        return LLMResponse(content="outer", usage=UsageInfo(4, 2, 6))

    monkeypatch.setattr(outer, "generate_async", generate)
    await llm_call(outer, "outer prompt")

    infos = [entry["data"] for entry in processing_log_var.get()]
    assert [info.completion for info in infos] == ["inner", "outer"]
    assert infos[0].id != infos[1].id
    assert [info.total_tokens for info in infos] == [3, 6]
    assert llm_stats_var.get().get_stat("total_tokens") == 9
    assert llm_stats_var.get().get_stat("total_calls") == 2
    assert llm_call_info_var.get().completion == "outer"
    assert [info.completion for info in explain_info_var.get().llm_calls] == ["outer", "inner"]


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_mutate_an_inherited_record(monkeypatch):
    inherited = LLMCallInfo(task="shared caller")
    llm_call_info_var.set(inherited)
    ready = asyncio.Event()
    arrived = 0

    async def run(label):
        model = FakeLLMModel(responses=[])

        async def generate(prompt, *, stop=None, **kwargs):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await ready.wait()
            return LLMResponse(content=label, request_id=label)

        monkeypatch.setattr(model, "generate_async", generate)
        await llm_call(model, label)
        return llm_call_info_var.get()

    first, second = await asyncio.gather(run("first"), run("second"))

    assert first is not second
    assert first.id != second.id
    assert (first.completion, second.completion) == ("first", "second")
    assert first.task == second.task == "shared caller"
    assert inherited.id is None
    assert inherited.completion is None
    assert llm_call_info_var.get() is inherited
    assert len(processing_log_var.get()) == 2


@pytest.mark.asyncio
async def test_compiled_action_collects_each_invocation_as_a_distinct_record(monkeypatch):
    from nemoguardrails.actions.rail_outcome import RailOutcome
    from nemoguardrails.guardrails.compiled_rail import RailDependencies, compile_rail
    from nemoguardrails.llm.taskmanager import LLMTaskManager
    from nemoguardrails.manifests import RailDirection
    from nemoguardrails.rails.llm.config import Model, RailsConfig

    config = RailsConfig(models=[Model(type="content_safety", engine="fake", model="safety")])
    model = FakeLLMModel(llm_responses=[LLMResponse(content="first"), LLMResponse(content="second")])
    deps = RailDependencies(llms={"content_safety": model}, llm_task_manager=LLMTaskManager(config), config=config)

    async def action(**kwargs):
        llm_call_info_var.set(LLMCallInfo(task="two model checks"))
        await llm_call(model, "first prompt")
        await llm_call(model, "second prompt")
        return RailOutcome.allow()

    monkeypatch.setattr("nemoguardrails.library.content_safety.actions.content_safety_check_input", action)
    execution = await compile_rail(
        "content safety check input $model=content_safety", RailDirection.INPUT, deps
    ).execute([{"role": "user", "content": "hello"}])

    assert [info.completion for info in execution.llm_calls] == ["first", "second"]
    assert execution.llm_calls[0].id != execution.llm_calls[1].id
    assert all(info.task == "two model checks" for info in execution.llm_calls)
    assert processing_log_var.get() == []


@pytest.mark.asyncio
async def test_logging_failure_restores_context_without_masking_provider_failure():
    class RejectingLog(list):
        def append(self, entry):
            raise RuntimeError("log sink failed")

    seed = LLMCallInfo(task="caller")
    llm_call_info_var.set(seed)
    reasoning_trace_var.set("caller reasoning")
    processing_log_var.set(RejectingLog())
    provider_error = ValueError("provider failed")

    with pytest.raises(LLMCallException) as caught:
        await llm_call(FakeLLMModel(llm_exception=provider_error), "prompt")

    assert caught.value.__cause__ is provider_error
    assert llm_call_info_var.get() is seed
    assert reasoning_trace_var.get() == "caller reasoning"
    processing_log_var.set([])
    await llm_call(FakeLLMModel(responses=["later"]), "later prompt")
    assert llm_call_info_var.get().completion == "later"
