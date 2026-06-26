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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemoguardrails.actions.v2_x import generation as v2_generation
from nemoguardrails.colang.v2_x.lang.colang_ast import Flow, SpecOp
from nemoguardrails.colang.v2_x.runtime.errors import LlmResponseError
from nemoguardrails.colang.v2_x.runtime.flows import InternalEvents
from nemoguardrails.context import generation_options_var, raw_llm_request
from nemoguardrails.rails.llm.options import GenerationOptions


class FakeIndex:
    def __init__(self, results=None):
        self.results = results or []
        self.items = []
        self.built = False
        self.search_calls = []

    async def add_items(self, items):
        self.items.extend(items)

    async def build(self):
        self.built = True

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.results


class FakeTaskManager:
    def __init__(self, prompt="prompt"):
        self.prompt = prompt
        self.rendered_prompts = []
        self.parsed_outputs = []
        self.rendered_strings = []

    def render_task_prompt(self, task, events, context):
        self.rendered_prompts.append((task, events, context))
        return self.prompt

    def get_stop_tokens(self, task):
        return ["STOP"]

    def parse_task_output(self, task, output):
        self.parsed_outputs.append((task, output))
        return output

    def _render_string(self, text, context, events):
        self.rendered_strings.append((text, context, events))
        return text.format(**context)


class RuntimeFlowConfig:
    def __init__(self, flow_id, source_code="", elements=None, decorators=None, has_user_intent=False):
        self.id = flow_id
        self.source_code = source_code
        self.elements = elements or []
        self.decorators = decorators or {}
        self.has_user_intent = has_user_intent

    def has_meta_tag(self, tag):
        return tag == "user_intent" and self.has_user_intent


def make_actions(flows=None, prompt="prompt"):
    actions = v2_generation.LLMGenerationActionsV2dotx.__new__(v2_generation.LLMGenerationActionsV2dotx)
    user_messages = SimpleNamespace(
        embeddings_only=False,
        embeddings_only_similarity_threshold=0.7,
        embeddings_only_fallback_intent=None,
    )
    actions.config = SimpleNamespace(
        core=SimpleNamespace(embedding_search_provider=None),
        flows=flows or [],
        rails=SimpleNamespace(dialog=SimpleNamespace(user_messages=user_messages)),
        lowest_temperature=0.0,
    )
    actions.llm = object()
    actions.llm_task_manager = FakeTaskManager(prompt=prompt)
    actions.user_message_index = None
    actions.flows_index = None
    actions.instruction_flows_index = None
    actions._init_lock = asyncio.Lock()
    actions._last_docstring = "Fallback for {name}"
    actions.get_embedding_search_provider_instance = lambda provider: FakeIndex()
    return actions


def patch_llm_call(monkeypatch, *outputs):
    calls = []
    pending = list(outputs)

    async def fake_llm_call(llm, prompt, **kwargs):
        calls.append((llm, prompt, kwargs))
        return SimpleNamespace(content=pending.pop(0))

    monkeypatch.setattr(v2_generation, "llm_call", fake_llm_call)
    return calls


@pytest.mark.asyncio
async def test_init_flows_index_builds_instruction_index():
    flows = [
        Flow(name="included", source_code="flow included\n  # instruction\n  bot say hi"),
        Flow(name="regular", source_code="flow regular\n  bot say hi"),
        Flow(name="excluded", source_code="flow excluded\n  bot say no", file_info={"exclude_from_llm": True}),
    ]
    actions = make_actions(flows=flows)
    indexes = []

    def make_index(provider):
        index = FakeIndex()
        indexes.append(index)
        return index

    actions.get_embedding_search_provider_instance = make_index

    await actions._init_flows_index()

    assert [item.text for item in indexes[0].items] == [
        "flow included\n  # instruction\n  bot say hi",
        "flow regular\n  bot say hi",
    ]
    assert [item.text for item in indexes[1].items] == ["flow included\n  # instruction\n  bot say hi"]
    assert actions.flows_index is indexes[0]
    assert actions.instruction_flows_index is indexes[1]


@pytest.mark.asyncio
async def test_collect_user_intent_examples_from_index_and_active_match(monkeypatch):
    actions = make_actions()
    actions.user_message_index = FakeIndex(
        [
            SimpleNamespace(text="hello", meta={"intent": "user greet"}),
            SimpleNamespace(text="help", meta={"intent": "user ask help"}),
        ]
    )
    doc_element = {
        "_type": "doc_string_stmt",
        "elements": [{"elements": [{"elements": ['"""documented utterance"""']}]}],
    }
    active_config = RuntimeFlowConfig("documented", elements=[{}, doc_element], has_user_intent=True)
    state = SimpleNamespace(
        flow_states={"head": object(), "expected": object()},
        flow_configs={"documented": active_config},
        flow_id_states={"documented": [SimpleNamespace(context={})]},
    )
    heads = [SimpleNamespace(flow_state_uid="head"), SimpleNamespace(flow_state_uid="expected")]

    monkeypatch.setattr(v2_generation, "find_all_active_event_matchers", lambda state, event=None: heads)
    monkeypatch.setattr(v2_generation, "get_element_from_head", lambda state, head: SpecOp(op="match"))

    def get_event_from_element(state, flow_state, element):
        flow_id = "documented" if flow_state is state.flow_states["head"] else "expected only"
        return SimpleNamespace(name=InternalEvents.FLOW_FINISHED, arguments={"flow_id": flow_id})

    monkeypatch.setattr(v2_generation, "get_event_from_element", get_event_from_element)

    intents, examples, is_embedding_only = await actions._collect_user_intent_and_examples(state, "hi", 3)

    assert intents == ["user ask help", "user greet", "expected only"]
    assert 'user action: user said "help"' in examples
    assert "user action: <documented utterance>" in examples
    assert "user intent: expected only" in examples
    assert is_embedding_only is False


@pytest.mark.asyncio
async def test_generate_user_intent_embedding_only_and_llm(monkeypatch):
    actions = make_actions()
    state = SimpleNamespace(context={"topic": "support"})
    actions._collect_user_intent_and_examples = AsyncMock(return_value=(["user cached intent"], "", True))

    assert await actions.generate_user_intent(state, [], "hello") == "user cached intent"

    actions._collect_user_intent_and_examples = AsyncMock(return_value=(["user ask"], "examples", False))
    calls = patch_llm_call(monkeypatch, "user intent: ask about account")

    assert await actions.generate_user_intent(state, [{"type": "event"}], "help") == "ask about account"
    assert calls[0][2]["llm_params"] == {"temperature": 0.0}
    assert actions.llm_task_manager.rendered_prompts[-1][2]["potential_user_intents"] == "user ask"


@pytest.mark.asyncio
async def test_generate_user_intent_and_bot_action_success_and_error(monkeypatch):
    actions = make_actions()
    state = SimpleNamespace(context={})
    actions._collect_user_intent_and_examples = AsyncMock(return_value=(["user ask"], "", False))
    patch_llm_call(
        monkeypatch,
        'user intent: ask help\nbot intent: provide help\nbot action: bot say "Here"',
        "assistant response only",
    )

    result = await actions.generate_user_intent_and_bot_action(state, [], "help")

    assert result == {
        "user_intent": "ask help",
        "bot_intent": "provide help",
        "bot_action": 'bot say "Here"',
    }
    monkeypatch.setattr(v2_generation, "get_first_bot_action", lambda lines: None)
    with pytest.raises(LlmResponseError):
        await actions.generate_user_intent_and_bot_action(state, [], "help")


@pytest.mark.asyncio
async def test_passthrough_llm_action_branches(monkeypatch):
    actions = make_actions()
    events = [{"type": "UtteranceUserActionFinished", "final_transcript": "rewritten"}]

    with pytest.raises(RuntimeError, match="No LLM provided"):
        await actions.passthrough_llm_action("message", SimpleNamespace(), events)

    with pytest.raises(RuntimeError, match="couldn't find last user utterance"):
        await actions.passthrough_llm_action("message", SimpleNamespace(), [], llm=object())

    calls = patch_llm_call(monkeypatch, "parsed response")
    raw_token = raw_llm_request.set([{"role": "user", "content": "original"}])
    options_token = generation_options_var.set(GenerationOptions(llm_params={"top_p": 0.2}))
    try:
        result = await actions.passthrough_llm_action("message", SimpleNamespace(), events, llm=object())
    finally:
        raw_llm_request.reset(raw_token)
        generation_options_var.reset(options_token)

    assert result == "parsed response"
    assert calls[0][1] == "message"
    assert calls[0][2]["llm_params"] == {"top_p": 0.2}


@pytest.mark.asyncio
async def test_check_flow_helpers(monkeypatch):
    actions = make_actions()
    state = SimpleNamespace(flow_id_states={"known": []}, flow_configs={"defined": object()})

    assert await actions.check_if_flow_exists(state, "known") is True
    assert await actions.check_if_flow_exists(state, "missing") is False
    assert await actions.check_if_flow_defined(state, "defined") is True
    assert await actions.check_if_flow_defined(state, "missing") is False

    captured = []
    monkeypatch.setattr(
        v2_generation, "find_all_active_event_matchers", lambda state, event: captured.append(event) or [object()]
    )

    assert await actions.check_for_active_flow_finished_match(state, InternalEvents.FLOW_FINISHED, flow_id="x") is True
    assert await actions.check_for_active_flow_finished_match(state, "SomeActionFinished", uid="x") is True
    assert await actions.check_for_active_flow_finished_match(state, "ExternalEvent", uid="x") is True
    assert [event.name for event in captured] == [InternalEvents.FLOW_FINISHED, "SomeActionFinished", "ExternalEvent"]


@pytest.mark.asyncio
async def test_generate_flow_from_instructions_success_and_fallback(monkeypatch):
    actions = make_actions()
    actions.instruction_flows_index = FakeIndex([SimpleNamespace(meta={"flow": "flow example\n  bot say hi"})])
    state = SimpleNamespace(context={"name": "Ada"})
    monkeypatch.setattr(v2_generation, "new_uuid", lambda: "abcd1234")
    patch_llm_call(monkeypatch, '\n  bot say "ok"', "bot say missing indent")

    generated = await actions.generate_flow_from_instructions(state, "say ok", [])
    fallback = await actions.generate_flow_from_instructions(state, "say ok", [])

    assert generated == {"name": "dynamic_abcd", "body": 'flow dynamic_abcd\n  bot say "ok"'}
    assert fallback["name"] == "bot inform LLM issue"
    assert "GenerateFlowFromInstructionsAction" in fallback["body"]


@pytest.mark.asyncio
async def test_generate_flow_from_name_success_variants(monkeypatch):
    actions = make_actions()
    actions.flows_index = FakeIndex()
    actions.instruction_flows_index = FakeIndex([SimpleNamespace(meta={"flow": "flow sample\n  bot say hi"})])
    state = SimpleNamespace(context={})
    patch_llm_call(monkeypatch, 'flow generated\n  bot say "hello"', 'bot say "hello"')

    assert await actions.generate_flow_from_name(state, "generated", []) == 'flow generated\n  bot say "hello"'
    assert await actions.generate_flow_from_name(state, "fallback", []) == 'flow fallback\n  bot say "hello"'

    actions.flows_index = None
    with pytest.raises(RuntimeError, match="No flows index"):
        await actions.generate_flow_from_name(state, "missing", [])


@pytest.mark.asyncio
async def test_generate_flow_continuation_success_fallback_and_error(monkeypatch):
    actions = make_actions()
    actions.flows_index = FakeIndex([SimpleNamespace(meta={"flow": "flow example\n  bot say hi # remove"})])
    actions.instruction_flows_index = FakeIndex()
    state = SimpleNamespace(context={})
    monkeypatch.setattr(v2_generation, "colang", lambda events: "user said hi\nbot line")
    monkeypatch.setattr(v2_generation, "new_uuid", lambda: "12345678abcd")
    patch_llm_call(
        monkeypatch,
        'bot intent: provide answer\nbot action: bot say "Answer"',
        "\n",
        "assistant response only",
    )

    generated = await actions.generate_flow_continuation(state, [], temperature=0.3)
    fallback = await actions.generate_flow_continuation(state, [])

    assert generated["name"] == "_dynamic_12345678 provide answer"
    assert generated["parameters"] == []
    assert 'bot say "Answer"' in generated["body"]
    assert fallback["name"] == "bot inform LLM issue"
    monkeypatch.setattr(v2_generation, "get_first_bot_action", lambda lines: None)
    with pytest.raises(LlmResponseError):
        await actions.generate_flow_continuation(state, [])


@pytest.mark.asyncio
async def test_create_flow_escapes_name_and_applies_decorators(monkeypatch):
    actions = make_actions()
    monkeypatch.setattr(v2_generation, "new_uuid", lambda: "abcdef123456")

    result = await actions.create_flow([], "bot greet", 'bot say "hello"', decorators="@active")

    assert result == {
        "name": "_dynamic_abcdef12 bot greet",
        "parameters": [],
        "body": '@active\nflow _dynamic_abcdef12 bot greet\n  bot say "hello"',
    }


@pytest.mark.asyncio
async def test_generate_value_parses_prompt_variants_and_errors(monkeypatch):
    actions = make_actions(prompt="value =")
    actions.flows_index = FakeIndex(
        [
            SimpleNamespace(text="flow example\n  $value = 1"),
            SimpleNamespace(text="flow ignored\n  GenerateValueAction()"),
        ]
    )
    state = SimpleNamespace(context={})
    patch_llm_call(monkeypatch, "value = {'ok': True};", "$answer = ['a'];", "not python")

    assert await actions.generate_value(state, "make dict", [], var_name="value") == {"ok": True}

    actions.llm_task_manager.prompt = [{"role": "user", "content": "$answer = "}]
    assert await actions.generate_value(state, "make list", [], var_name="answer") == ["a"]

    with pytest.raises(Exception, match="Invalid LLM response"):
        await actions.generate_value(state, "make bad", [], var_name="bad")


@pytest.mark.asyncio
async def test_generate_flow_success_and_error_paths(monkeypatch):
    actions = make_actions()
    trigger_config = RuntimeFlowConfig(
        "trigger",
        source_code='flow trigger\n  """Help {name} using {tool_names}.\n{tools}"""\n  ...',
    )
    tool_config = RuntimeFlowConfig(
        "lookup",
        source_code='@meta(tool=True)\nflow lookup $query\n  """Lookup docs"""\n  await LookupAction()',
        decorators={"meta": {"tool": True}},
    )
    state = SimpleNamespace(
        context={"name": "Ada"},
        flow_configs={"trigger": trigger_config, "lookup": tool_config},
        flow_id_states={"trigger": [SimpleNamespace(context={"name": "Ada Lovelace"})]},
    )
    monkeypatch.setattr(v2_generation, "new_uuid", lambda: "fedcba987654")
    patch_llm_call(monkeypatch, 'codeblock\nbot say "hello"\n  user said something')

    result = await actions.generate_flow(state, [], flow_id="trigger")

    assert result["name"] == "_dynamic_fedcba98"
    assert result["parameters"] == []
    assert result["body"] == 'flow _dynamic_fedcba98\n  bot say "hello"\n  wait user input\n  ...'
    assert "`lookup`" in actions.llm_task_manager.rendered_strings[-1][1]["tool_names"]

    with pytest.raises(RuntimeError, match="No flow_id"):
        await actions.generate_flow(state, [])

    state.flow_configs["empty"] = RuntimeFlowConfig("empty", source_code="")
    with pytest.raises(RuntimeError, match="No source_code"):
        await actions.generate_flow(state, [], flow_id="empty")
