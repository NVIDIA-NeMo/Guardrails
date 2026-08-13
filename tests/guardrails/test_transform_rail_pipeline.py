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

"""A masking rail among judging rails, driven end to end through IORails.

The single-rail tests prove a rewrite is applied. These prove the pipeline around it, in both
directions and under both concurrency settings:

- **input** — four rails, the masking one written *last*: it still runs first, every rail behind
  it waits and reads the masked text, and so does the main model;
- **output** — three rails, same shape: the masking one runs first and the caller reads the
  masked response;
- **parallel** — the same two configs asking for concurrent rails, which is refused with a
  warning and run in order instead, because concurrent rails would each read the text as it
  arrived and the mask would reach none of them;
- **streaming** — masking applied to the batch that ships, judged after masking, and an input
  rewrite reaching the output rails, which is where a stale conversation would leak it.

Nothing is stubbed at the rail boundary -- the shipped actions run, against canned model and
HTTP replies -- so a rail that stopped reading its conversation variable, or an ordering rule
that stopped reordering, fails here rather than in a config.
"""

import copy
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from nemoguardrails.guardrails.guardrails_types import RailDirection
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse, LLMResponseChunk
from tests.guardrails.async_helpers import JAILBREAK_NIM_URL, started_iorails
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

GLINER_URL = "http://gliner.example/v1/extract"

# One PII entity, chosen so no other rail has an opinion about it: a name is not unsafe content,
# not off topic, and not a jailbreak attempt. What each rail reads is then the only variable.
PERSON = "Ada Lovelace"
USER_INPUT = f"my name is {PERSON} and I need help with my order"
MASKED_INPUT = "my name is [PERSON] and I need help with my order"
MAIN_OUTPUT = "Happy to help with your order."

SAFE_VERDICT = json.dumps({"User Safety": "safe"})
UNSAFE_VERDICT = json.dumps({"User Safety": "unsafe", "Safety Categories": "S1: Violence"})
ON_TOPIC_VERDICT = "on-topic"

# The masking rail is configured *after* the three that only judge, so the run order asserted
# below can only come from the transform-first rule rather than from the config.
INPUT_FLOWS = [
    "content safety check input $model=content_safety",
    "topic safety check input $model=topic_control",
    "jailbreak detection model",
    "gliner mask pii on input",
]

CONTENT_SAFETY_RAIL = "content_safety"
TOPIC_CONTROL_RAIL = "topic_control"
GLINER_RAIL = "gliner"
JAILBREAK_RAIL = "jailbreak"


def _pipeline_config() -> dict:
    """The four-rail config both the allow and the block case are built from."""
    config = copy.deepcopy(NEMOGUARDS_CONFIG)
    config["rails"]["input"]["flows"] = list(INPUT_FLOWS)
    config["rails"]["output"] = {"flows": []}
    config["rails"]["config"]["gliner"] = {
        "server_endpoint": GLINER_URL,
        "input": {"entities": ["person"]},
    }
    return config


def _gliner_entities(text: str) -> list[dict]:
    """The detection GLiNER returns for *text*, positioned where the name actually is.

    Nothing when the name is absent, which is what a streamed batch carrying none of it gets.
    """
    start = text.find(PERSON)
    if start == -1:
        return []
    return [
        {
            "value": PERSON,
            "suggested_label": "person",
            "start_position": start,
            "end_position": start + len(PERSON),
        }
    ]


class RailCallLog:
    """An ordered record of which rail called out, and with what text.

    One log across both seams -- the model engines and the HTTP endpoints -- because the
    question is what order the *rails* ran in, which neither seam can answer alone.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def record(self, rail: str, text: str) -> None:
        self.calls.append((rail, text))

    @property
    def order(self) -> list[str]:
        return [rail for rail, _ in self.calls]

    def text_seen_by(self, rail: str) -> str:
        """The text *rail* was given, failing loudly when it never ran."""
        for name, text in self.calls:
            if name == rail:
                return text
        raise AssertionError(f"{rail!r} never ran; the log holds {self.order}")


def _model_double(log: RailCallLog, model_type: str, verdict: str) -> AsyncMock:
    """Answer one model-backed rail with *verdict*, recording the prompt it was sent.

    One double per model type rather than one shared double told apart by prompt text: each
    rail reaches its own engine, so the engine already knows which rail is calling. The whole
    prompt is recorded, not its first message, because the rails place the text under check at
    different positions and which text a rail reads is what is under test.
    """

    async def _chat_completion(messages, **kwargs):
        log.record(model_type, "\n".join(str(message.get("content", "")) for message in messages))
        return LLMResponse(content=verdict)

    return AsyncMock(side_effect=_chat_completion)


def _register_http_doubles(httpx_mock, log: RailCallLog, *, gliner_runs: bool = True) -> None:
    """Answer the two HTTP-backed rails, recording the text each was sent.

    *gliner_runs* is False for a config with no masking rail, where the callback would otherwise
    be registered and never requested.
    """

    def _gliner(request):
        # The entities are positioned against the text this call actually carried, so a rail
        # handed the wrong text masks nothing rather than masking at a stale offset.
        text = json.loads(request.content)["text"]
        log.record(GLINER_RAIL, text)
        return httpx.Response(200, json={"entities": _gliner_entities(text)})

    def _jailbreak(request):
        log.record(JAILBREAK_RAIL, json.loads(request.content)["input"])
        return httpx.Response(200, json={"jailbreak": False, "score": 0.01})

    # Where the masking rail is configured, leaving its callback required makes "gliner was
    # called" an assertion the fixture enforces. Jailbreak runs last and is skipped whenever a
    # rail ahead of it blocks, which is a case below, so its callback is always optional.
    httpx_mock.add_callback(_gliner, url=GLINER_URL, is_optional=not gliner_runs)
    httpx_mock.add_callback(_jailbreak, url=JAILBREAK_NIM_URL, is_optional=True)


@pytest.fixture
def call_log() -> RailCallLog:
    return RailCallLog()


@pytest_asyncio.fixture
async def pipeline_iorails():
    """IORails with the four input rails, started and stopped around the test."""
    async with started_iorails(_pipeline_config()) as engine:
        yield engine


def _wire_answering(engine: IORails, log: RailCallLog, httpx_mock, verdicts: dict[str, str], answer: str) -> AsyncMock:
    """Wire the rails and have the main model answer *answer*, which the output rails then judge."""
    main = _wire(engine, log, httpx_mock, verdicts)
    main.return_value = LLMResponse(content=answer)
    return main


def _wire(engine: IORails, log: RailCallLog, httpx_mock, verdicts: dict[str, str]) -> AsyncMock:
    """Attach a double per model-backed rail plus the HTTP ones, and return the main-model double."""
    for model_type, rail_engine in engine.engine_registry.llms.items():
        if model_type in verdicts:
            rail_engine.chat_completion = _model_double(log, model_type, verdicts[model_type])
    _register_http_doubles(httpx_mock, log)

    main = AsyncMock(return_value=LLMResponse(content=MAIN_OUTPUT))
    engine.engine_registry._engines["main"].chat_completion = main
    return main


ALL_ALLOW = {CONTENT_SAFETY_RAIL: SAFE_VERDICT, TOPIC_CONTROL_RAIL: ON_TOPIC_VERDICT}
CONTENT_SAFETY_BLOCKS = {CONTENT_SAFETY_RAIL: UNSAFE_VERDICT, TOPIC_CONTROL_RAIL: ON_TOPIC_VERDICT}


@pytest.mark.asyncio
class TestMaskingRailAheadOfJudgingRails:
    """Four input rails, the masking one configured last: it runs first and the rest read its work."""

    async def test_the_masking_rail_runs_before_every_rail_that_only_judges(
        self, pipeline_iorails, call_log, httpx_mock
    ):
        """Configured last, run first -- so the order comes from the rule, not from the config."""
        _wire(pipeline_iorails, call_log, httpx_mock, ALL_ALLOW)

        await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.order[0] == GLINER_RAIL
        assert set(call_log.order) == {GLINER_RAIL, CONTENT_SAFETY_RAIL, TOPIC_CONTROL_RAIL, JAILBREAK_RAIL}

    async def test_the_masking_rail_reads_the_message_as_it_arrived(self, pipeline_iorails, call_log, httpx_mock):
        """Nothing precedes it, so it is the one rail that sees the unmasked text."""
        _wire(pipeline_iorails, call_log, httpx_mock, ALL_ALLOW)

        await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.text_seen_by(GLINER_RAIL) == USER_INPUT

    @pytest.mark.parametrize("rail", [CONTENT_SAFETY_RAIL, TOPIC_CONTROL_RAIL, JAILBREAK_RAIL])
    async def test_every_rail_behind_it_waits_and_reads_the_masked_text(
        self, pipeline_iorails, call_log, httpx_mock, rail
    ):
        """None of the three could hold masked text unless it ran after the mask was applied.

        This is the sequencing assertion as well as the threading one: a rail that started
        alongside the masking rail would have been handed the text as it arrived.
        """
        _wire(pipeline_iorails, call_log, httpx_mock, ALL_ALLOW)

        await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        seen = call_log.text_seen_by(rail)
        assert MASKED_INPUT in seen
        assert PERSON not in seen

    async def test_the_main_model_reads_the_masked_text(self, pipeline_iorails, call_log, httpx_mock):
        """The point of masking on input: the name never reaches the model being guarded."""
        main = _wire(pipeline_iorails, call_log, httpx_mock, ALL_ALLOW)

        await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        sent = main.call_args.args[0]
        assert sent[-1]["content"] == MASKED_INPUT

    async def test_the_caller_gets_the_answer_and_keeps_their_own_message(self, pipeline_iorails, call_log, httpx_mock):
        """Masking is internal to the turn: the reply is the model's, and the caller's list is theirs."""
        _wire(pipeline_iorails, call_log, httpx_mock, ALL_ALLOW)
        messages = [{"role": "user", "content": USER_INPUT}]

        response = await pipeline_iorails.generate_async(messages=messages)

        assert response == {"role": "assistant", "content": MAIN_OUTPUT}
        assert messages == [{"role": "user", "content": USER_INPUT}]

    async def test_a_rail_behind_the_mask_can_still_block(self, pipeline_iorails, call_log, httpx_mock):
        """A rewrite ahead of a block does not rescue the request, and the block is the verdict."""
        main = _wire(pipeline_iorails, call_log, httpx_mock, CONTENT_SAFETY_BLOCKS)

        response = await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert response == {"role": "assistant", "content": REFUSAL_MESSAGE}
        main.assert_not_called()

    async def test_the_blocking_rail_judged_the_masked_text(self, pipeline_iorails, call_log, httpx_mock):
        """The rail that blocked read what the mask produced, so its verdict is about that text."""
        _wire(pipeline_iorails, call_log, httpx_mock, CONTENT_SAFETY_BLOCKS)

        await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.order[0] == GLINER_RAIL
        assert PERSON not in call_log.text_seen_by(CONTENT_SAFETY_RAIL)

    async def test_the_rails_behind_a_block_never_run(self, pipeline_iorails, call_log, httpx_mock):
        """Short-circuiting survives the reordering: after content safety blocks, nothing else runs."""
        _wire(pipeline_iorails, call_log, httpx_mock, CONTENT_SAFETY_BLOCKS)

        await pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.order == [GLINER_RAIL, CONTENT_SAFETY_RAIL]


def test_the_four_rail_config_routes_to_iorails():
    """A config mixing a masking rail with three judging ones is servable, so the rest is reachable."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        config = RailsConfig.from_content(config=_pipeline_config())

    assert IORails.unsupported_reason(config, llm=None) is None


SELF_CHECK_RAIL = "self_check_output"
SELF_CHECK_PROMPT = (
    "Should the following bot response be blocked?\n"
    "Bot response: {{ bot_response }}\n"
    'Answer "yes" to block and "no" to allow.'
)

MAIN_OUTPUT_WITH_PII = f"Sure thing, {PERSON} — your order ships tomorrow."
MASKED_MAIN_OUTPUT = "Sure thing, [PERSON] — your order ships tomorrow."

# Same shape as the input list: the masking rail is written last, so the run order asserted
# below can only come from the transform-first rule.
OUTPUT_FLOWS = [
    "self check output",
    "content safety check output $model=content_safety",
    "gliner mask pii on output",
]

SELF_CHECK_ALLOWS = "no"
SELF_CHECK_BLOCKS = "yes"
SAFE_OUTPUT_VERDICT = json.dumps({"User Safety": "safe", "Response Safety": "safe"})
UNSAFE_OUTPUT_VERDICT = json.dumps(
    {"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S17: Malware"}
)

OUTPUT_ALL_ALLOW = {SELF_CHECK_RAIL: SELF_CHECK_ALLOWS, CONTENT_SAFETY_RAIL: SAFE_OUTPUT_VERDICT}
OUTPUT_CONTENT_SAFETY_BLOCKS = {SELF_CHECK_RAIL: SELF_CHECK_ALLOWS, CONTENT_SAFETY_RAIL: UNSAFE_OUTPUT_VERDICT}


def _output_pipeline_config(*, parallel: bool = False) -> dict:
    """The three-rail output config, optionally asking for the rails to run concurrently."""
    config = copy.deepcopy(NEMOGUARDS_CONFIG)
    config["models"].append({"type": SELF_CHECK_RAIL, "engine": "nim", "model": "meta/llama-3.3-70b-instruct"})
    config["prompts"].append({"task": "self_check_output", "content": SELF_CHECK_PROMPT})
    config["rails"]["input"] = {"flows": []}
    config["rails"]["output"] = {"flows": list(OUTPUT_FLOWS), "parallel": parallel}
    config["rails"]["config"]["gliner"] = {
        "server_endpoint": GLINER_URL,
        "output": {"entities": ["person"]},
    }
    return config


def _input_pipeline_config(*, parallel: bool) -> dict:
    """The four-rail input config, asking for the rails to run concurrently."""
    config = _pipeline_config()
    config["rails"]["input"]["parallel"] = parallel
    return config


@pytest_asyncio.fixture
async def output_pipeline_iorails():
    """IORails with the three output rails, started and stopped around the test."""
    async with started_iorails(_output_pipeline_config()) as engine:
        yield engine


@pytest.mark.asyncio
class TestMaskingRailAheadOfJudgingOutputRails:
    """Three output rails, the masking one configured last: it runs first and the rest read its work."""

    async def test_the_masking_rail_runs_before_every_rail_that_only_judges(
        self, output_pipeline_iorails, call_log, httpx_mock
    ):
        """Configured last, run first -- the output direction schedules by the same rule."""
        _wire_answering(output_pipeline_iorails, call_log, httpx_mock, OUTPUT_ALL_ALLOW, MAIN_OUTPUT_WITH_PII)

        await output_pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.order == [GLINER_RAIL, SELF_CHECK_RAIL, CONTENT_SAFETY_RAIL]

    async def test_the_masking_rail_reads_the_response_as_generated(
        self, output_pipeline_iorails, call_log, httpx_mock
    ):
        """Nothing precedes it, so it is the one rail that sees the model's unmasked answer."""
        _wire_answering(output_pipeline_iorails, call_log, httpx_mock, OUTPUT_ALL_ALLOW, MAIN_OUTPUT_WITH_PII)

        await output_pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.text_seen_by(GLINER_RAIL) == MAIN_OUTPUT_WITH_PII

    @pytest.mark.parametrize("rail", [SELF_CHECK_RAIL, CONTENT_SAFETY_RAIL])
    async def test_every_rail_behind_it_waits_and_reads_the_masked_response(
        self, output_pipeline_iorails, call_log, httpx_mock, rail
    ):
        """Neither could hold the masked answer unless it ran after the mask was applied.

        The *response* is what an output mask rewrites; the user's own turn is untouched, and
        both these rails are handed it as conversation context. So the assertion is that the
        unmasked answer is gone, not that the name is gone from the prompt entirely.
        """
        _wire_answering(output_pipeline_iorails, call_log, httpx_mock, OUTPUT_ALL_ALLOW, MAIN_OUTPUT_WITH_PII)

        await output_pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        seen = call_log.text_seen_by(rail)
        assert MASKED_MAIN_OUTPUT in seen
        assert MAIN_OUTPUT_WITH_PII not in seen

    async def test_the_caller_reads_the_masked_response(self, output_pipeline_iorails, call_log, httpx_mock):
        """The point of masking on output: the name never reaches whoever asked."""
        _wire_answering(output_pipeline_iorails, call_log, httpx_mock, OUTPUT_ALL_ALLOW, MAIN_OUTPUT_WITH_PII)

        response = await output_pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert response == {"role": "assistant", "content": MASKED_MAIN_OUTPUT}

    async def test_a_rail_behind_the_mask_can_still_block(self, output_pipeline_iorails, call_log, httpx_mock):
        """A rewrite ahead of a block does not rescue the response, and nothing behind it runs."""
        _wire_answering(
            output_pipeline_iorails, call_log, httpx_mock, OUTPUT_CONTENT_SAFETY_BLOCKS, MAIN_OUTPUT_WITH_PII
        )

        response = await output_pipeline_iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert response == {"role": "assistant", "content": REFUSAL_MESSAGE}
        assert call_log.order == [GLINER_RAIL, SELF_CHECK_RAIL, CONTENT_SAFETY_RAIL]


@pytest.mark.asyncio
class TestParallelIsRefusedWhenARailRewrites:
    """A config asking for concurrent rails gets sequential ones, because a rewrite cannot compose.

    Concurrent rails all read the text as it arrived, so the masking rail's work would reach
    none of them. The engine says so at construction and runs them in order instead.
    """

    @pytest.mark.parametrize(
        "config_dict, direction, flows",
        [
            (_input_pipeline_config(parallel=True), RailDirection.INPUT, INPUT_FLOWS),
            (_output_pipeline_config(parallel=True), RailDirection.OUTPUT, OUTPUT_FLOWS),
        ],
        ids=["input", "output"],
    )
    async def test_the_downgrade_is_announced_and_applied(self, config_dict, direction, flows):
        """Both parallel flags go off together, and the warning names the rail that forced it."""
        with pytest.warns(UserWarning, match="not honored alongside a rail that rewrites"):
            async with started_iorails(config_dict) as engine:
                assert engine.rails_manager.input_parallel is False
                assert engine.rails_manager.output_parallel is False
                assert engine.rails_manager.transform_flows[direction] == (flows[-1],)

    async def test_the_input_rails_still_run_masking_first(self, call_log, httpx_mock):
        """The downgraded config behaves exactly as the sequential one, rather than merely warning."""
        with pytest.warns(UserWarning, match="not honored"):
            engine = IORails(RailsConfig.from_content(config=_input_pipeline_config(parallel=True)))

        async with engine:
            _wire(engine, call_log, httpx_mock, ALL_ALLOW)
            await engine.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.order[0] == GLINER_RAIL
        assert PERSON not in call_log.text_seen_by(CONTENT_SAFETY_RAIL)

    async def test_the_output_rails_still_run_masking_first(self, call_log, httpx_mock):
        """Same for the output direction, where the rewrite is the response the caller receives."""
        with pytest.warns(UserWarning, match="not honored"):
            engine = IORails(RailsConfig.from_content(config=_output_pipeline_config(parallel=True)))

        async with engine:
            _wire_answering(engine, call_log, httpx_mock, OUTPUT_ALL_ALLOW, MAIN_OUTPUT_WITH_PII)
            response = await engine.generate_async(messages=[{"role": "user", "content": USER_INPUT}])

        assert call_log.order == [GLINER_RAIL, SELF_CHECK_RAIL, CONTENT_SAFETY_RAIL]
        assert response == {"role": "assistant", "content": MASKED_MAIN_OUTPUT}


STREAMED_ANSWER = f"Sure thing, {PERSON} — your order ships tomorrow."
MASKED_STREAMED_ANSWER = "Sure thing, [PERSON] — your order ships tomorrow."
GLINER_OUTPUT_FLOW = "gliner mask pii on output"
CONTENT_SAFETY_OUTPUT_FLOW = "content safety check output $model=content_safety"


def _streaming_config(flows: list[str], *, stream_first: bool = False, context_size: int = 0) -> dict:
    """An output-rail streaming config, masking-safe by default per the settings RailsConfig requires."""
    config = copy.deepcopy(NEMOGUARDS_CONFIG)
    # No jailbreak flow runs in these configs, and leaving its section in warns on every one.
    config["rails"]["config"].pop("jailbreak_detection", None)
    config["rails"]["input"] = {"flows": []}
    config["rails"]["output"] = {
        "flows": flows,
        "streaming": {
            "enabled": True,
            "chunk_size": 200,
            "context_size": context_size,
            "stream_first": stream_first,
        },
    }
    config["rails"]["config"]["gliner"] = {"server_endpoint": GLINER_URL, "output": {"entities": ["person"]}}
    return config


async def _stream_of(text: str):
    """A main-model stream delivering *text* in one chunk, so one batch reaches the output rails."""

    async def _stream(model_type, messages, **kwargs):
        yield LLMResponseChunk(delta_content=text)

    return _stream


async def _streamed(engine: IORails) -> str:
    """Everything the caller received, joined."""
    chunks = [str(chunk) async for chunk in engine.stream_async(messages=[{"role": "user", "content": USER_INPUT}])]
    return "".join(chunks)


@pytest.mark.asyncio
class TestMaskingRailWhileStreaming:
    """A masking output rail applies to what is streamed, rather than being computed and dropped."""

    async def _engine(self, flows: list[str], call_log: RailCallLog, httpx_mock, verdicts: dict, **streaming):
        engine = IORails(RailsConfig.from_content(config=_streaming_config(flows, **streaming)))
        for model_type, rail_engine in engine.engine_registry.llms.items():
            if model_type in verdicts:
                rail_engine.chat_completion = _model_double(call_log, model_type, verdicts[model_type])
        _register_http_doubles(httpx_mock, call_log, gliner_runs=GLINER_OUTPUT_FLOW in flows)
        engine.engine_registry.stream_model_call = await _stream_of(STREAMED_ANSWER)
        return engine

    async def test_a_single_masking_rail_masks_what_is_streamed(self, call_log, httpx_mock):
        """The one rail case: the caller reads the masked answer, never the name."""
        engine = await self._engine([GLINER_OUTPUT_FLOW], call_log, httpx_mock, {})

        async with engine:
            streamed = await _streamed(engine)

        assert MASKED_STREAMED_ANSWER in streamed
        assert PERSON not in streamed

    async def test_a_judging_rail_behind_the_mask_reads_the_masked_answer(self, call_log, httpx_mock):
        """Masking first, judging second — the same order the non-streaming path runs them in."""
        engine = await self._engine(
            [CONTENT_SAFETY_OUTPUT_FLOW, GLINER_OUTPUT_FLOW], call_log, httpx_mock, OUTPUT_ALL_ALLOW
        )

        async with engine:
            streamed = await _streamed(engine)

        assert call_log.order == [GLINER_RAIL, CONTENT_SAFETY_RAIL]
        # The response is what an output mask rewrites; the user's own turn reaches this rail as
        # conversation context and is untouched, since no input mask is configured here.
        assert MASKED_STREAMED_ANSWER in call_log.text_seen_by(CONTENT_SAFETY_RAIL)
        assert STREAMED_ANSWER not in call_log.text_seen_by(CONTENT_SAFETY_RAIL)
        assert MASKED_STREAMED_ANSWER in streamed

    async def test_a_judging_rail_behind_the_mask_can_still_block(self, call_log, httpx_mock):
        """The masked batch is judged before it ships, so a block stops it reaching the caller."""
        engine = await self._engine(
            [CONTENT_SAFETY_OUTPUT_FLOW, GLINER_OUTPUT_FLOW], call_log, httpx_mock, OUTPUT_CONTENT_SAFETY_BLOCKS
        )

        async with engine:
            streamed = await _streamed(engine)

        assert MASKED_STREAMED_ANSWER not in streamed
        assert PERSON not in streamed
        assert "content_blocked" in streamed

    async def test_a_judging_rail_alone_still_streams_first(self, call_log, httpx_mock):
        """No masking rail, so the setting that ships chunks before judging them is untouched."""
        engine = await self._engine(
            [CONTENT_SAFETY_OUTPUT_FLOW],
            call_log,
            httpx_mock,
            OUTPUT_ALL_ALLOW,
            stream_first=True,
            context_size=50,
        )

        async with engine:
            streamed = await _streamed(engine)

        assert engine.config.rails.output.streaming.stream_first is True
        assert STREAMED_ANSWER in streamed


@pytest.mark.asyncio
class TestInputMaskingReachesStreamingOutputRails:
    """An input rewrite reaches the output rails, which judge the turn the model answered."""

    async def test_the_output_rail_reads_the_masked_user_message(self, call_log, httpx_mock):
        """Masking input and then handing the raw text to a vendor rail would defeat the mask."""
        config = _streaming_config([CONTENT_SAFETY_OUTPUT_FLOW])
        config["rails"]["input"] = {"flows": ["gliner mask pii on input"]}
        config["rails"]["config"]["gliner"]["input"] = {"entities": ["person"]}
        engine = IORails(RailsConfig.from_content(config=config))
        for model_type, rail_engine in engine.engine_registry.llms.items():
            if model_type in OUTPUT_ALL_ALLOW:
                rail_engine.chat_completion = _model_double(call_log, model_type, OUTPUT_ALL_ALLOW[model_type])
        _register_http_doubles(httpx_mock, call_log)
        engine.engine_registry.stream_model_call = await _stream_of(MAIN_OUTPUT)

        async with engine:
            await _streamed(engine)

        assert MASKED_INPUT in call_log.text_seen_by(CONTENT_SAFETY_RAIL)
        assert PERSON not in call_log.text_seen_by(CONTENT_SAFETY_RAIL)
