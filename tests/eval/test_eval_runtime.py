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

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.eval.eval import (
    _extract_interaction_log,
    _extract_interaction_outputs,
    _extract_spans,
    _load_eval_output,
    run_eval,
)
from nemoguardrails.eval.models import (
    EvalConfig,
    EvalOutput,
    InteractionLog,
    InteractionOutput,
    InteractionSet,
    Policy,
    Span,
)
from nemoguardrails.eval.utils import (
    _collect_span_metrics,
    get_output_paths,
    load_dict_from_file,
    load_dict_from_path,
    save_dict_to_file,
    save_eval_output,
    update_dict_at_path,
)
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    ExecutedAction,
    GenerationLog,
    GenerationResponse,
)


def _eval_config():
    return EvalConfig(
        policies=[
            Policy(id="global", description="global policy"),
            Policy(id="targeted", description="targeted policy", apply_to_all=False),
            Policy(id="included", description="included policy", apply_to_all=False),
            Policy(id="excluded", description="excluded policy"),
        ],
        interactions=[
            InteractionSet(
                id="set",
                inputs=[
                    "hello",
                    {
                        "type": "messages",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                ],
                expected_output=[{"type": "refusal", "policy": "targeted"}],
                include_policies=["included"],
                exclude_policies=["excluded"],
            )
        ],
    )


def _activated_rail():
    return ActivatedRail(
        type="input",
        name="check input",
        started_at=10.0,
        finished_at=13.0,
        duration=3.0,
        executed_actions=[
            ExecutedAction(
                action_name="self_check",
                started_at=10.5,
                finished_at=12.5,
                duration=2.0,
                llm_calls=[
                    LLMCallInfo(
                        llm_model_name="model/name",
                        started_at=11.0,
                        finished_at=12.0,
                        duration=1.0,
                        prompt_tokens=7,
                        completion_tokens=3,
                        total_tokens=10,
                    )
                ],
            )
        ],
    )


def test_extract_interaction_outputs_initializes_policy_statuses():
    outputs = _extract_interaction_outputs(_eval_config())

    assert [output.id for output in outputs] == ["set/0", "set/1"]
    assert outputs[0].input == "hello"
    assert outputs[1].input["messages"][0]["content"] == "hello"
    assert outputs[0].compliance == {
        "global": None,
        "targeted": None,
        "included": None,
        "excluded": "n/a",
    }


def test_load_eval_output_reuses_matching_results_and_replaces_changed_inputs(tmp_path):
    existing_output = EvalOutput(
        results=[
            InteractionOutput(id="set/0", input="hello", output="old output", compliance={"global": True}),
            InteractionOutput(id="set/1", input="old input", output="stale output", compliance={"global": False}),
        ],
        logs=[InteractionLog(id="set/0", events=[{"type": "Existing"}])],
    )
    save_eval_output(existing_output, str(tmp_path), "json")

    output = _load_eval_output(str(tmp_path), _eval_config())

    assert output.results[0].output == "old output"
    assert output.logs[0].events == [{"type": "Existing"}]
    assert output.results[1].output is None
    assert output.results[1].input["messages"][0]["content"] == "hello"
    assert output.logs[1] == InteractionLog(id="set/1")


def test_extract_spans_builds_trace_and_metrics():
    spans = _extract_spans([_activated_rail()])

    assert [span.name for span in spans] == [
        "interaction",
        "rail: check input",
        "action: self_check",
        "LLM: model/name",
    ]
    assert spans[0].duration == 3.0
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].metrics["action_self_check_seconds_total"] == 2.0
    assert spans[3].metrics["llm_call_model_name_prompt_tokens_total"] == 7
    assert spans[3].metrics["llm_call_model_name_tokens_total"] == 10


def test_extract_interaction_log_uses_generation_log_data():
    interaction = InteractionOutput(id="set/0", input="hello")
    generation_log = GenerationLog(
        activated_rails=[_activated_rail()],
        internal_events=[{"type": "UserIntent", "intent": "greet"}],
    )

    log = _extract_interaction_log(interaction, generation_log)

    assert log.id == "set/0"
    assert log.activated_rails == [_activated_rail()]
    assert log.events == [{"type": "UserIntent", "intent": "greet"}]
    assert log.trace[0].name == "interaction"


def test_collect_span_metrics_sums_totals_and_averages_avg_metrics():
    metrics = _collect_span_metrics(
        [
            Span(span_id="1", name="one", start_time=0, end_time=1, duration=1, metrics={"calls_total": 1}),
            Span(
                span_id="2",
                name="two",
                start_time=1,
                end_time=2,
                duration=1,
                metrics={"calls_total": 2, "latency_avg": 4.0},
            ),
            Span(
                span_id="3",
                name="three",
                start_time=2,
                end_time=3,
                duration=1,
                metrics={"latency_avg": 6.0},
            ),
        ]
    )

    assert metrics == {"calls_total": 3, "latency_avg": 5.0}


def test_eval_output_compute_compliance_counts_statuses():
    eval_config = EvalConfig(
        policies=[
            Policy(id="policy", description="policy"),
            Policy(id="missing", description="missing"),
        ],
        interactions=[],
    )
    output = EvalOutput(
        results=[
            InteractionOutput(id="1", input="a", compliance={"policy": True, "missing": None}),
            InteractionOutput(id="2", input="b", compliance={"policy": False, "missing": None}),
            InteractionOutput(id="3", input="c", compliance={"policy": "n/a", "missing": None}),
            InteractionOutput(id="4", input="d", compliance={"policy": None, "missing": None}),
        ]
    )

    compliance = output.compute_compliance(eval_config)

    assert compliance["policy"]["rate"] == pytest.approx(1 / 3)
    assert compliance["policy"]["interactions_comply_count"] == 1
    assert compliance["policy"]["interactions_violation_count"] == 1
    assert compliance["policy"]["interactions_not_applicable_count"] == 1
    assert compliance["policy"]["interactions_not_rated_count"] == 1
    assert compliance["missing"]["interactions_not_rated_count"] == 4


def test_load_save_and_update_dict_helpers(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yaml_file = config_dir / "policies.yaml"
    json_file = config_dir / "interactions.json"
    yaml_file.write_text("policies:\n  - id: p1\n    description: one\n", encoding="utf-8")
    json_file.write_text(json.dumps({"interactions": [{"id": "i1"}]}), encoding="utf-8")

    assert load_dict_from_file(str(yaml_file)) == {"policies": [{"id": "p1", "description": "one"}]}
    assert load_dict_from_path(str(config_dir)) == {
        "policies": [{"id": "p1", "description": "one"}],
        "interactions": [{"id": "i1"}],
    }

    update_dict_at_path(str(config_dir), {"interactions": [{"id": "i2"}]})
    assert load_dict_from_file(str(json_file)) == {"interactions": [{"id": "i2"}]}

    save_dict_to_file({"value": 1}, str(tmp_path / "saved"), "json")
    assert load_dict_from_file(str(tmp_path / "saved.json")) == {"value": 1}

    (tmp_path / "run-a").mkdir()
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / ".hidden").mkdir()
    monkeypatch.chdir(tmp_path)
    assert get_output_paths() == [str(tmp_path / "run-a")]


@pytest.mark.asyncio
async def test_run_eval_generates_for_string_and_message_inputs(tmp_path):
    eval_config = _eval_config()
    rails = MagicMock()
    rails.generate_async = AsyncMock(
        return_value=GenerationResponse(
            response="ok",
            log=GenerationLog(
                activated_rails=[_activated_rail()],
                internal_events=[{"type": "Done"}],
            ),
        )
    )

    with (
        patch("nemoguardrails.eval.eval.EvalConfig.from_path", return_value=eval_config) as mock_eval_config,
        patch("nemoguardrails.eval.eval.RailsConfig.from_path", return_value=SimpleNamespace()) as mock_rails_config,
        patch("nemoguardrails.eval.eval.LLMRails", return_value=rails) as mock_rails_cls,
    ):
        await run_eval(
            eval_config_path=str(tmp_path / "eval"),
            guardrail_config_path=str(tmp_path / "guardrails"),
            output_path=str(tmp_path / "output"),
            output_format="json",
            parallel=1,
        )

    mock_eval_config.assert_called_once()
    mock_rails_config.assert_called_once_with(str(tmp_path / "guardrails"))
    mock_rails_cls.assert_called_once()
    assert rails.generate_async.await_count == 2
    first_call = rails.generate_async.await_args_list[0].kwargs
    second_call = rails.generate_async.await_args_list[1].kwargs
    assert first_call["prompt"] == "hello"
    assert second_call["messages"] == [{"role": "user", "content": "hello"}]
    output = EvalOutput.from_path(str(tmp_path / "output"))
    assert [result.output for result in output.results] == ["ok", "ok"]
    assert output.logs[0].events == [{"type": "Done"}]
