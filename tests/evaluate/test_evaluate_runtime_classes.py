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

from nemoguardrails.evaluate.evaluate_factcheck import FactCheckEvaluation
from nemoguardrails.evaluate.evaluate_hallucination import HallucinationRailsEvaluation
from nemoguardrails.evaluate.evaluate_moderation import ModerationRailsEvaluation


def _moderation_evaluator():
    evaluator = ModerationRailsEvaluation.__new__(ModerationRailsEvaluation)
    evaluator.llm = MagicMock()
    evaluator.llm_task_manager = MagicMock()
    evaluator.llm_task_manager.render_task_prompt.return_value = "rendered prompt"
    evaluator.check_input = True
    evaluator.check_output = True
    evaluator.dataset = ["prompt"]
    evaluator.split = "harmful"
    evaluator.write_outputs = False
    evaluator.output_dir = "unused"
    evaluator.dataset_path = "harmful.txt"
    return evaluator


def test_moderation_init_loads_rails_dataset_and_creates_output_dir(tmp_path):
    dataset = tmp_path / "moderation.txt"
    output_dir = tmp_path / "outputs"
    dataset.write_text("one\ntwo\n", encoding="utf-8")
    rails = SimpleNamespace(llm="llm")

    with (
        patch(
            "nemoguardrails.evaluate.evaluate_moderation.RailsConfig.from_path", return_value="config"
        ) as mock_config,
        patch("nemoguardrails.evaluate.evaluate_moderation.LLMRails", return_value=rails) as mock_rails,
        patch(
            "nemoguardrails.evaluate.evaluate_moderation.LLMTaskManager", return_value="task-manager"
        ) as mock_task_manager,
    ):
        evaluator = ModerationRailsEvaluation(
            config="config-path",
            dataset_path=str(dataset),
            num_samples=1,
            output_dir=str(output_dir),
        )

    mock_config.assert_called_once_with("config-path")
    mock_rails.assert_called_once_with("config")
    mock_task_manager.assert_called_once_with("config")
    assert evaluator.dataset == ["one\n"]
    assert evaluator.llm == "llm"
    assert output_dir.exists()


def test_moderation_get_jailbreak_results_counts_flags_and_correct_predictions():
    evaluator = _moderation_evaluator()
    results = {"flagged": 0, "correct": 0, "error": 0, "label": "yes"}

    with patch(
        "nemoguardrails.evaluate.evaluate_moderation.llm_call",
        AsyncMock(return_value=SimpleNamespace(content="YES")),
    ):
        prediction, updated = evaluator.get_jailbreak_results("prompt", results)

    assert prediction == "yes"
    assert updated["flagged"] == 1
    assert updated["correct"] == 1
    evaluator.llm_task_manager.render_task_prompt.assert_called_once()


def test_moderation_get_jailbreak_results_records_error_after_retries():
    evaluator = _moderation_evaluator()
    results = {"flagged": 0, "correct": 0, "error": 0, "label": "yes"}

    with patch(
        "nemoguardrails.evaluate.evaluate_moderation.llm_call",
        AsyncMock(side_effect=RuntimeError("failed")),
    ):
        prediction, updated = evaluator.get_jailbreak_results("prompt", results)

    assert prediction is None
    assert updated["error"] == 1


def test_moderation_get_check_output_results_counts_flags_and_correct_predictions():
    evaluator = _moderation_evaluator()
    results = {"flagged": 0, "correct": 0, "error": 0, "label": "yes"}

    with patch(
        "nemoguardrails.evaluate.evaluate_moderation.llm_call",
        AsyncMock(
            side_effect=[
                SimpleNamespace(content="bot response"),
                SimpleNamespace(content="yes"),
            ]
        ),
    ):
        bot_response, prediction, updated = evaluator.get_check_output_results("prompt", results)

    assert bot_response == "bot response"
    assert prediction == "yes"
    assert updated["flagged"] == 1
    assert updated["correct"] == 1


def test_moderation_check_moderation_combines_enabled_checks():
    evaluator = _moderation_evaluator()
    evaluator.get_jailbreak_results = MagicMock(
        return_value=("yes", {"flagged": 1, "correct": 1, "error": 0, "label": "yes"})
    )
    evaluator.get_check_output_results = MagicMock(
        return_value=(
            "bot",
            "yes",
            {"flagged": 1, "correct": 1, "error": 0, "label": "yes"},
        )
    )

    predictions, jailbreak_results, check_output_results = evaluator.check_moderation()

    assert predictions == [
        {
            "prompt": "prompt",
            "jailbreak": "yes",
            "bot_response": "bot",
            "check_output": "yes",
        }
    ]
    assert jailbreak_results["correct"] == 1
    assert check_output_results["flagged"] == 1


def test_moderation_run_writes_predictions(tmp_path):
    evaluator = _moderation_evaluator()
    evaluator.write_outputs = True
    evaluator.output_dir = str(tmp_path)
    evaluator.dataset_path = "harmful.txt"
    evaluator.check_moderation = MagicMock(
        return_value=(
            [{"prompt": "prompt", "jailbreak": "yes"}],
            {"flagged": 1, "correct": 1, "error": 0},
            {"flagged": 0, "correct": 0, "error": 0},
        )
    )

    evaluator.run()

    output_path = tmp_path / "harmful_harmful_moderation_results.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"prompt": "prompt", "jailbreak": "yes"}]


def _hallucination_evaluator():
    evaluator = HallucinationRailsEvaluation.__new__(HallucinationRailsEvaluation)
    evaluator.dataset = ["question"]
    evaluator.write_outputs = False
    evaluator.output_dir = "unused"
    evaluator.dataset_path = "sample.txt"
    evaluator.llm_task_manager = MagicMock()
    evaluator.llm_task_manager.render_task_prompt.return_value = "check hallucination"
    evaluator.llm = MagicMock(return_value="no")
    return evaluator


def test_hallucination_init_loads_rails_dataset_and_creates_output_dir(tmp_path):
    dataset = tmp_path / "hallucination.txt"
    output_dir = tmp_path / "outputs"
    dataset.write_text("one\ntwo\n", encoding="utf-8")
    rails = SimpleNamespace(llm="llm")

    with (
        patch(
            "nemoguardrails.evaluate.evaluate_hallucination.RailsConfig.from_path", return_value="config"
        ) as mock_config,
        patch("nemoguardrails.evaluate.evaluate_hallucination.LLMRails", return_value=rails) as mock_rails,
        patch(
            "nemoguardrails.evaluate.evaluate_hallucination.LLMTaskManager", return_value="task-manager"
        ) as mock_task_manager,
    ):
        evaluator = HallucinationRailsEvaluation(
            config="config-path",
            dataset_path=str(dataset),
            num_samples=1,
            output_dir=str(output_dir),
        )

    mock_config.assert_called_once_with("config-path")
    mock_rails.assert_called_once_with("config")
    mock_task_manager.assert_called_once_with("config")
    assert evaluator.dataset == ["one\n"]
    assert evaluator.llm == "llm"
    assert output_dir.exists()


def test_hallucination_get_response_with_retries_uses_bound_llm_params():
    evaluator = _hallucination_evaluator()
    bound = MagicMock(return_value="bound response")
    evaluator.llm = MagicMock()
    evaluator.llm.bind.return_value = bound

    response = evaluator.get_response_with_retries(
        "prompt",
        max_tries=2,
        llm_params={"temperature": 1.0},
    )

    assert response == "bound response"
    evaluator.llm.bind.assert_called_once_with(temperature=1.0)
    bound.assert_called_once_with("prompt")


def test_hallucination_get_extra_responses_skips_errors():
    evaluator = _hallucination_evaluator()
    evaluator.get_response_with_retries = MagicMock(side_effect=[None, "extra"])

    assert evaluator.get_extra_responses("prompt", num_responses=2) == ["extra"]


def test_hallucination_get_response_with_retries_returns_none_after_errors():
    evaluator = _hallucination_evaluator()
    evaluator.llm = MagicMock(side_effect=RuntimeError("failed"))

    assert evaluator.get_response_with_retries("prompt", max_tries=2) is None


def test_hallucination_self_check_counts_no_as_flagged():
    evaluator = _hallucination_evaluator()
    evaluator.get_response_with_retries = MagicMock(return_value="main response")
    evaluator.get_extra_responses = MagicMock(return_value=["extra one", "extra two"])

    predictions, num_flagged, num_error = evaluator.self_check_hallucination()

    assert num_flagged == 1
    assert num_error == 0
    assert predictions[0]["hallucination_agreement"] == "no"


def test_hallucination_self_check_records_error_when_main_response_fails():
    evaluator = _hallucination_evaluator()
    evaluator.get_response_with_retries = MagicMock(return_value=None)

    predictions, num_flagged, num_error = evaluator.self_check_hallucination()

    assert num_flagged == 0
    assert num_error == 1
    assert predictions[0]["hallucination_agreement"] == "na"


def test_hallucination_run_writes_predictions(tmp_path):
    evaluator = _hallucination_evaluator()
    evaluator.write_outputs = True
    evaluator.output_dir = str(tmp_path)
    evaluator.dataset_path = "sample.txt"
    evaluator.self_check_hallucination = MagicMock(return_value=([{"question": "q"}], 1, 0))

    evaluator.run()

    output_path = tmp_path / "sample_hallucination_predictions.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"question": "q"}]


def _factcheck_evaluator():
    evaluator = FactCheckEvaluation.__new__(FactCheckEvaluation)
    evaluator.dataset = [
        {
            "question": "q",
            "evidence": "e",
            "answer": "a",
            "incorrect_answer": "bad",
        }
    ]
    evaluator.llm = MagicMock()
    evaluator.llm_task_manager = MagicMock()
    evaluator.llm_task_manager.render_task_prompt.return_value = "fact prompt"
    evaluator.llm_task_manager.get_stop_tokens.return_value = ["stop"]
    evaluator.create_negatives = False
    evaluator.write_outputs = False
    evaluator.output_dir = "unused"
    evaluator.dataset_path = "sample.json"
    return evaluator


def test_factcheck_init_loads_rails_dataset_and_creates_output_dir(tmp_path):
    dataset = tmp_path / "fact.json"
    output_dir = tmp_path / "outputs"
    dataset.write_text(json.dumps([{"question": "q"}]), encoding="utf-8")
    rails = SimpleNamespace(llm="llm")

    with (
        patch("nemoguardrails.evaluate.evaluate_factcheck.RailsConfig.from_path", return_value="config") as mock_config,
        patch("nemoguardrails.evaluate.evaluate_factcheck.LLMRails", return_value=rails) as mock_rails,
        patch(
            "nemoguardrails.evaluate.evaluate_factcheck.LLMTaskManager", return_value="task-manager"
        ) as mock_task_manager,
    ):
        evaluator = FactCheckEvaluation(
            config="config-path",
            dataset_path=str(dataset),
            num_samples=1,
            output_dir=str(output_dir),
        )

    mock_config.assert_called_once_with("config-path")
    mock_rails.assert_called_once_with("config")
    mock_task_manager.assert_called_once_with("config")
    assert evaluator.dataset == [{"question": "q"}]
    assert evaluator.llm == "llm"
    assert output_dir.exists()


@pytest.mark.asyncio
async def test_factcheck_create_negative_samples_adds_incorrect_answer():
    evaluator = _factcheck_evaluator()
    evaluator.llm.generate_async = AsyncMock(return_value=SimpleNamespace(content=" incorrect "))
    dataset = [{"question": "q", "evidence": "e", "answer": "a"}]

    result = await evaluator.create_negative_samples(dataset)

    assert result[0]["incorrect_answer"] == "incorrect"


def test_factcheck_check_facts_uses_positive_and_negative_labels():
    evaluator = _factcheck_evaluator()

    with (
        patch(
            "nemoguardrails.evaluate.evaluate_factcheck.llm_call",
            AsyncMock(return_value=SimpleNamespace(content="yes")),
        ) as mock_llm_call,
        patch("nemoguardrails.evaluate.evaluate_factcheck.time.sleep"),
    ):
        predictions, num_correct, total_time = evaluator.check_facts(split="positive")

    assert predictions[0]["answer"] == "a"
    assert predictions[0]["label"] == "yes"
    assert num_correct == 1
    assert total_time >= 0
    mock_llm_call.assert_awaited_once()

    with (
        patch(
            "nemoguardrails.evaluate.evaluate_factcheck.llm_call",
            AsyncMock(return_value=SimpleNamespace(content="no")),
        ),
        patch("nemoguardrails.evaluate.evaluate_factcheck.time.sleep"),
    ):
        predictions, num_correct, _ = evaluator.check_facts(split="negative")

    assert predictions[0]["answer"] == "bad"
    assert predictions[0]["label"] == "no"
    assert num_correct == 1


def test_factcheck_run_writes_positive_and_negative_predictions(tmp_path):
    evaluator = _factcheck_evaluator()
    evaluator.write_outputs = True
    evaluator.output_dir = str(tmp_path)
    evaluator.dataset_path = "sample.json"
    evaluator.check_facts = MagicMock(
        side_effect=[
            ([{"label": "yes"}], 1, 0.1),
            ([{"label": "no"}], 1, 0.2),
        ]
    )

    evaluator.run()

    positive_path = tmp_path / "sample_positive_fact_check_predictions.json"
    negative_path = tmp_path / "sample_negative_fact_check_predictions.json"
    assert json.loads(positive_path.read_text(encoding="utf-8")) == [{"label": "yes"}]
    assert json.loads(negative_path.read_text(encoding="utf-8")) == [{"label": "no"}]


def test_factcheck_run_creates_negative_samples_when_enabled(tmp_path):
    evaluator = _factcheck_evaluator()
    evaluator.create_negatives = True
    evaluator.create_negative_samples = AsyncMock(return_value=evaluator.dataset)
    evaluator.check_facts = MagicMock(
        side_effect=[
            ([{"label": "yes"}], 1, 0.1),
            ([{"label": "no"}], 1, 0.2),
        ]
    )

    evaluator.run()

    evaluator.create_negative_samples.assert_awaited_once_with(evaluator.dataset)
    assert evaluator.check_facts.call_count == 2
