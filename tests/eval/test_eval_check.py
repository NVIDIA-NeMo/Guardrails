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
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from nemoguardrails.eval.check import LLMJudgeComplianceChecker
from nemoguardrails.eval.models import (
    EvalConfig,
    EvalOutput,
    InteractionLog,
    InteractionOutput,
    InteractionSet,
    Policy,
)
from nemoguardrails.eval.ui.utils import EvalData
from nemoguardrails.rails.llm.config import Model


def _checker(*, force=False, reset=False, policy_apply_to_all=True, response="Reason: ok\nCompliance: Yes"):
    checker = LLMJudgeComplianceChecker.__new__(LLMJudgeComplianceChecker)
    checker.eval_config = EvalConfig(policies=[Policy(id="policy", description="policy")], interactions=[], prompts=[])
    checker.policies = checker.eval_config.policies
    checker.policy_by_id = {"policy": Policy(id="policy", description="policy", apply_to_all=policy_apply_to_all)}
    checker.policy_ids = ["policy"]
    checker.verbose = False
    checker.force = force
    checker.reset = reset
    checker.parallel = 1
    checker.llm = MagicMock()
    checker.llm_judge_model = "judge"
    checker.progress = MagicMock()
    checker.llm_task_manager = MagicMock()
    checker.llm_task_manager.render_task_prompt.return_value = "rendered prompt"
    checker.llm_response = response
    return checker


def _interaction_set(*, include=None, exclude=None, expected=None):
    return InteractionSet(
        id="set",
        inputs=["hello"],
        expected_output=expected or [],
        include_policies=include or [],
        exclude_policies=exclude or [],
    )


def test_compliance_checker_init_builds_model_task_manager_and_eval_data(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "token")
    eval_config = EvalConfig(
        policies=[Policy(id="policy", description="policy")],
        interactions=[],
        models=[
            Model(
                type="judge",
                engine="mock",
                model="judge-model",
                api_key_env_var="JUDGE_API_KEY",
                parameters={"temperature": 0},
            )
        ],
        prompts=[],
    )
    task_manager = MagicMock()

    with (
        patch("nemoguardrails.eval.check.EvalConfig.from_path", return_value=eval_config) as mock_from_path,
        patch("nemoguardrails.eval.check.init_llm_model", return_value="llm") as mock_init_llm_model,
        patch(
            "nemoguardrails.eval.check.RailsConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        ) as mock_config,
        patch("nemoguardrails.eval.check.LLMTaskManager", return_value=task_manager) as mock_task_manager,
    ):
        checker = LLMJudgeComplianceChecker(
            eval_config_path=str(tmp_path),
            output_paths=["run-a"],
            llm_judge_model="judge-model",
            policy_ids=[],
            verbose=True,
            force=True,
            reset=True,
            parallel=2,
        )

    mock_from_path.assert_called_once_with(str(tmp_path))
    mock_init_llm_model.assert_called_once_with(
        model_name="judge-model",
        provider_name="mock",
        mode="chat",
        kwargs={"temperature": 0, "api_key": "token"},
    )
    task_manager_models = mock_config.call_args.kwargs["models"]
    assert task_manager_models[0] == eval_config.models[0]
    assert task_manager_models[1].type == "main"
    assert task_manager_models[1].model == "judge-model"
    mock_task_manager.assert_called_once()
    assert checker.llm == "llm"
    assert checker.policy_ids == ["policy"]
    assert checker.eval_data.output_paths == ["run-a"]
    assert checker.verbose is True
    assert checker.force is True
    assert checker.reset is True
    assert checker.parallel == 2


def test_compliance_checker_print_helpers_delegate_to_progress():
    checker = LLMJudgeComplianceChecker.__new__(LLMJudgeComplianceChecker)
    checker.progress = MagicMock()
    checker.parallel = 1

    checker.print_prompt("[cyan]prompt[/]\n[/]\nplain")
    checker.print_completion("completion")
    checker.print_progress_detail("detail")

    printed = [call.args[0] for call in checker.progress.print.call_args_list]
    assert len(printed) == 4
    assert printed[-1] == "detail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reason: good\nCompliance: Yes", True),
        ("Reason: bad\nCompliance: No", False),
        ("Reason: skip\nCompliance: n/a", "n/a"),
    ],
)
async def test_check_interaction_compliance_records_valid_judgements(response, expected):
    checker = _checker(response=response)
    output = InteractionOutput(id="set/0", input="hello", compliance={"policy": None})
    log = InteractionLog(id="set/0", events=[{"type": "Event"}])

    with patch("nemoguardrails.eval.check.llm_call", AsyncMock(return_value=SimpleNamespace(content=response))):
        changed = await checker.check_interaction_compliance(output, log, _interaction_set(), 1)

    assert changed is True
    assert output.compliance["policy"] == expected
    assert output.compliance_checks[0].method == "judge"
    assert output.compliance_checks[0].compliance == {"policy": expected}
    assert log.compliance_checks[0].llm_calls[0].task == "llm_judge_check_single_policy_compliance"


@pytest.mark.asyncio
async def test_check_interaction_compliance_turns_targeted_na_into_failure():
    checker = _checker(response="Reason: skip\nCompliance: n/a")
    output = InteractionOutput(id="set/0", input="hello", compliance={"policy": None})
    log = InteractionLog(id="set/0", events=[])

    with patch(
        "nemoguardrails.eval.check.llm_call",
        AsyncMock(return_value=SimpleNamespace(content="Reason: skip\nCompliance: n/a")),
    ):
        changed = await checker.check_interaction_compliance(
            output,
            log,
            _interaction_set(include=["policy"]),
            1,
        )

    assert changed is True
    assert output.compliance["policy"] is False
    assert "not acceptable" in output.compliance_checks[0].details


@pytest.mark.asyncio
async def test_check_interaction_compliance_skips_not_applicable_policy():
    checker = _checker(policy_apply_to_all=False)
    output = InteractionOutput(id="set/0", input="hello", compliance={"policy": None})
    log = InteractionLog(id="set/0", events=[])

    with patch("nemoguardrails.eval.check.llm_call", AsyncMock()) as mock_llm_call:
        changed = await checker.check_interaction_compliance(output, log, _interaction_set(), 1)

    assert changed is True
    assert output.compliance["policy"] == "n/a"
    mock_llm_call.assert_not_called()


@pytest.mark.asyncio
async def test_check_interaction_compliance_skips_existing_rating_without_force():
    checker = _checker(force=False)
    output = InteractionOutput(id="set/0", input="hello", compliance={"policy": True})
    log = InteractionLog(id="set/0", events=[])

    with patch("nemoguardrails.eval.check.llm_call", AsyncMock()) as mock_llm_call:
        changed = await checker.check_interaction_compliance(output, log, _interaction_set(), 1)

    assert changed is False
    assert output.compliance_checks == []
    mock_llm_call.assert_not_called()


@pytest.mark.asyncio
async def test_check_interaction_compliance_force_rechecks_existing_rating():
    checker = _checker(force=True, response="Reason: changed\nCompliance: No")
    output = InteractionOutput(id="set/0", input="hello", compliance={"policy": True})
    log = InteractionLog(id="set/0", events=[])

    with patch(
        "nemoguardrails.eval.check.llm_call",
        AsyncMock(return_value=SimpleNamespace(content="Reason: changed\nCompliance: No")),
    ):
        changed = await checker.check_interaction_compliance(output, log, _interaction_set(), 1)

    assert changed is True
    assert output.compliance["policy"] is False


@pytest.mark.asyncio
async def test_check_interaction_compliance_reset_clears_existing_checks():
    checker = _checker(reset=True)
    output = InteractionOutput(
        id="set/0",
        input="hello",
        compliance={"policy": None},
        compliance_checks=[
            {
                "id": "old",
                "created_at": "2024-01-01T00:00:00",
                "interaction_id": "set/0",
                "method": "old",
                "compliance": {"policy": False},
                "details": "",
            }
        ],
    )
    log = InteractionLog(id="set/0", compliance_checks=[{"id": "old", "llm_calls": []}])

    with patch(
        "nemoguardrails.eval.check.llm_call",
        AsyncMock(return_value=SimpleNamespace(content="Reason: ok\nCompliance: Yes")),
    ):
        changed = await checker.check_interaction_compliance(output, log, _interaction_set(), 1)

    assert changed is True
    assert len(output.compliance_checks) == 1
    assert output.compliance_checks[0].method == "judge"
    assert len(log.compliance_checks) == 1


@pytest.mark.asyncio
async def test_check_interaction_compliance_ignores_invalid_response():
    checker = _checker(response="not parseable")
    output = InteractionOutput(id="set/0", input="hello", compliance={"policy": None})
    log = InteractionLog(id="set/0", events=[])

    with patch(
        "nemoguardrails.eval.check.llm_call",
        AsyncMock(return_value=SimpleNamespace(content="not parseable")),
    ):
        changed = await checker.check_interaction_compliance(output, log, _interaction_set(), 1)

    assert changed is False
    assert output.compliance["policy"] is None
    assert output.compliance_checks == []


@pytest.mark.asyncio
async def test_compliance_checker_run_updates_changed_outputs(tmp_path):
    interaction_set = _interaction_set()
    eval_config = EvalConfig(
        policies=[Policy(id="policy", description="policy")],
        interactions=[interaction_set],
    )
    eval_output = EvalOutput(
        results=[
            InteractionOutput(id="set/0", input="hello", output="hi", compliance={"policy": None}),
            InteractionOutput(id="set/1", input="bye", output="bye", compliance={"policy": None}),
        ],
        logs=[InteractionLog(id="set/0"), InteractionLog(id="set/1")],
    )
    checker = LLMJudgeComplianceChecker.__new__(LLMJudgeComplianceChecker)
    checker.output_paths = [str(tmp_path)]
    checker.eval_config = eval_config
    checker.eval_data = EvalData(
        eval_config_path="config",
        eval_config=eval_config,
        output_paths=[str(tmp_path)],
        eval_outputs={},
    )
    checker.parallel = 1
    checker.check_interaction_compliance = AsyncMock(side_effect=[True, False])
    checker.progress_idx = 0

    with (
        patch("nemoguardrails.eval.check.EvalOutput.from_path", return_value=eval_output),
        patch.object(EvalData, "update_results_and_logs", autospec=True) as mock_update_results_and_logs,
    ):
        await checker.run()

    assert checker.eval_data.eval_outputs[str(tmp_path)] == eval_output
    assert checker.check_interaction_compliance.await_count == 2
    assert checker.check_interaction_compliance.await_args_list[0].kwargs["interaction_set"] == interaction_set
    assert mock_update_results_and_logs.call_args_list == [
        call(checker.eval_data, str(tmp_path)),
        call(checker.eval_data, str(tmp_path)),
    ]
