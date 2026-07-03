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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemoguardrails.eval import cli as eval_cli

runner = CliRunner()


def test_run_command_requires_guardrail_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    result = runner.invoke(eval_cli.app, ["run", "-e", str(config_dir)])

    assert result.exit_code == 1
    assert "No guardrail configuration provided" in result.stdout


def test_run_command_invokes_run_eval(tmp_path):
    eval_dir = tmp_path / "eval"
    guardrails_dir = tmp_path / "guardrails"
    output_dir = tmp_path / "output"
    eval_dir.mkdir()
    guardrails_dir.mkdir()

    with patch("nemoguardrails.eval.cli.run_eval", AsyncMock()) as mock_run_eval:
        result = runner.invoke(
            eval_cli.app,
            [
                "run",
                "-e",
                str(eval_dir),
                "-g",
                str(guardrails_dir),
                "-o",
                str(output_dir),
                "--output-format",
                "YAML",
                "--parallel",
                "3",
            ],
        )

    assert result.exit_code == 0
    mock_run_eval.assert_awaited_once_with(
        eval_config_path=str(eval_dir.resolve()),
        guardrail_config_path=str(guardrails_dir),
        output_path=str(output_dir),
        output_format="yaml",
        parallel=3,
    )


def test_check_compliance_command_uses_explicit_output_paths(tmp_path):
    eval_dir = tmp_path / "eval"
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"
    eval_dir.mkdir()
    out_a.mkdir()
    out_b.mkdir()
    checker = MagicMock()
    checker.run = AsyncMock()

    with patch("nemoguardrails.eval.cli.LLMJudgeComplianceChecker", return_value=checker) as mock_checker:
        result = runner.invoke(
            eval_cli.app,
            [
                "check-compliance",
                "--llm-judge",
                "judge-model",
                "-e",
                str(eval_dir),
                "-o",
                f"{out_a},{out_b}",
                "-p",
                "policy-a",
                "-p",
                "policy-b",
                "--force",
                "--reset",
                "--parallel",
                "2",
                "--disable-llm-cache",
            ],
        )

    assert result.exit_code == 0
    mock_checker.assert_called_once_with(
        eval_config_path=str(eval_dir),
        output_paths=[str(out_a), str(out_b)],
        llm_judge_model="judge-model",
        policy_ids=["policy-a", "policy-b"],
        verbose=False,
        force=True,
        reset=True,
        parallel=2,
    )
    checker.run.assert_awaited_once()


def test_check_compliance_command_discovers_output_paths(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    checker = MagicMock()
    checker.run = AsyncMock()

    with (
        patch("nemoguardrails.eval.cli.get_output_paths", return_value=["run-a"]) as mock_get_output_paths,
        patch("nemoguardrails.eval.cli.LLMJudgeComplianceChecker", return_value=checker) as mock_checker,
    ):
        result = runner.invoke(
            eval_cli.app,
            [
                "check-compliance",
                "--llm-judge",
                "judge-model",
                "-e",
                str(eval_dir),
                "--disable-llm-cache",
            ],
        )

    assert result.exit_code == 0
    mock_get_output_paths.assert_called_once()
    mock_checker.assert_called_once()
    assert mock_checker.call_args.kwargs["output_paths"] == ["run-a"]


def test_ui_command_launches_readme_page(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    with (
        patch("nemoguardrails.eval.cli.get_output_paths", return_value=["run-a"]) as mock_get_output_paths,
        patch("nemoguardrails.eval.cli._launch_ui") as mock_launch_ui,
    ):
        result = runner.invoke(eval_cli.app, ["ui", "--eval-config-path", str(eval_dir)])

    assert result.exit_code == 0
    mock_get_output_paths.assert_called_once()
    mock_launch_ui.assert_called_once_with("README.py", port=8501)


def test_launch_ui_exits_when_streamlit_is_missing():
    with patch.dict("sys.modules", {"streamlit.web": None}), pytest.raises(SystemExit) as exc_info:
        eval_cli._launch_ui("README.py")

    assert exc_info.value.code == 1
