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
from pathlib import Path

from typer.testing import CliRunner

from nemoguardrails.cli import app

runner = CliRunner()


def _write_cleanlab_config(path: Path) -> None:
    path.mkdir()
    path.joinpath("config.yml").write_text(
        "models: []\nrails:\n  output:\n    flows:\n      - cleanlab trustworthiness\n",
        encoding="utf-8",
    )


def test_requirements_human_output_separates_required_and_optional_packages():
    result = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "cleanlab", "--rail", "content_safety"],
    )

    assert result.exit_code == 0
    assert "Rails: cleanlab, content_safety" in result.stdout
    assert "Required Python packages:\n  - cleanlab-studio" in result.stdout
    assert "Required install command: pip install cleanlab-studio" in result.stdout
    assert "Optional Python packages:\n  - fast-langdetect>=1" in result.stdout
    assert "Optional install command: pip install 'fast-langdetect>=1'" in result.stdout


def test_requirements_human_output_aggregates_shared_optional_packages():
    result = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "hf_classifier", "--rail", "jailbreak_detection"],
    )

    assert result.exit_code == 0
    assert result.stdout.count("  - transformers>=4.35 ") == 1
    assert "rails: hf_classifier, jailbreak_detection" in result.stdout
    assert "Used for local execution; remote execution does not require it." in result.stdout


def test_requirements_selects_rails_from_config(tmp_path):
    config = tmp_path / "guardrails"
    _write_cleanlab_config(config)

    result = runner.invoke(app, ["rails", "requirements", "--config", str(config)])

    assert result.exit_code == 0
    assert "Rails: cleanlab" in result.stdout
    assert "cleanlab-studio" in result.stdout


def test_requirements_rejects_missing_or_conflicting_selection(tmp_path):
    config = tmp_path / "guardrails"
    _write_cleanlab_config(config)

    missing = runner.invoke(app, ["rails", "requirements"])
    conflicting = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "cleanlab", "--config", str(config)],
    )

    assert missing.exit_code == 2
    assert "Provide at least one --rail or a --config path" in missing.stderr
    assert conflicting.exit_code == 2
    assert "Use either --rail or --config, not both" in conflicting.stderr


def test_requirements_rejects_unknown_rail():
    result = runner.invoke(app, ["rails", "requirements", "--rail", "does_not_exist"])

    assert result.exit_code == 2
    assert "Unknown rail names: does_not_exist" in result.stderr


def test_requirements_json_output_is_structured_and_does_not_read_secret_values(monkeypatch):
    monkeypatch.setenv("CLEANLAB_API_KEY", "top-secret")

    result = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "cleanlab", "--format", "json"],
    )

    assert result.exit_code == 0
    assert "top-secret" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["rail_names"] == ["cleanlab"]
    assert payload["python_packages"]["required"][0]["requirement"] == "cleanlab-studio"
    assert payload["environment_variables"]["required"][0]["name"] == "CLEANLAB_API_KEY"


def test_requirements_file_output_keeps_optional_packages_commented():
    result = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "jailbreak_detection", "--format", "requirements"],
    )

    assert result.exit_code == 0
    assert "# Optional Python packages (commented out)" in result.stdout
    assert "# transformers>=4.35" in result.stdout
    assert "# torch>=2" in result.stdout
    assert "# huggingface-hub" in result.stdout
    assert "\ntransformers>=4.35\n" not in result.stdout
    assert "# NVIDIA_API_KEY (optional; rails: jailbreak_detection)" in result.stdout


def test_requirements_file_output_preserves_markers_and_models():
    result = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "sensitive_data_detection", "--format", "requirements"],
    )

    assert result.exit_code == 0
    assert "presidio-analyzer>=2.2; python_version < '3.13'" in result.stdout
    assert "spacy>=3.4.4,<4,!=3.7.0; python_version < '3.13'" in result.stdout
    assert "# spacy:en_core_web_lg (required; rails: sensitive_data_detection)" in result.stdout


def test_requirements_human_output_quotes_marker_requirements_for_the_shell():
    result = runner.invoke(
        app,
        ["rails", "requirements", "--rail", "sensitive_data_detection"],
    )

    assert result.exit_code == 0
    assert "'presidio-analyzer>=2.2; python_version < \"3.13\"'" in result.stdout
    assert "'\"'\"'" not in result.stdout
