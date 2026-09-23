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
import logging
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemoguardrails.evaluate.cli import evaluate
from nemoguardrails.evaluate.cli.simplify_formatter import SimplifyFormatter
from nemoguardrails.evaluate.data.moderation import process_anthropic_dataset
from nemoguardrails.evaluate.data.topical.dataset_tools import (
    Banking77Connector,
    ChitChatConnector,
    DatasetConnector,
    Intent,
    IntentExample,
)

runner = CliRunner()


def test_topical_command_rejects_multiple_configs(tmp_path):
    config_a = tmp_path / "a"
    config_b = tmp_path / "b"
    config_a.mkdir()
    config_b.mkdir()

    result = runner.invoke(
        evaluate.app,
        ["topical", "--config", str(config_a), "--config", str(config_b)],
    )

    assert result.exit_code == 1
    assert "Multiple configurations are not supported" in result.stdout


def test_topical_command_invokes_evaluation_class(tmp_path):
    config = tmp_path / "config"
    output = tmp_path / "output"
    config.mkdir()
    topical_eval = MagicMock()

    with (
        patch("nemoguardrails.evaluate.cli.evaluate.TopicalRailsEvaluation", return_value=topical_eval) as mock_cls,
        patch("nemoguardrails.evaluate.cli.evaluate.set_verbose") as mock_set_verbose,
    ):
        result = runner.invoke(
            evaluate.app,
            [
                "topical",
                "--config",
                str(config),
                "--verbose",
                "--test-percentage",
                "0.4",
                "--max-tests-intent",
                "4",
                "--max-samples-intent",
                "2",
                "--results-frequency",
                "5",
                "--sim-threshold",
                "0.75",
                "--random-seed",
                "123",
                "--output-dir",
                str(output),
            ],
        )

    assert result.exit_code == 0
    mock_set_verbose.assert_called_once_with(True)
    mock_cls.assert_called_once_with(
        config=str(config),
        verbose=True,
        test_set_percentage=0.4,
        max_samples_per_intent=2,
        max_tests_per_intent=4,
        print_test_results_frequency=5,
        similarity_threshold=0.75,
        random_seed=123,
        output_dir=str(output),
    )
    topical_eval.evaluate_topical_rails.assert_called_once()


def test_rail_commands_invoke_evaluation_classes(tmp_path):
    config = tmp_path / "config"
    dataset = tmp_path / "dataset.txt"
    output = tmp_path / "output"
    config.mkdir()
    dataset.write_text("prompt\n", encoding="utf-8")

    cases = [
        (
            [
                "moderation",
                "--config",
                str(config),
                "--dataset-path",
                str(dataset),
                "--num-samples",
                "3",
                "--output-dir",
                str(output),
                "--split",
                "helpful",
            ],
            "ModerationRailsEvaluation",
            (str(config), str(dataset), 3, True, True, str(output), True, "helpful"),
        ),
        (
            [
                "hallucination",
                "--config",
                str(config),
                "--dataset-path",
                str(dataset),
                "--num-samples",
                "2",
                "--output-dir",
                str(output),
            ],
            "HallucinationRailsEvaluation",
            (str(config), str(dataset), 2, str(output), True),
        ),
        (
            [
                "fact-checking",
                "--config",
                str(config),
                "--dataset-path",
                str(dataset),
                "--num-samples",
                "4",
                "--output-dir",
                str(output),
            ],
            "FactCheckEvaluation",
            (str(config), str(dataset), 4, True, str(output), True),
        ),
    ]

    for args, class_name, expected_args in cases:
        instance = MagicMock()
        with patch(f"nemoguardrails.evaluate.cli.evaluate.{class_name}", return_value=instance) as mock_cls:
            result = runner.invoke(evaluate.app, args)

        assert result.exit_code == 0
        mock_cls.assert_called_once_with(*expected_args)
        instance.run.assert_called_once()


def test_process_anthropic_dataset_helpers(tmp_path, monkeypatch):
    harmful_path = tmp_path / "harmful.json"
    helpful_path = tmp_path / "helpful.jsonl"
    harmful_path.write_text(
        json.dumps(
            [
                {
                    "transcript": "\n\nHuman: unsafe\n\nAssistant: no",
                    "task_descripton_harmlessness_score": -2,
                },
                {
                    "transcript": "\n\nHuman: safe\n\nAssistant: ok",
                    "task_descripton_harmlessness_score": 0,
                },
            ]
        ),
        encoding="utf-8",
    )
    helpful_path.write_text(
        json.dumps({"chosen": "\n\nHuman: useful\n\nAssistant: ok"}) + "\n",
        encoding="utf-8",
    )

    assert process_anthropic_dataset.split_messages("\n\nHuman: hello\n\nAssistant: hi") == (
        ["hello"],
        ["hi"],
    )
    assert len(process_anthropic_dataset.load_dataset(str(harmful_path), "harmful")) == 2
    assert len(process_anthropic_dataset.load_dataset(str(helpful_path), "helpful")) == 1

    monkeypatch.chdir(tmp_path)
    process_anthropic_dataset.process_anthropic_harmful_data(str(harmful_path), rating=4.0)
    process_anthropic_dataset.process_anthropic_helpful_data(str(helpful_path))

    assert (tmp_path / "anthropic_harmful.txt").read_text(encoding="utf-8") == "unsafe\n"
    assert (tmp_path / "anthropic_helpful.txt").read_text(encoding="utf-8") == "useful\n"


def test_process_anthropic_dataset_main_dispatches(tmp_path):
    with (
        patch.object(process_anthropic_dataset, "process_anthropic_harmful_data") as mock_harmful,
        patch.object(process_anthropic_dataset, "process_anthropic_helpful_data") as mock_helpful,
    ):
        process_anthropic_dataset.main(dataset_path="data.json", rating=3.0, split="harmful")
        process_anthropic_dataset.main(dataset_path="data.jsonl", rating=3.0, split="helpful")

    mock_harmful.assert_called_once_with("data.json", 3.0)
    mock_helpful.assert_called_once_with("data.jsonl")


def test_simplify_formatter_masks_noisy_log_details():
    formatter = SimplifyFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=(
            "Process internal event: {'id': '123e4567-e89b-12d3-a456-426614174000', "
            "'_created_at': '2024-01-01T00:00:00.000000+00:00', "
            "'final_transcript': 'hello', 'loop_id': 'main-123'} <Foo object at 0xabc>"
        ),
        args=(),
        exc_info=None,
    )

    text = formatter.format(record)

    assert "123e..." in text
    assert "_created_at" not in text
    assert "<>" in text
    assert ":thumbs_up:'final_transcript': 'hello':thumbs_up:" in text
    assert "Process internal event" not in text
    assert formatter.format("Processing event details") == ""
    assert formatter.format("prefix :: hidden") == ""


def test_dataset_connector_sampling_and_colang_output(tmp_path):
    connector = DatasetConnector(name="test")
    intent = Intent(intent_name="greet", canonical_form="greet user")
    connector.intents.add(intent)
    connector.intent_examples = [
        IntentExample(intent=intent, text='say "hello"'),
        IntentExample(intent=intent, text="good morning"),
    ]

    sample = connector.get_intent_sample("greet", num_samples=1)
    assert len(sample) == 1
    assert sample[0] in {'say "hello"', "good morning"}

    output_path = tmp_path / "user.co"
    connector.write_colang_output(str(output_path), num_samples_per_intent=0)
    content = output_path.read_text(encoding="utf-8")
    assert "define user greet user" in content
    assert '"say hello"' in content


def test_dataset_connector_edge_cases(tmp_path):
    connector = DatasetConnector(name="test")
    with pytest.raises(NotImplementedError):
        connector.read_dataset("missing")

    assert connector.write_colang_output(None) is None

    no_canonical = Intent(intent_name="unclear")
    duplicate_a = Intent(intent_name="a", canonical_form="same canonical")
    duplicate_b = Intent(intent_name="b", canonical_form="same canonical")
    connector.intents.update([no_canonical, duplicate_a, duplicate_b])
    connector.intent_examples.extend(
        [
            IntentExample(intent=duplicate_a, text="sample a"),
            IntentExample(intent=duplicate_b, text="sample b"),
        ]
    )
    output_path = tmp_path / "duplicates.co"

    connector.write_colang_output(str(output_path), num_samples_per_intent=1)

    assert "define user same canonical" in output_path.read_text(encoding="utf-8")


def test_banking_connector_reads_canonical_forms_and_dataset(tmp_path, monkeypatch):
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps([["greet", "greet user"], ["bad"], ["bye", "say bye"]]), encoding="utf-8")

    assert Banking77Connector._read_canonical_forms(str(canonical_path)) == {
        "greet": "greet user",
        "bye": "say bye",
    }

    dataset_dir = tmp_path / "banking"
    dataset_dir.mkdir()
    (dataset_dir / "train.csv").write_text("text,category\nhello,greet\n", encoding="utf-8")
    (dataset_dir / "test.csv").write_text("goodbye,bye\nunknown,missing\n", encoding="utf-8")
    monkeypatch.setattr(
        Banking77Connector,
        "_read_canonical_forms",
        staticmethod(lambda: {"greet": "greet user", "bye": "say bye"}),
    )
    connector = Banking77Connector()

    connector.read_dataset(str(dataset_dir) + "/")

    assert Intent(intent_name="greet", canonical_form="greet user") in connector.intents
    assert Intent(intent_name="bye", canonical_form="say bye") in connector.intents
    assert Intent(intent_name="missing", canonical_form=None) in connector.intents
    assert [example.dataset_split for example in connector.intent_examples] == ["train", "test", "test"]


def test_chitchat_connector_reads_rasa_markdown(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "chitchat"
    dataset_dir.mkdir()
    (dataset_dir / "nlu.md").write_text("## intent:greet\n- hello\n- hi\n", encoding="utf-8")

    monkeypatch.setattr(ChitChatConnector, "_read_canonical_forms", staticmethod(lambda: {"greet": "greet user"}))
    connector = ChitChatConnector()
    connector.read_dataset(str(dataset_dir) + "/")

    assert connector.intents == {Intent(intent_name="greet", canonical_form="greet user")}
    assert [example.text for example in connector.intent_examples] == ["hello", "hi"]


def test_chitchat_connector_reads_canonical_forms(tmp_path):
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps([["greet", "greet user"], ["bad"], ["bye", "say bye"]]), encoding="utf-8")

    assert ChitChatConnector._read_canonical_forms(str(canonical_path)) == {
        "greet": "greet user",
        "bye": "say bye",
    }
