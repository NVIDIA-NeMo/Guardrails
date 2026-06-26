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
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.evaluate.evaluate_topical import (
    TopicalRailsEvaluation,
    _split_test_set_from_config,
    cosine_similarity,
    sync_wrapper,
)
from nemoguardrails.evaluate.utils import load_dataset


def test_load_dataset_reads_json_and_text(tmp_path):
    json_path = tmp_path / "data.json"
    text_path = tmp_path / "data.txt"
    json_path.write_text(json.dumps([{"question": "q"}]), encoding="utf-8")
    text_path.write_text("one\ntwo\n", encoding="utf-8")

    assert load_dataset(str(json_path)) == [{"question": "q"}]
    assert load_dataset(str(text_path)) == ["one\n", "two\n"]


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_sync_wrapper_runs_async_function():
    async def add(left, right):
        return left + right

    assert sync_wrapper(add)(2, 3) == 5


def test_split_test_set_from_config_uses_seed_and_limits_remaining_samples():
    config = SimpleNamespace(user_messages={"greet": ["a", "b", "c", "d"]})
    test_set = {}

    _split_test_set_from_config(
        config,
        test_set_percentage=0.5,
        test_set=test_set,
        max_samples_per_intent=1,
        random_seed=7,
    )

    assert len(test_set["greet"]) == 2
    assert len(config.user_messages["greet"]) == 1
    assert set(test_set["greet"]).isdisjoint(config.user_messages["greet"])


def test_split_test_set_ignores_single_sample_intents():
    config = SimpleNamespace(user_messages={"solo": ["only"]})
    test_set = {}

    _split_test_set_from_config(config, 0.5, test_set, 0)

    assert test_set == {}
    assert config.user_messages == {"solo": ["only"]}


def test_topical_init_initializes_rails_seed_and_embeddings(tmp_path):
    rails_config = SimpleNamespace(user_messages={"greet": ["a", "b"]}, flows=[], models=[])
    rails_app = SimpleNamespace(config=rails_config)

    with (
        patch(
            "nemoguardrails.evaluate.evaluate_topical.RailsConfig.from_path", return_value=rails_config
        ) as mock_config,
        patch("nemoguardrails.evaluate.evaluate_topical.LLMRails", return_value=rails_app) as mock_rails,
        patch("nemoguardrails.evaluate.evaluate_topical.random.seed") as mock_seed,
        patch.object(TopicalRailsEvaluation, "_initialize_embeddings_model") as mock_embeddings,
    ):
        evaluator = TopicalRailsEvaluation(
            config=str(tmp_path),
            verbose=True,
            test_set_percentage=0.5,
            max_tests_per_intent=1,
            max_samples_per_intent=1,
            similarity_threshold=0.0,
            random_seed=11,
        )

    mock_config.assert_called_once_with(config_path=str(tmp_path))
    mock_rails.assert_called_once_with(rails_config, verbose=True)
    mock_seed.assert_called_once_with(11)
    mock_embeddings.assert_called_once_with()
    assert evaluator.rails_app == rails_app
    assert len(evaluator.test_set["greet"]) == 1


def test_topical_initialize_embeddings_model_import_error(monkeypatch):
    evaluator = TopicalRailsEvaluation.__new__(TopicalRailsEvaluation)

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="sentence_transformers"):
        evaluator._initialize_embeddings_model()


def test_topical_initialize_embeddings_model_creates_model(monkeypatch):
    evaluator = TopicalRailsEvaluation.__new__(TopicalRailsEvaluation)
    evaluator.similarity_threshold = 0.5
    sentence_transformers = types.ModuleType("sentence_transformers")
    model_cls = MagicMock(return_value="model")
    sentence_transformers.SentenceTransformer = model_cls
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    evaluator._initialize_embeddings_model()

    model_cls.assert_called_once_with("all-MiniLM-L6-v2")
    assert evaluator._model == "model"


def test_topical_helper_methods_compute_embeddings_and_similarity():
    evaluator = TopicalRailsEvaluation.__new__(TopicalRailsEvaluation)
    evaluator.similarity_threshold = 0.5
    evaluator._model = SimpleNamespace(
        encode=lambda values: [[1.0, 0.0], [0.0, 1.0]] if isinstance(values, list) else [1.0, 0.0]
    )

    evaluator._compute_intent_embeddings(["greet", "bye"])

    assert evaluator._intent_embeddings == {"greet": [1.0, 0.0], "bye": [0.0, 1.0]}
    assert evaluator._get_most_similar_intent("hello") == "greet"


def test_topical_helper_methods_return_original_intent_without_model():
    evaluator = TopicalRailsEvaluation.__new__(TopicalRailsEvaluation)
    evaluator.similarity_threshold = 0.9
    evaluator._model = None

    evaluator._compute_intent_embeddings(["greet"])

    assert not hasattr(evaluator, "_intent_embeddings")
    assert evaluator._get_most_similar_intent("generated") == "generated"


def test_get_main_llm_model():
    evaluator = TopicalRailsEvaluation.__new__(TopicalRailsEvaluation)
    evaluator.rails_app = SimpleNamespace(
        config=SimpleNamespace(
            models=[
                SimpleNamespace(type="embedding", model="embed"),
                SimpleNamespace(type="main", model="main-model"),
            ]
        )
    )

    assert evaluator._get_main_llm_model() == "main-model"

    evaluator.rails_app.config.models = []
    assert evaluator._get_main_llm_model() == "unknown_main_llm"


def test_evaluate_topical_rails_runs_with_mock_runtime_and_writes_predictions(tmp_path):
    generate_events = AsyncMock(
        return_value=[
            {"type": "UserIntent", "intent": "wrong"},
            {"type": "BotIntent", "intent": "wrong bot"},
            {"type": "StartUtteranceBotAction", "script": "unexpected"},
        ]
    )
    evaluator = TopicalRailsEvaluation.__new__(TopicalRailsEvaluation)
    evaluator.config_path = str(tmp_path / "configs" / "topical")
    evaluator.test_set = {"greet": ["hello"]}
    evaluator.max_tests_per_intent = 1
    evaluator.max_samples_per_intent = 0
    evaluator.print_test_results_frequency = 1
    evaluator.similarity_threshold = 0.0
    evaluator.output_dir = str(tmp_path)
    evaluator._model = None
    evaluator.rails_app = SimpleNamespace(
        runtime=SimpleNamespace(generate_events=generate_events),
        config=SimpleNamespace(
            flows=[
                {
                    "elements": [
                        {"_type": "UserIntent", "intent_name": "greet"},
                        {
                            "_type": "run_action",
                            "action_name": "utter",
                            "action_params": {"value": "bot greet"},
                        },
                    ]
                }
            ],
            bot_messages={"bot greet": ["hello there"]},
            models=[SimpleNamespace(type="main", model="mock-model")],
        ),
    )

    evaluator.evaluate_topical_rails()

    generate_events.assert_awaited_once_with([{"type": "UtteranceUserActionFinished", "final_transcript": "hello"}])
    output_files = list(tmp_path.glob("*_topical_results.json"))
    assert len(output_files) == 1
    predictions = json.loads(output_files[0].read_text(encoding="utf-8"))
    assert predictions[0]["generated_user_intent"] == "wrong"
    assert predictions[0]["generated_bot_intent"] == "wrong bot"
