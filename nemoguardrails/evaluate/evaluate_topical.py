# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import os
import random
import textwrap
from typing import Dict, List, Optional

import numpy as np

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.llm.utils import (
    get_last_bot_intent_event,
    get_last_bot_utterance_event,
    get_last_user_intent_event,
)


def sync_wrapper(async_func):
    """Wrapper for the evaluate_topical_rails method which is async."""

    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(async_func(*args, **kwargs))

    return wrapper


def cosine_similarity(v1, v2):
    """Compute the dot product between two embeddings using numpy functions."""
    np_v1 = np.array(v1)
    np_v2 = np.array(v2)
    return np.dot(np_v1, np_v2) / (np.linalg.norm(np_v1) * np.linalg.norm(np_v2))


def _split_test_set_from_config(
    config: RailsConfig,
    test_set_percentage: float,
    test_set: Dict[str, List],
    max_samples_per_intent: int,
    random_seed: Optional[int] = None,
):
    """Extracts a test set of user messages from a config.

    Args:
        config: The config from which the test set will be extracted.
        test_set_percentage: The percentage used for the test set.
        test_set: A dictionary where the test set will be added.
        max_samples_per_intent: A limit on the number of samples per intent to be enforced.
    """
    if config.user_messages and test_set_percentage > 0:
        for intent, samples in config.user_messages.items():
            # We need at least 2 samples to create a test split
            if len(samples) > 1:
                if random_seed:
                    random.Random(random_seed).shuffle(samples)
                else:
                    random.shuffle(samples)

                num_test_elements = int(len(samples) * test_set_percentage)
                test_set[intent] = samples[:num_test_elements]
                config.user_messages[intent] = samples[num_test_elements:]

                # Limit the number of samples per intent if specified
                if 0 < max_samples_per_intent < len(config.user_messages[intent]):
                    config.user_messages[intent] = config.user_messages[intent][
                        :max_samples_per_intent
                    ]


class TopicalRailsEvaluation:
    """Helper class for running the topical rails evaluation for a Guardrails app.
    It contains all the configuration parameters required to run the evaluation."""

    def _initialize_rails_app(self):
        self.test_set = {}
        rails_config = RailsConfig.from_path(
            config_path=self.config_path,
        )
        _split_test_set_from_config(
            rails_config,
            test_set_percentage=self.test_set_percentage,
            max_samples_per_intent=self.max_samples_per_intent,
            test_set=self.test_set,
        )
        """Initializes the Rails app used for evaluation."""

        # TODO: add support to register additional actions
        # rails_app.register_action(...)

        self.rails_app = LLMRails(rails_config, verbose=self.verbose)

    def _initialize_embeddings_model(self):
        """Instantiate a sentence transformer if we use a similarity check for canonical forms."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers, please install it with "
                "`pip install sentence-transformers`."
            )

        self._model = None
        if self.similarity_threshold > 0:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _initialize_random_seed(self):
        """Initialize random seed"""
        if self.random_seed:
            random.seed(self.random_seed)

    def _compute_intent_embeddings(self, intents):
        """Compute intent embeddings if we have a sentence transformer model."""
        if not self._model:
            return
        self._intent_embeddings = {}
        embeddings = self._model.encode(intents)
        for i, intent in enumerate(intents):
            self._intent_embeddings[intent] = embeddings[i]

    def _get_most_similar_intent(self, generated_intent):
        """Retrieves the most similar intent using sentence transformers embeddings.
        If the most similar intent is below the similarity threshold,
        the generated intent is not changed."""
        if not self._model or self.similarity_threshold <= 0:
            return generated_intent

        generated_intent_embeddings = self._model.encode(generated_intent)

        max_similarity = 0
        max_intent = None
        for intent, embedding in self._intent_embeddings.items():
            similarity = cosine_similarity(embedding, generated_intent_embeddings)
            if similarity > max_similarity and similarity > self.similarity_threshold:
                max_similarity = similarity
                max_intent = intent

        return max_intent or generated_intent

    def _get_main_llm_model(self):
        for model in self.rails_app.config.models:
            if model.type == "main":
                return model.model if model.model else model.type
        return "unknown_main_llm"

    @staticmethod
    def _print_evaluation_results(
        processed_samples,
        total_test_samples,
        num_user_intent_errors,
        num_bot_intent_errors,
        num_bot_utterance_errors,
    ):
        """Prints a summary of the evaluation results."""
        print(
            textwrap.dedent(
                f"Processed {processed_samples}/{total_test_samples} samples! "
                f"Num intent errors: {num_user_intent_errors}. "
                f"Num bot intent errors {num_bot_intent_errors}. "
                f"Num bot message errors {num_bot_utterance_errors}."
            )
        )

    def __init__(
        self,
        config: str,
        verbose: Optional[bool] = False,
        test_set_percentage: Optional[float] = 0.3,
        max_tests_per_intent: Optional[int] = 3,
        max_samples_per_intent: Optional[int] = 0,
        print_test_results_frequency: Optional[int] = 10,
        similarity_threshold: Optional[float] = 0.0,
        random_seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ):
        """A topical rails evaluation has the following parameters:

        - config_path: The Guardrails app to be evaluated.
        - verbose: If the Guardrails app should be run in verbose mode
        - test_set_percentage: Percentage of the samples for an intent to be used as test set
        - max_tests_per_intent: Maximum number of test samples per intent to be used when testing
        (useful to have balanced test data for unbalanced datasets). If the value is 0,
        this parameter is not used.
        - max_samples_per_intent: Maximum number of samples per intent to be used in the
        vector database. If the value is 0, all samples not in test set are used.
        - print_test_results_frequency: If we want to print intermediate results about the
        current evaluation, this is the step.
        - similarity_threshold: If larger than 0, for intents that do not have an exact match
        pick the most similar intent above this threshold.
        - random_seed: Random seed used by the evaluation.
        - output_dir: Output directory for predictions.
        """
        self.config_path = config
        self.verbose = verbose
        self.test_set_percentage = test_set_percentage
        self.max_tests_per_intent = max_tests_per_intent
        self.max_samples_per_intent = max_samples_per_intent
        self.print_test_results_frequency = print_test_results_frequency
        self.similarity_threshold = similarity_threshold
        self.random_seed = random_seed
        self.output_dir = output_dir

        self._initialize_random_seed()
        self._initialize_rails_app()
        self._initialize_embeddings_model()

    @sync_wrapper
    async def evaluate_topical_rails(self):
        """Runs the topical evaluation for the Guardrails app with the current configuration."""

        # Find the intents that do not have a flow that matches them
        intents_with_flows = {}
        for flow in self.rails_app.config.flows:
            intent_next_actions = None
            for event in flow["elements"]:
                if event["_type"] == "UserIntent":
                    intent_name = event["intent_name"]
                    if intent_name in intents_with_flows:
                        print(intent_name)
                    intent_next_actions = intents_with_flows.get(intent_name, [])
                    if intent_name not in intents_with_flows:
                        intents_with_flows[intent_name] = intent_next_actions
                elif event["_type"] == "run_action" and event["action_name"] == "utter":
                    if intent_next_actions is not None:
                        intent_next_actions.append(event["action_params"]["value"])

        num_intents_with_flows = len(
            set(self.test_set.keys()).intersection(intents_with_flows.keys())
        )

        # Compute the embeddings for each intent if needed
        self._compute_intent_embeddings(list(self.test_set.keys()))

        # Limit the number of test samples per intent, if we want to have a balanced test set
        total_test_samples = 0
        for intent in self.test_set.keys():
            samples = self.test_set[intent]
            if 0 < self.max_tests_per_intent < len(samples):
                samples = samples[: self.max_tests_per_intent]
                self.test_set[intent] = samples
            total_test_samples += len(samples)

        print(
            textwrap.dedent(
                f"""Started processing rails app from path: {self.config_path}.
                Number of intents: {len(self.test_set.keys())}.
                Number of flows: {len(self.rails_app.config.flows)}.
                Number of test samples: {total_test_samples}.
                Number of intents that have an associated flow: {num_intents_with_flows}.
                Intents without associated flows: {set(self.test_set.keys()).difference(intents_with_flows.keys())}."""
            )
        )

        # Run evaluation experiment, for each test sample start a new conversation
        processed_samples = 0
        num_user_intent_errors = 0
        num_bot_intent_errors = 0
        num_bot_utterance_errors = 0
        topical_predictions = []
        num_user_intent_errors_from_empty_intent = 0

        for intent, samples in self.test_set.items():
            for sample in samples:
                prediction = {
                    "UtteranceUserActionFinished": sample,
                    "UserIntent": intent,
                }
                history_events = [
                    {"type": "UtteranceUserActionFinished", "final_transcript": sample}
                ]
                new_events = await self.rails_app.runtime.generate_events(
                    history_events
                )

                generated_user_intent = None
                last_user_intent_event = get_last_user_intent_event(new_events)
                if last_user_intent_event is not None:
                    generated_user_intent = last_user_intent_event["intent"]
                    prediction["generated_user_intent"] = generated_user_intent
                    wrong_intent = False
                if generated_user_intent is None:
                    num_user_intent_errors_from_empty_intent += 1
                    print("Error!: Generated empty user intent")
                if generated_user_intent is None or generated_user_intent != intent:
                    wrong_intent = True
                    # Employ semantic similarity if needed
                    if (
                        generated_user_intent is not None
                        and self.similarity_threshold > 0
                    ):
                        sim_user_intent = self._get_most_similar_intent(
                            generated_user_intent
                        )
                        prediction["sim_user_intent"] = sim_user_intent
                        if sim_user_intent == intent:
                            wrong_intent = False

                    if wrong_intent:
                        num_user_intent_errors += 1
                        if self.similarity_threshold > 0:
                            print(
                                f"Error!: Generated intent: {generated_user_intent} ; "
                                f"Most similar intent: {sim_user_intent} <> "
                                f"Expected intent: {intent}"
                            )
                        else:
                            print(
                                f"Error!: Generated intent: {generated_user_intent} <> "
                                f"Expected intent: {intent}"
                            )

                # If the intent is correct, the generated bot intent and bot message
                # are also correct. For user intent similarity check,
                # the bot intent (next step) and bot message may appear different in
                # the verbose logs as they are generated using the generated user intent,
                # before applying similarity checking.
                if wrong_intent:
                    generated_bot_intent = get_last_bot_intent_event(new_events)[
                        "intent"
                    ]
                    prediction["generated_bot_intent"] = generated_bot_intent
                    prediction["bot_intents"] = intents_with_flows[intent]
                    if generated_bot_intent not in intents_with_flows[intent]:
                        num_bot_intent_errors += 1
                        print(
                            f"Error!: Generated bot intent: {generated_bot_intent} <> "
                            f"Expected bot intent: {intents_with_flows[intent]}"
                        )

                    generated_bot_utterance = get_last_bot_utterance_event(new_events)[
                        "script"
                    ]
                    prediction["generated_bot_said"] = generated_bot_utterance
                    found_utterance = False
                    found_bot_message = False
                    for bot_intent in intents_with_flows[intent]:
                        bot_messages = self.rails_app.config.bot_messages
                        if bot_intent in bot_messages:
                            found_bot_message = True
                            if generated_bot_utterance in bot_messages[bot_intent]:
                                found_utterance = True
                    if found_bot_message and not found_utterance:
                        prediction["bot_said"] = bot_messages[bot_intent]
                        num_bot_utterance_errors += 1
                        print(
                            f"Error!: Generated bot message: {generated_bot_utterance} <> "
                            f"Expected bot message: {bot_messages[bot_intent]}"
                        )

                topical_predictions.append(prediction)
                processed_samples += 1
                if (
                    self.print_test_results_frequency
                    and processed_samples % self.print_test_results_frequency == 0
                ):
                    TopicalRailsEvaluation._print_evaluation_results(
                        processed_samples,
                        total_test_samples,
                        num_user_intent_errors,
                        num_bot_intent_errors,
                        num_bot_utterance_errors,
                    )

        TopicalRailsEvaluation._print_evaluation_results(
            processed_samples,
            total_test_samples,
            num_user_intent_errors,
            num_bot_intent_errors,
            num_bot_utterance_errors,
        )

        if self.output_dir:
            # Extract filename from config path (use last 2 directory names if possible)
            filename = "default"
            words = self.config_path.split(os.path.sep)
            if len(words) > 2:
                filename = "_".join(words[-2:])
            elif len(words) == 1:
                filename = words[0]

            model_name = self._get_main_llm_model()
            filename += (
                f"_{model_name}_shots{self.max_samples_per_intent}"
                f"_sim{self.similarity_threshold}"
                f"_topical_results.json"
            )

            output_path = f"{self.output_dir}/{filename}"
            with open(output_path, "w") as f:
                json.dump(topical_predictions, f, indent=4)

                print(f"Predictions written to file {output_path}")
