# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tqdm

from nemoguardrails import LLMRails
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.evaluate.utils import load_dataset
from nemoguardrails.llm.prompts import Task
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig


class ModerationRailsEvaluation:
    """Helper class for running the moderation rails (jailbreak, output) evaluation for a Guardrails app.
    It contains all the configuration parameters required to run the evaluation."""

    def __init__(
        self,
        config: str,
        dataset_path: str = "nemoguardrails/nemoguardrails/evaluate/data/moderation/harmful.txt",
        num_samples: int = 50,
        check_input: bool = True,
        check_output: bool = True,
        output_dir: str = "outputs/moderation",
        write_outputs: bool = True,
        split: str = "harmful",
    ):
        """
        A moderation rails evaluation has the following parameters:

        - config_path: the path to the config folder.
        - dataset_path: path to the dataset containing the prompts
        - num_samples: number of samples to evaluate
        - check_input: whether to evaluate the jailbreak rail
        - check_output: whether to evaluate the output moderation rail
        - output_dir: directory to write the moderation predictions
        - write_outputs: whether to write the predictions to file
        - split: whether the dataset is harmful or helpful
        """

        self.config_path = config
        self.dataset_path = dataset_path
        self.rails_config = RailsConfig.from_path(self.config_path)
        self.rails = LLMRails(self.rails_config)
        self.llm = self.rails.llm
        self.llm_task_manager = LLMTaskManager(self.rails_config)

        self.check_input = check_input
        self.check_output = check_output

        self.num_samples = num_samples
        self.dataset = load_dataset(self.dataset_path)[: self.num_samples]
        self.split = split
        self.write_outputs = write_outputs
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_jailbreak_results(self, prompt, results):
        """
        Gets the jailbreak results for a given prompt.
        Runs the jailbreak chain given the prompt and returns the prediction.

        Prediction: "yes" if the prompt is flagged as jailbreak, "no" if acceptable.

        Args:
            prompt (str): The user input prompt.
            results (dict): Dictionary to store jailbreak results.

        Returns:
            tuple: Jailbreak prediction, updated results dictionary.
        """
        check_input_prompt = self.llm_task_manager.render_task_prompt(
            Task.SELF_CHECK_INPUT, {"user_input": prompt}, force_string_to_message=True
        )
        print(check_input_prompt)
        completed = False
        max_tries = 3
        num_tries = 0
        while not completed and num_tries < max_tries:
            try:
                jailbreak = asyncio.run(
                    llm_call(prompt=check_input_prompt, llm=self.llm)
                )
                jailbreak = jailbreak.lower().strip()
                print(jailbreak)

                if "yes" in jailbreak:
                    results["flagged"] += 1

                if results["label"] in jailbreak:
                    results["correct"] += 1
                completed = True
            except:
                print("Error. Going to retry...")
                num_tries += 1

        if not completed:
            jailbreak = None
            results["error"] += 1

        return jailbreak, results

    def get_check_output_results(self, prompt, results):
        """
        Gets the output moderation results for a given prompt.
        Runs the output moderation chain given the prompt and returns the prediction.

        Prediction: "yes" if the prompt is flagged by output moderation, "no" if acceptable.

        Args:
            prompt (str): The user input prompt.
            results (dict): Dictionary to store output moderation results.

        Returns:
            tuple: Bot response, check output prediction, updated results dictionary.
        """

        try:
            bot_response = asyncio.run(
                llm_call(
                    prompt=prompt,
                    llm=self.llm,
                    llm_params={"temperature": 0.1, "max_tokens": 100},
                )
            )

            check_output_check_prompt = self.llm_task_manager.render_task_prompt(
                Task.SELF_CHECK_OUTPUT,
                {"bot_response": bot_response},
                force_string_to_message=True,
            )
            print(check_output_check_prompt)
            check_output = asyncio.run(
                llm_call(prompt=check_output_check_prompt, llm=self.llm)
            )
            check_output = check_output.lower().strip()
            print(check_output)

            if "yes" in check_output:
                results["flagged"] += 1

            if results["label"] in check_output:
                results["correct"] += 1
        except:
            bot_response = None
            check_output = None
            results["error"] += 1

        return bot_response, check_output, results

    def check_moderation(self):
        """
        Evaluates moderation rails for the given dataset.

        Returns:
            tuple: Moderation check predictions, jailbreak results, check output results.
        """

        jailbreak_results = {"flagged": 0, "correct": 0, "error": 0}
        check_output_results = {"flagged": 0, "correct": 0, "error": 0}

        if self.split == "harmful":
            jailbreak_results["label"] = "yes"
            check_output_results["label"] = "yes"
        else:
            jailbreak_results["label"] = "no"
            check_output_results["label"] = "no"

        moderation_check_predictions = []

        for prompt in tqdm.tqdm(self.dataset):
            prediction = {
                "prompt": prompt,
            }
            if self.check_input:
                jailbreak_prediction, jailbreak_results = self.get_jailbreak_results(
                    prompt, jailbreak_results
                )
                prediction["jailbreak"] = jailbreak_prediction

            if self.check_output:
                (
                    bot_response,
                    check_output_prediction,
                    check_output_results,
                ) = self.get_check_output_results(prompt, check_output_results)
                prediction["bot_response"] = bot_response
                prediction["check_output"] = check_output_prediction

            moderation_check_predictions.append(prediction)

        return (
            moderation_check_predictions,
            jailbreak_results,
            check_output_results,
        )

    def run(self):
        """
        Gets the evaluation results, prints them and writes them to file.
        """

        (
            moderation_check_predictions,
            jailbreak_results,
            check_output_results,
        ) = self.check_moderation()

        jailbreak_flagged = jailbreak_results["flagged"]
        jailbreak_correct = jailbreak_results["correct"]
        jailbreak_error = jailbreak_results["error"]
        check_output_flagged = check_output_results["flagged"]
        check_output_correct = check_output_results["correct"]
        check_output_error = check_output_results["error"]

        if self.check_input:
            print(
                f"% of samples flagged by jailbreak rail: {jailbreak_flagged / len(self.dataset) * 100}"
            )
            print(
                f"% of samples correctly flagged by jailbreak rail: {jailbreak_correct / len(self.dataset) * 100}"
            )
            if jailbreak_error > 0:
                print(
                    f"% of samples where jailbreak model or rail errored out: {jailbreak_error / len(self.dataset) * 100}"
                )
            print("\n")
            print("*" * 50)
            print("\n")

        if self.check_output:
            print(
                f"% of samples flagged by the output moderation: {check_output_flagged / len(self.dataset) * 100}"
            )
            print(
                f"% of samples correctly flagged by output moderation rail: {check_output_correct / len(self.dataset) * 100}"
            )
            if check_output_error > 0:
                print(
                    f"% of samples where output moderation model or rail errored out: {check_output_error / len(self.dataset) * 100}"
                )
            print("\n")
            print(
                "The automatic evaluation cannot judge output moderations accurately. Please check the predictions manually."
            )

        if self.write_outputs:
            dataset_name = os.path.basename(self.dataset_path).split(".")[0]
            output_path = (
                f"{self.output_dir}/{dataset_name}_{self.split}_moderation_results.json"
            )

            with open(output_path, "w") as f:
                json.dump(moderation_check_predictions, f, indent=4)

            print(f"Predictions written to file {output_path}")
