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

import json
import logging
import os
from logging import log

import tqdm
import typer

from nemoguardrails import LLMRails
from nemoguardrails.evaluate.utils import load_dataset
from nemoguardrails.llm.prompts import Task
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig


class HallucinationRailsEvaluation:
    """Helper class for running the hallucination rails evaluation for a Guardrails app.
    It contains all the configuration parameters required to run the evaluation."""

    def __init__(
        self,
        config: str,
        dataset_path: str = "data/hallucination/sample.txt",
        num_samples: int = 50,
        output_dir: str = "outputs/hallucination",
        write_outputs: bool = True,
    ):
        """
        A hallucination rails evaluation has the following parameters:
        - config_path: the path to the config folder.
        - dataset_path: path to the dataset containing the prompts
        - llm: the LLM provider to use
        - model_name: the LLM model to use
        - num_samples: number of samples to evaluate
        - output_dir: directory to write the hallucination predictions
        - write_outputs: whether to write the predictions to file
        """

        self.config_path = config
        self.dataset_path = dataset_path
        self.rails_config = RailsConfig.from_path(self.config_path)
        self.rails = LLMRails(self.rails_config)
        self.llm = self.rails.llm
        self.llm_task_manager = LLMTaskManager(self.rails_config)

        self.num_samples = num_samples
        self.dataset = load_dataset(self.dataset_path)[: self.num_samples]
        self.write_outputs = write_outputs
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_response_with_retries(self, prompt, max_tries=1, llm_params=None):
        num_tries = 0
        while num_tries < max_tries:
            try:
                if llm_params:
                    llm_configured = self.llm.bind(**llm_params)
                    response = llm_configured(prompt)
                else:
                    response = self.llm(prompt)
                return response
            except:
                num_tries += 1
        return None

    def get_extra_responses(self, prompt, num_responses=2):
        """
        Sample extra responses with temperature=1.0 from the LLM for hallucination check.

        Args:
            prompt (str): The prompt to generate extra responses for.
            num_responses (int): Number of extra responses to generate.

        Returns:
            List[str]: The list of extra responses.
        """
        extra_responses = []
        for _ in range(num_responses):
            extra_response = self.get_response_with_retries(
                prompt, llm_params={"temperature": 1.0, "max_tokens": 100}
            )

            if extra_response is None:
                log(
                    logging.WARNING,
                    f"LLM produced an error generating extra response for question '{prompt}'.",
                )
            else:
                extra_responses.append(extra_response)

        return extra_responses

    def self_check_hallucination(self):
        """
        Run the hallucination rail evaluation.
        For each prompt, generate 2 extra responses from the LLM and check consistency with the bot response.
        If inconsistency is detected, flag the prompt as hallucination.

        Returns:
            Tuple[List[HallucinationPrediction], int]: Tuple containing hallucination predictions and the number flagged.
        """

        hallucination_check_predictions = []
        num_flagged = 0
        num_error = 0

        for question in tqdm.tqdm(self.dataset):
            errored_out = False
            # Using temperature=0.2 and max_tokens=100 for consistent responses
            bot_response = self.get_response_with_retries(
                question, llm_params={"temperature": 0.2, "max_tokens": 100}
            )

            if bot_response is None:
                log(
                    logging.WARNING,
                    f"LLM produced an error for question '{question}'.",
                )
                extra_responses = None
                errored_out = True
            else:
                extra_responses = self.get_extra_responses(question, num_responses=2)
                if len(extra_responses) == 0:
                    # Log message and return that no hallucination was found
                    log(
                        logging.WARNING,
                        f"No extra LLM responses were generated for '{bot_response}' hallucination check.",
                    )
                    errored_out = True

            if errored_out:
                num_error += 1
                prediction = {
                    "question": question,
                    "hallucination_agreement": "na",
                    "bot_response": bot_response,
                    "extra_responses": extra_responses,
                }
                hallucination_check_predictions.append(prediction)
            else:
                paragraph = ". ".join(extra_responses)
                hallucination_check_prompt = self.llm_task_manager.render_task_prompt(
                    Task.SELF_CHECK_HALLUCINATION,
                    {"paragraph": paragraph, "statement": bot_response},
                )
                hallucination = self.llm(hallucination_check_prompt)
                hallucination = hallucination.lower().strip()

                prediction = {
                    "question": question,
                    "hallucination_agreement": hallucination,
                    "bot_response": bot_response,
                    "extra_responses": extra_responses,
                }
                hallucination_check_predictions.append(prediction)
                if "no" in hallucination:
                    num_flagged += 1

        return hallucination_check_predictions, num_flagged, num_error

    def run(self):
        """
        Run  and print the hallucination rail evaluation.
        """

        (
            hallucination_check_predictions,
            num_flagged,
            num_error,
        ) = self.self_check_hallucination()
        print(
            f"% of samples flagged as hallucinations: {num_flagged / len(self.dataset) * 100}"
        )
        print(
            f"% of samples where model errored out: {num_error / len(self.dataset) * 100}"
        )
        print(
            "The automatic evaluation cannot catch predictions that are not hallucinations. Please check the predictions manually."
        )

        if self.write_outputs:
            dataset_name = os.path.basename(self.dataset_path).split(".")[0]
            output_path = (
                f"{self.output_dir}/{dataset_name}_hallucination_predictions.json"
            )
            with open(output_path, "w") as f:
                json.dump(hallucination_check_predictions, f, indent=4)
            print(f"Predictions written to file {output_path}.json")


def main(
    config: str,
    data_path: str = typer.Option("data/hallucination/sample.txt", help="Dataset path"),
    num_samples: int = typer.Option(50, help="Number of samples to evaluate"),
    output_dir: str = typer.Option("outputs/hallucination", help="Output directory"),
    write_outputs: bool = typer.Option(True, help="Write outputs to file"),
):
    """
    Main function to run the hallucination rails evaluation.

    Args:
        config (str): The path to the config folder.
        data_path (str): Dataset path.
        num_samples (int): Number of samples to evaluate.
        output_dir (str): Output directory for predictions.
        write_outputs (bool): Whether to write the predictions to a file.
    """
    hallucination_check = HallucinationRailsEvaluation(
        config,
        data_path,
        num_samples,
        output_dir,
        write_outputs,
    )
    hallucination_check.run()


if __name__ == "__main__":
    typer.run(main)
