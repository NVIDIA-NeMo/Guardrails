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
import time

import tqdm
import typer
from langchain_core.prompts import PromptTemplate

from nemoguardrails import LLMRails
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.evaluate.utils import load_dataset
from nemoguardrails.llm.prompts import Task
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig


class FactCheckEvaluation:
    """Helper class for running the fact checking evaluation for a Guardrails app.
    It contains all the configuration parameters required to run the evaluation."""

    def __init__(
        self,
        config: str,
        dataset_path: str = "data/factchecking/sample.json",
        num_samples: int = 50,
        create_negatives: bool = True,
        output_dir: str = "outputs/factchecking",
        write_outputs: bool = True,
    ):
        """
        A fact checking evaluation has the following parameters:
        - config_path: the path to the config folder.
        - dataset_path: path to the dataset containing the prompts
        - llm: the LLM provider to use
        - model_name: the LLM model to use
        - num_samples: number of samples to evaluate
        - create_negatives: whether to create synthetic negative samples
        - output_dir: directory to write the fact checking predictions
        - write_outputs: whether to write the predictions to file
        """

        self.config_path = config
        self.dataset_path = dataset_path
        self.rails_config = RailsConfig.from_path(self.config_path)
        self.rails = LLMRails(self.rails_config)
        self.llm = self.rails.llm
        self.llm_task_manager = LLMTaskManager(self.rails_config)

        self.create_negatives = create_negatives
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.dataset = load_dataset(self.dataset_path)[: self.num_samples]
        self.write_outputs = write_outputs

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_negative_samples(self, dataset):
        """
        Create synthetic negative samples for fact checking. The negative samples are created by an LLM that acts
        as an adversary and modifies the answer to make it incorrect.

        Args:
            dataset (List[Dict]): The dataset to create negative samples for.

        Returns:
            List[Dict]: The dataset with synthetic negative samples.
        """

        create_negatives_template = """You will play the role of an adversary to confuse people with answers
        that seem correct, but are wrong. Given evidence and a question, your task is to respond with an
        answer that remains as close to the original answer, but is wrong. make the response incorrect such
        that it will not be grounded in the evidence passage. change details in the answer to make the answer
        wrong but yet believable.\nevidence: {evidence}\nanswer: {answer}\nincorrect answer:"""

        create_negatives_prompt = PromptTemplate(
            template=create_negatives_template,
            input_variables=["evidence", "answer"],
        )

        # Bind config parameters to the LLM for generating negative samples
        llm_with_config = self.llm.bind(temperature=0.8, max_tokens=300)

        print("Creating negative samples...")
        for data in tqdm.tqdm(dataset):
            assert "evidence" in data and "question" in data and "answer" in data
            evidence = data["evidence"]
            answer = data["answer"]

            # Format the prompt and invoke the LLM directly
            formatted_prompt = create_negatives_prompt.format(evidence=evidence, answer=answer)
            negative_answer = llm_with_config.invoke(formatted_prompt)
            if isinstance(negative_answer, str):
                data["incorrect_answer"] = negative_answer.strip()
            else:
                data["incorrect_answer"] = negative_answer.content.strip()

        return dataset

    def check_facts(self, split="positive"):
        """
        Check facts using the fact checking rail. The fact checking rail is a binary classifier that takes in
        evidence and a response and predicts whether the response is grounded in the evidence or not.

        Args:
            split (str): The split type for checking facts. Either "positive" or "negative".

        Returns:
            Tuple[List[FactCheckPrediction], int, float]: Tuple containing fact check predictions,
            number of correct predictions, and total time taken.
        """

        fact_check_predictions = []
        num_correct = 0
        total_time = 0

        for sample in tqdm.tqdm(self.dataset):
            assert "evidence" in sample and "answer" in sample and "incorrect_answer" in sample
            evidence = sample["evidence"]
            if split == "positive":
                answer = sample["answer"]
                label = "yes"
            else:
                answer = sample["incorrect_answer"]
                label = "no"

            start_time = time.time()
            fact_check_prompt = self.llm_task_manager.render_task_prompt(
                Task.SELF_CHECK_FACTS,
                {"evidence": evidence, "response": answer},
                force_string_to_message=True,
            )
            stop = self.llm_task_manager.get_stop_tokens(Task.SELF_CHECK_FACTS)
            fact_check = asyncio.run(llm_call(prompt=fact_check_prompt, llm=self.llm, stop=stop))
            end_time = time.time()
            time.sleep(0.5)  # avoid rate-limits
            fact_check = fact_check.lower().strip()

            if label in fact_check:
                num_correct += 1

            prediction = {
                "question": sample["question"],
                "evidence": evidence,
                "answer": answer,
                "fact_check": fact_check,
                "label": label,
            }
            fact_check_predictions.append(prediction)
            total_time += end_time - start_time

        return fact_check_predictions, num_correct, total_time

    def run(self):
        """
        Run the fact checking evaluation and print the results.
        """
        if self.create_negatives:
            self.dataset = self.create_negative_samples(self.dataset)

        print("Checking facts - positive entailment")
        positive_fact_check_predictions, pos_num_correct, pos_time = self.check_facts(split="positive")
        print("Checking facts - negative entailment")
        negative_fact_check_predictions, neg_num_correct, neg_time = self.check_facts(split="negative")

        print(f"Positive Accuracy: {pos_num_correct / len(self.dataset) * 100}")
        print(f"Negative Accuracy: {neg_num_correct / len(self.dataset) * 100}")
        print(f"Overall Accuracy: {(pos_num_correct + neg_num_correct) / (2 * len(self.dataset)) * 100}")

        print("---Time taken per sample:---")
        print(f"Ask LLM:\t{(pos_time + neg_time) * 1000 / (2 * len(self.dataset)):.1f}ms")

        if self.write_outputs:
            dataset_name = os.path.basename(self.dataset_path).split(".")[0]
            with open(
                f"{self.output_dir}/{dataset_name}_positive_fact_check_predictions.json",
                "w",
            ) as f:
                json.dump(positive_fact_check_predictions, f, indent=4)

            with open(
                f"{self.output_dir}/{dataset_name}_negative_fact_check_predictions.json",
                "w",
            ) as f:
                json.dump(negative_fact_check_predictions, f, indent=4)


def main(
    config: str,
    dataset_path: str = typer.Option(
        "./data/factchecking/sample.json",
        help="Path to the folder containing the dataset",
    ),
    num_samples: int = typer.Option(50, help="Number of samples to be evaluated"),
    create_negatives: bool = typer.Option(True, help="create synthetic negative samples"),
    output_dir: str = typer.Option(
        "eval_outputs/factchecking",
        help="Path to the folder where the outputs will be written",
    ),
    write_outputs: bool = typer.Option(True, help="Write outputs to the output directory"),
):
    fact_check = FactCheckEvaluation(
        config,
        dataset_path,
        num_samples,
        create_negatives,
        output_dir,
        write_outputs,
    )
    fact_check.run()


if __name__ == "__main__":
    typer.run(main)
