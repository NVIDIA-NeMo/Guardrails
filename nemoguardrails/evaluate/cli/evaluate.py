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

import logging
from typing import List

import typer

from nemoguardrails.evaluate.evaluate_factcheck import FactCheckEvaluation
from nemoguardrails.evaluate.evaluate_hallucination import HallucinationRailsEvaluation
from nemoguardrails.evaluate.evaluate_moderation import ModerationRailsEvaluation
from nemoguardrails.evaluate.evaluate_topical import TopicalRailsEvaluation
from nemoguardrails.logging.verbose import set_verbose

app = typer.Typer()

logging.getLogger().setLevel(logging.WARNING)


@app.command()
def topical(
    config: List[str] = typer.Option(
        default=[""],
        exists=True,
        help="Path to a directory containing configuration files of the Guardrails application for evaluation. "
        "Can also point to a single configuration file.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="If the chat should be verbose and output the prompts.",
    ),
    test_percentage: float = typer.Option(
        default=0.3,
        help="Percentage of the samples for an intent to be used as test set.",
    ),
    max_tests_intent: int = typer.Option(
        default=3,
        help="Maximum number of test samples per intent to be used when testing. If value is 0, no limit is used.",
    ),
    max_samples_intent: int = typer.Option(
        default=0,
        help="Maximum number of samples per intent indexed in vector database. If value is 0, all samples are used.",
    ),
    results_frequency: int = typer.Option(
        default=10,
        help="Print evaluation intermediate results using this step.",
    ),
    sim_threshold: float = typer.Option(
        default=0.0,
        help="Minimum similarity score to select the intent when exact match fails.",
    ),
    random_seed: int = typer.Option(default=None, help="Random seed used by the evaluation."),
    output_dir: str = typer.Option(default=None, help="Output directory for predictions."),
):
    """Evaluates the performance of the topical rails defined in a Guardrails application.
    Computes accuracy for canonical form detection, next step generation, and next bot message generation.
    Only a single Guardrails application can be specified in the config option.

    Args:
        config (List[str], optional): Path to a directory containing configuration files of the Guardrails application for evaluation.
            Can also point to a single configuration file. Defaults to [""].
        verbose (bool, optional): If the chat should be verbose and output the prompts. Defaults to False.
        test_percentage (float, optional): Percentage of the samples for an intent to be used as test set. Defaults to 0.3.
        max_tests_intent (int, optional): Maximum number of test samples per intent to be used when testing.
            If value is 0, no limit is used. Defaults to 3.
        max_samples_intent (int, optional): Maximum number of samples per intent indexed in vector database.
            If value is 0, all samples are used. Defaults to 0.
        results_frequency (int, optional): Print evaluation intermediate results using this step. Defaults to 10.
        sim_threshold (float, optional): Minimum similarity score to select the intent when exact match fails. Defaults to 0.0.
        random_seed (int, optional): Random seed used by the evaluation. Defaults to None.
        output_dir (str, optional): Output directory for predictions. Defaults to None.
    """
    if verbose:
        set_verbose(True)

    if len(config) > 1:
        typer.secho("Multiple configurations are not supported.", fg=typer.colors.RED)
        typer.echo("Please provide a single config path (folder or config file).")
        raise typer.Exit(1)

    if config[0] == "":
        typer.echo("Please provide a value for the config path.")
        raise typer.Exit(1)

    typer.echo(f"Starting the evaluation for app: {config[0]}...")

    topical_eval = TopicalRailsEvaluation(
        config=config[0],
        verbose=verbose,
        test_set_percentage=test_percentage,
        max_samples_per_intent=max_samples_intent,
        max_tests_per_intent=max_tests_intent,
        print_test_results_frequency=results_frequency,
        similarity_threshold=sim_threshold,
        random_seed=random_seed,
        output_dir=output_dir,
    )
    topical_eval.evaluate_topical_rails()


@app.command()
def moderation(
    config: str = typer.Option(help="The path to the guardrails config.", default="config"),
    dataset_path: str = typer.Option(
        "nemoguardrails/evaluate/data/moderation/harmful.txt",
        help="Path to dataset containing prompts",
    ),
    num_samples: int = typer.Option(50, help="Number of samples to evaluate"),
    check_input: bool = typer.Option(True, help="Evaluate input self-check rail"),
    check_output: bool = typer.Option(True, help="Evaluate output self-check rail"),
    output_dir: str = typer.Option("eval_outputs/moderation", help="Output directory for predictions"),
    write_outputs: bool = typer.Option(True, help="Write outputs to file"),
    split: str = typer.Option("harmful", help="Whether prompts are harmful or helpful"),
):
    """
    Evaluate the performance of the moderation rails defined in a Guardrails application.

    This command computes accuracy for jailbreak detection and output moderation.

    Args:
        config (str): The path to the guardrails config. Defaults to "config".
        dataset_path (str): Path to the dataset containing prompts.
            Defaults to "nemoguardrails/evaluate/data/moderation/harmful.txt".
        num_samples (int): Number of samples to evaluate. Defaults to 50.
        check_input (bool): Evaluate the input self-check rail. Defaults to True.
        check_output (bool): Evaluate the output self-check rail. Defaults to True.
        output_dir (str): Output directory for predictions.
            Defaults to "eval_outputs/moderation".
        write_outputs (bool): Write outputs to file. Defaults to True.
        split (str): Whether prompts are harmful or helpful. Defaults to "harmful".
    """
    moderation_check = ModerationRailsEvaluation(
        config,
        dataset_path,
        num_samples,
        check_input,
        check_output,
        output_dir,
        write_outputs,
        split,
    )
    typer.echo(f"Starting the moderation evaluation for data: {dataset_path} ...")
    moderation_check.run()


@app.command()
def hallucination(
    config: str = typer.Option(help="The path to the guardrails config.", default="config"),
    dataset_path: str = typer.Option("nemoguardrails/evaluate/data/hallucination/sample.txt", help="Dataset path"),
    num_samples: int = typer.Option(50, help="Number of samples to evaluate"),
    output_dir: str = typer.Option("eval_outputs/hallucination", help="Output directory"),
    write_outputs: bool = typer.Option(True, help="Write outputs to file"),
):
    """
    Evaluate the performance of the hallucination rails defined in a Guardrails application.

    This command computes accuracy for hallucination detection.

    Args:
        config (str): The path to the guardrails config. Defaults to "config".
        dataset_path (str): Dataset path. Defaults to "nemoguardrails/evaluate/data/hallucination/sample.txt".
        num_samples (int): Number of samples to evaluate. Defaults to 50.
        output_dir (str): Output directory. Defaults to "eval_outputs/hallucination".
        write_outputs (bool): Write outputs to file. Defaults to True.
    """
    hallucination_check = HallucinationRailsEvaluation(
        config,
        dataset_path,
        num_samples,
        output_dir,
        write_outputs,
    )
    typer.echo(f"Starting the hallucination evaluation for data: {dataset_path} ...")
    hallucination_check.run()


@app.command()
def fact_checking(
    config: str = typer.Option(help="The path to the guardrails config.", default="config"),
    dataset_path: str = typer.Option(
        "nemoguardrails/evaluate/data/factchecking/sample.json",
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
    """
    Evaluate the performance of the fact-checking rails defined in a Guardrails application.

    This command computes accuracy for fact-checking.
    Negatives can be created synthetically by an LLM that acts as an adversary and modifies the answer to make it incorrect.

    Args:
        config (str): The path to the guardrails config. Defaults to "config".
        dataset_path (str): Path to the folder containing the dataset. Defaults to "nemoguardrails/evaluate/data/factchecking/sample.json".
        num_samples (int): Number of samples to be evaluated. Defaults to 50.
        create_negatives (bool): Create synthetic negative samples. Defaults to True.
        output_dir (str): Path to the folder where the outputs will be written. Defaults to "eval_outputs/factchecking".
        write_outputs (bool): Write outputs to the output directory. Defaults to True.
    """
    fact_check = FactCheckEvaluation(
        config,
        dataset_path,
        num_samples,
        create_negatives,
        output_dir,
        write_outputs,
    )
    typer.echo(f"Starting the fact checking evaluation for data: {dataset_path} ...")
    fact_check.run()
