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
from typing import List

from rich.progress import Progress

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.eval.models import (
    EvalConfig,
    EvalOutput,
    InteractionLog,
    InteractionOutput,
    Span,
)
from nemoguardrails.eval.utils import _collect_span_metrics, save_eval_output
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationResponse,
)
from nemoguardrails.utils import console, new_uuid

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")


def _extract_interaction_outputs(eval_config: EvalConfig) -> List[InteractionOutput]:
    """Extract the list of interactions that should be run.

    Creates the output objects with no data.
    """
    results = []
    compliance_dict = {
        policy.id: None if policy.apply_to_all else "n/a"
        for policy in eval_config.policies
    }

    for interaction_set in eval_config.interactions:
        for i, interaction_input in enumerate(interaction_set.inputs):
            interaction_id = interaction_set.id + "/" + str(i)

            # We make sure that the compliance dict is initialized correctly,
            # with "n/a" for all policies that are not applicable, and None for the rest.
            _compliance_dict = compliance_dict.copy()

            for policy_id in interaction_set.exclude_policies:
                _compliance_dict[policy_id] = "n/a"
            for policy_id in interaction_set.include_policies:
                _compliance_dict[policy_id] = None

            # For the policies for which there's an expected output,
            # we consider them to be included policies.
            for item in interaction_set.expected_output:
                _compliance_dict[item.policy] = None

            results.append(
                InteractionOutput(
                    id=interaction_id,
                    input=interaction_input,
                    compliance=_compliance_dict,
                )
            )

    return results


def _load_eval_output(output_path: str, eval_config: EvalConfig) -> EvalOutput:
    """Loads the existing evaluation output from the provided path.

    If existing data is found, it is automatically loaded.
    If the evaluation config has changed meanwhile, it will try to reuse as much as possible.
    For example, if a new interaction is added, the output will keep all the data from the previous interactions.
    """
    existing_eval_output = EvalOutput.from_path(output_path)

    # Create reverse indexes for the output interactions and logs, for easy lookup
    id_to_output = {}
    id_to_log = {}
    for _interaction_output in existing_eval_output.results:
        id_to_output[_interaction_output.id] = _interaction_output
    for _log in existing_eval_output.logs:
        id_to_log[_log.id] = _log

    eval_output = EvalOutput()

    # We extract the interactions that correspond to the current config
    interaction_outputs = _extract_interaction_outputs(eval_config)

    for _interaction_output in interaction_outputs:
        _id = _interaction_output.id

        # If we have an existing id, for the same input, we re-use
        if _id in id_to_output and _interaction_output.input == id_to_output[_id].input:
            eval_output.results.append(id_to_output[_id])

            # If we have a log, we use it as well, if not, we add an empty one
            eval_output.logs.append(id_to_log.get(_id, InteractionLog(id=_id)))
        else:
            eval_output.results.append(_interaction_output)
            eval_output.logs.append(InteractionLog(id=_id))

    return eval_output


def _extract_interaction_log(
    interaction_output: InteractionOutput, generation_log: GenerationLog
) -> InteractionLog:
    """Extracts an `InteractionLog` object from an `GenerationLog` object."""
    return InteractionLog(
        id=interaction_output.id,
        activated_rails=generation_log.activated_rails,
        events=generation_log.internal_events,
        trace=_extract_spans(generation_log.activated_rails),
    )


def _extract_spans(activated_rails: List[ActivatedRail]) -> List[Span]:
    """Extract a simplified span view from the log of activated rails."""
    spans = []
    ref_time = activated_rails[0].started_at
    interaction_span = Span(
        span_id=new_uuid(),
        name="interaction",
        start_time=activated_rails[0].started_at - ref_time,
        end_time=activated_rails[-1].finished_at - ref_time,
        duration=activated_rails[-1].finished_at - activated_rails[0].started_at,
    )

    interaction_span.metrics.update(
        {
            "interaction_total": 1,
            "interaction_seconds_avg": interaction_span.duration,
            "interaction_seconds_total": interaction_span.duration,
        }
    )
    spans.append(interaction_span)

    for activated_rail in activated_rails:
        rail_span = Span(
            span_id=new_uuid(),
            name="rail: " + activated_rail.name,
            parent_id=interaction_span.span_id,
            start_time=activated_rail.started_at - ref_time,
            end_time=activated_rail.finished_at - ref_time,
            duration=activated_rail.duration,
        )
        spans.append(rail_span)

        for action in activated_rail.executed_actions:
            action_span = Span(
                span_id=new_uuid(),
                name="action: " + action.action_name,
                parent_id=rail_span.span_id,
                start_time=action.started_at - ref_time,
                end_time=action.finished_at - ref_time,
                duration=action.duration,
            )
            # Compute the metrics
            base_metric_name = f"action_{action.action_name}"

            # Update the latency metrics
            action_span.metrics.update(
                {
                    f"{base_metric_name}_total": 1,
                    f"{base_metric_name}_seconds_avg": action.duration,
                    f"{base_metric_name}_seconds_total": action.duration,
                }
            )
            spans.append(action_span)

            for llm_call in action.llm_calls:
                model_name = llm_call.llm_model_name
                llm_span = Span(
                    span_id=new_uuid(),
                    name="LLM: " + model_name,
                    parent_id=action_span.span_id,
                    start_time=llm_call.started_at - ref_time,
                    end_time=llm_call.finished_at - ref_time,
                    duration=llm_call.duration,
                )

                # Compute the metrics
                base_metric_name = f"llm_call_{model_name.replace('/', '_')}"

                # Update the latency metrics
                llm_span.metrics.update(
                    {
                        f"{base_metric_name}_total": 1,
                        f"{base_metric_name}_seconds_avg": llm_call.duration,
                        f"{base_metric_name}_seconds_total": llm_call.duration,
                    }
                )

                llm_span.metrics.update(
                    {
                        f"{base_metric_name}_prompt_tokens_total": llm_call.prompt_tokens,
                        f"{base_metric_name}_completion_tokens_total": llm_call.completion_tokens,
                        f"{base_metric_name}_tokens_total": llm_call.total_tokens,
                    }
                )

                spans.append(llm_span)

    return spans


async def run_eval(
    eval_config_path: str,
    guardrail_config_path: str,
    output_path: str,
    output_format: str = "json",
    parallel: int = 1,
):
    """Run a guardrail evaluation.

    Args:
        eval_config_path (str): Path to a directory containing eval configuration files.
        guardrail_config_path (str): Path to a directory containing the guardrail configuration.
        output_path (str, optional): Output directory for predictions. Defaults to None.
        output_format (str, optional): Output format. Supported values are "yaml" and "json". Defaults to "yaml".
        parallel (int, optional): Number degree of parallelism to use. Defaults to 1.
    """
    t0 = time.time()

    console.print(f"Loading eval configuration [bold]{eval_config_path}[/] ...")
    eval_config_path = os.path.abspath(eval_config_path)
    eval_config = EvalConfig.from_path(eval_config_path)
    interactions = _extract_interaction_outputs(eval_config)

    console.print(
        f"Loaded {len(eval_config.policies)} policies and {len(interactions)} interactions."
    )

    console.print(
        f"Loading guardrail configuration [bold]{guardrail_config_path}[/] ..."
    )
    if parallel > 1:
        console.print(f"[bold]Parallelism set to {parallel}[/]")
    rails_config = RailsConfig.from_path(guardrail_config_path)
    rails = LLMRails(config=rails_config)

    # Create the output paths if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Start running the interactions.
    eval_output = _load_eval_output(output_path, eval_config)
    save_eval_output(eval_output, output_path, output_format)

    generation_options = {"log": {"activated_rails": True, "internal_events": True}}

    progress = Progress()
    with progress:
        task_id = progress.add_task(
            f"Running {len(interactions)} interactions ...", total=len(interactions)
        )
        i = 0

        async def _worker():
            """Async worker that processes interactions until the end."""
            nonlocal i

            while i < len(interactions):
                interaction = eval_output.results[i]
                idx = i
                i += 1

                result: GenerationResponse

                if isinstance(interaction.input, str):
                    progress.print(f'[{i}] "{interaction.input}"')
                    result = await rails.generate_async(
                        prompt=interaction.input,
                        options=generation_options,
                    )
                else:
                    progress.print(f"[{i}] {json.dumps(interaction.input)}")
                    result = await rails.generate_async(
                        messages=interaction.input["messages"],
                        options=generation_options,
                    )

                interaction.output = result.response
                interaction_log = _extract_interaction_log(interaction, result.log)
                eval_output.logs[idx] = interaction_log

                metrics = _collect_span_metrics(interaction_log.trace)
                interaction.resource_usage = {
                    k: v for k, v in metrics.items() if "_seconds" not in k
                }
                interaction.latencies = {
                    k: v for k, v in metrics.items() if "_seconds" in k
                }

                save_eval_output(eval_output, output_path, output_format)

                progress.update(task_id, advance=1)

        # Start the desired number of workers.
        await asyncio.gather(*[_worker() for _ in range(parallel)])

    console.print(f"The run for {output_path} took {time.time() - t0:.2f} seconds.")
