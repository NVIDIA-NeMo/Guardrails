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

from time import time
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from nemoguardrails.eval.models import (
    EvalConfig,
    EvalOutput,
    InteractionLog,
    InteractionOutput,
    Span,
)
from nemoguardrails.eval.utils import _collect_span_metrics, update_dict_at_path


class EvalData(BaseModel):
    """Data relation to an evaluation, relevant for the UI."""

    eval_config_path: str
    eval_config: EvalConfig
    output_paths: List[str]
    eval_outputs: Dict[str, EvalOutput]
    selected_output_path: Optional[str] = None

    def update_results(self):
        """Updates back the evaluation results."""
        t0 = time()
        results = [r.dict() for r in self.eval_outputs[self.selected_output_path].results]
        update_dict_at_path(self.selected_output_path, {"results": results})
        print(f"Updating output results took {time() - t0:.2f} seconds.")

    def update_results_and_logs(self, output_path: str):
        """Update back the results and the logs."""
        t0 = time()
        results = [r.dict() for r in self.eval_outputs[output_path].results]
        logs = [r.dict() for r in self.eval_outputs[output_path].logs]
        update_dict_at_path(output_path, {"results": results, "logs": logs})
        print(f"Updating output results took {time() - t0:.2f} seconds.")

    def update_config_latencies(self):
        """Update back the expected latencies."""
        t0 = time()
        update_dict_at_path(
            self.eval_config_path,
            {"expected_latencies": self.eval_config.expected_latencies},
        )
        print(f"Updating expected latencies took {time() - t0:.2f} seconds.")


def collect_interaction_metrics(
    interaction_outputs: List[InteractionOutput],
) -> Dict[str, Union[int, float]]:
    """Collects and aggregates the metrics from all the interactions."""
    metrics = {}
    counters = {}
    for interaction_output in interaction_outputs:
        for metric in interaction_output.resource_usage:
            metrics[metric] = metrics.get(metric, 0) + interaction_output.resource_usage[metric]
            counters[metric] = counters.get(metric, 0) + 1

        for metric in interaction_output.latencies:
            metrics[metric] = metrics.get(metric, 0) + interaction_output.latencies[metric]
            counters[metric] = counters.get(metric, 0) + 1

    # For the avg metrics, we need to average them
    for metric in counters:
        if metric.endswith("_avg"):
            metrics[metric] = metrics[metric] / counters[metric]

    return metrics


def collect_interaction_metrics_with_expected_latencies(
    interaction_outputs: List[InteractionOutput],
    interaction_logs: List[InteractionLog],
    expected_latencies: Dict[str, float],
):
    """Similar to collect_interaction_metrics but with expected latencies."""
    metrics = {}
    counters = {}
    for interaction_output, interaction_log in zip(interaction_outputs, interaction_logs):
        # Resource usage computation stays the same
        for metric in interaction_output.resource_usage:
            metrics[metric] = metrics.get(metric, 0) + interaction_output.resource_usage[metric]
            counters[metric] = counters.get(metric, 0) + 1

        # For the latency part, we need to first update the spans and then recompute the latencies.
        updated_spans = [Span.parse_obj(span.dict()) for span in interaction_log.trace]

        # We create an index so that we can quickly look up the parents.
        updated_span_by_idx = {}
        for updated_span in updated_spans:
            updated_span_by_idx[updated_span.span_id] = updated_span

        for span in updated_spans:
            metric_names = list(span.metrics.keys())
            if metric_names:
                if metric_names[0].startswith("llm_call_"):
                    # The first metric should be the total number of calls
                    assert metric_names[0].endswith("_total")
                    llm_name = metric_names[0][9:-6]

                    # If we don't have prompt info, we skip
                    if f"llm_call_{llm_name}_prompt_tokens_total" not in span.metrics:
                        continue

                    prompt_tokens = span.metrics[f"llm_call_{llm_name}_prompt_tokens_total"]
                    completion_tokens = span.metrics[f"llm_call_{llm_name}_completion_tokens_total"]

                    fixed_latency = expected_latencies.get(f"llm_call_{llm_name}_fixed_latency", 0.25)
                    prompt_token_latency = expected_latencies.get(f"llm_call_{llm_name}_prompt_token_latency", 0.0001)
                    completion_token_latency = expected_latencies.get(
                        f"llm_call_{llm_name}_completion_token_latency", 0.01
                    )

                    # This is a heuristic to approximate the latency based on a set of
                    # pre-defined latencies and prompt/completion size.
                    latency = (
                        fixed_latency
                        + prompt_token_latency * prompt_tokens
                        + completion_token_latency * completion_tokens
                    )

                    current_latency = span.metrics[f"llm_call_{llm_name}_seconds_avg"]
                    span.metrics[f"llm_call_{llm_name}_seconds_avg"] = latency
                    span.metrics[f"llm_call_{llm_name}_seconds_total"] = latency

                    diff = latency - current_latency
                    span.duration += diff

                    while span.parent_id:
                        span = updated_span_by_idx[span.parent_id]
                        span.duration += diff
                        for metric in span.metrics.keys():
                            if "_seconds" in metric:
                                span.metrics[metric] += diff

        _metrics = _collect_span_metrics(updated_spans)
        latencies = {k: v for k, v in _metrics.items() if "_seconds" in k}

        for metric in latencies:
            metrics[metric] = metrics.get(metric, 0) + latencies[metric]
            counters[metric] = counters.get(metric, 0) + 1

    # For the avg metrics, we need to average them
    for metric in counters:
        if metric.endswith("_avg"):
            metrics[metric] = metrics[metric] / counters[metric]

    return metrics
