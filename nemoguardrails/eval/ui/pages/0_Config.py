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
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from nemoguardrails.eval.models import EvalConfig
from nemoguardrails.eval.ui.streamlit_utils import load_eval_data
from nemoguardrails.eval.ui.utils import EvalData


def _render_policies(eval_config: EvalConfig):
    """Render the list of policies."""
    st.header("Policies")
    df_policies = pd.DataFrame(
        [[policy.id, policy.description] for policy in eval_config.policies],
        columns=["Policy ID", "Description"],
    )
    st.dataframe(df_policies, use_container_width=True)


def _render_interactions_info(eval_data: EvalData):
    """Render info about the interactions."""
    st.header("Interactions")
    counters = {"all": 0}
    eval_config = eval_data.eval_config

    inputs_array = []

    for interaction_set in eval_config.interactions:
        counters["all"] += len(interaction_set.inputs)

        implicitly_include_policies = []
        for item in interaction_set.expected_output:
            implicitly_include_policies.append(item.policy)

        target_policies = []
        for policy in eval_config.policies:
            if (
                (policy.apply_to_all and policy.id not in interaction_set.exclude_policies)
                or policy.id in interaction_set.include_policies
                or policy.id in implicitly_include_policies
            ):
                counters[policy.id] = counters.get(policy.id, 0) + len(interaction_set.inputs)
                target_policies.append(True)
            else:
                target_policies.append(False)

        for item in interaction_set.inputs:
            inputs_array.append([str(item)] + target_policies)

    st.write(f"This evaluation dataset contains {counters['all']} interactions.")

    # Render the table of interactions
    df = pd.DataFrame(inputs_array, columns=["Input"] + [policy.id for policy in eval_config.policies])
    st.dataframe(df, use_container_width=True)

    # Render chart with interactions per policy
    df = pd.DataFrame(
        [[k, v] for k, v in counters.items()],
        columns=["Policy", "Number of interactions"],
    )
    fig = px.bar(
        df,
        x="Policy",
        y="Number of interactions",
        title="Number of interactions per policy",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_expected_latencies(eval_data: EvalData):
    """Render the configuration of expected latencies"""
    eval_config = eval_data.eval_config
    st.header("Expected latencies")

    st.markdown(
        """
        Expected latencies are used to report latency stats consistently across
        multiple evaluation runs. They are not influenced by network latencies,
        service load or other factors.
        They can also be used to model the expected latencies in various deployment types.
    """
    )

    df_expected_latencies = pd.DataFrame(
        [[metric, value] for metric, value in eval_config.expected_latencies.items()],
        columns=["Metric", "Value (seconds)"],
    )
    df_expected_latencies = st.data_editor(df_expected_latencies, use_container_width=True, num_rows="dynamic")

    changes = False
    for i, row in df_expected_latencies.iterrows():
        metric = row["Metric"]
        value = row["Value (seconds)"]
        if (
            metric is not None
            and (not isinstance(value, float) or not math.isnan(value))
            and value != eval_config.expected_latencies.get(metric)
        ):
            changes = True
            try:
                eval_config.expected_latencies[metric] = float(value)
            except ValueError:
                eval_config.expected_latencies[metric] = 0

    # We also need to remove the latencies that are no longer present
    for k in list(eval_config.expected_latencies.keys()):
        if k not in df_expected_latencies["Metric"].to_numpy():
            del eval_config.expected_latencies[k]
            changes = True

    if changes:
        eval_data.update_config_latencies()
        st.rerun()


def main():
    """Show a summary of the evaluation results."""

    st.title("Evaluation Config")

    st.markdown(
        """
        This section includes information about the evaluation configuration.
    """
    )

    # Load the evaluation data
    eval_data = load_eval_data()
    eval_config = eval_data.eval_config

    # Policies
    _render_policies(eval_config)

    # Interactions
    _render_interactions_info(eval_data)

    # Expected Latencies
    _render_expected_latencies(eval_data)


if __name__ == "__main__":
    main()
