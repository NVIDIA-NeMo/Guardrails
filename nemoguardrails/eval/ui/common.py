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
from typing import List, Tuple

import pandas as pd
import streamlit as st
from pandas import DataFrame

from nemoguardrails.eval.models import EvalConfig, EvalOutput
from nemoguardrails.eval.ui.chart_utils import (
    plot_as_series,
    plot_bar_series,
    plot_matrix_series,
)
from nemoguardrails.eval.ui.streamlit_utils import load_eval_data
from nemoguardrails.eval.ui.utils import (
    EvalData,
    collect_interaction_metrics,
    collect_interaction_metrics_with_expected_latencies,
)

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def _render_sidebar(output_names: List[str], policy_options: List[str], tags: List[str]):
    _output_names = []
    _policy_options = []
    _tags = []

    with st.sidebar:
        st.write("If you change the result files outside of the Eval UI, you must reload from disk. ")
        if st.button("Reload"):
            load_eval_data.clear()
            st.rerun()

        st.session_state.use_expected_latencies = st.checkbox(
            "Use expected latencies",
            value=False,
            help="Expected latencies are computed according to pre-defined values. \n"
            "For more details go to the 'Config > Expected Latencies' section.",
        )

        with st.sidebar.expander("Guardrail Configurations", expanded=True):
            for output_path in output_names:
                if st.checkbox(output_path, True):
                    _output_names.append(output_path)

    with st.sidebar.expander("Policies", expanded=True):
        for policy in policy_options:
            if st.checkbox(policy, True):
                _policy_options.append(policy)

    with st.sidebar.expander("Tags", expanded=True):
        for tag in tags:
            if st.checkbox(tag, True, key=f"tag-{tag}"):
                _tags.append(tag)

    return _output_names, _policy_options, _tags


def _get_compliance_df(output_names: List[str], policy_options: List[str], eval_data: EvalData) -> DataFrame:
    """Computes a DataFrame with information about compliance.

    Returns
        DataFrame: ["Guardrail Config", "Policy", "Compliance Rate", "Violations Count", "Interaction Count"]
    """
    data = []
    for output_name in output_names:
        compliance_info = eval_data.eval_outputs[output_name].compute_compliance(eval_data.eval_config)

        for policy_id in policy_options:
            compliance_rate = round(compliance_info[policy_id]["rate"] * 100, 2)
            violations_count = compliance_info[policy_id]["interactions_violation_count"]
            interactions_count = compliance_info[policy_id]["interactions_count"]

            data.append(
                [
                    output_name,
                    policy_id,
                    compliance_rate,
                    violations_count,
                    interactions_count,
                ]
            )

    return DataFrame(
        data,
        columns=[
            "Guardrail Config",
            "Policy",
            "Compliance Rate",
            "Violations Count",
            "Interactions Count",
        ],
    )


def _render_compliance_data(
    output_names: List[str],
    policy_options: List[str],
    eval_data: EvalData,
    short: bool = False,
):
    st.text(f"({len(eval_data.eval_outputs[output_names[0]].results)} interactions)")
    st.header("Compliance")
    st.markdown(
        """
        The *overall compliance rate* is the weighted average of the compliance rate across all policies.
        The *compliance rate* is the percentage of interactions which comply with a policy out
        of the number of interactions for which the policy is applicable.
    """
    )
    df_compliance = _get_compliance_df(output_names, policy_options, eval_data)

    df_overall_compliance = (
        df_compliance.groupby("Guardrail Config")
        .apply(
            # lambda x: (x["Value"] * x["Weight"]).sum() / x["Weight"].sum(),
            lambda g: g["Compliance Rate"].mean(),
            include_groups=False,
        )
        .reset_index(name="Compliance Rate")
    )

    plot_as_series(df_overall_compliance, range_y=[0, 100], title="Overall Compliance Rate")

    if short:
        return

    with st.expander("Table", expanded=False):
        st.dataframe(df_overall_compliance)

    plot_bar_series(
        df_compliance[["Guardrail Config", "Policy", "Compliance Rate"]],
        title="Compliance Rate per Policy",
        include_table=True,
    )

    st.subheader("Violations")

    st.markdown(
        """
        *Violations* are interactions which don't comply with one or more policies.
        You can review them individually by using the "Non-compliant interactions" filter in *Review mode*.
    """
    )

    plot_bar_series(
        df_compliance[["Guardrail Config", "Policy", "Violations Count"]],
        title="Violations Count per Policy",
        include_table=True,
    )

    plot_bar_series(
        df_compliance[["Guardrail Config", "Policy", "Interactions Count"]],
        title="Interactions Count per Policy",
        include_table=True,
    )

    st.info(
        "**Note**: For policies where the LLM judge can decide if the policy is applicable or not, "
        "the number of interactions can be different between different evaluations."
    )


def _get_resource_usage_and_latencies_df(
    output_names: List[str],
    eval_data: EvalData,
    eval_config: EvalConfig,
    use_expected_latencies: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    """Computes a DataFrame with information about resource usage and latencies.

    Returns
        DataFrame: ["Metric", *output_names]
    """
    resource_usage_table = []
    latencies_table = []
    metrics = {}
    all_metrics = []

    def _update_value(table, column, metric, value):
        """Helper to update the value of a metric in the table."""
        for row in table:
            if row[0] == metric:
                row[column + 1] = value
                return

    for output_name in output_names:
        if not use_expected_latencies:
            metrics[output_name] = collect_interaction_metrics(eval_data.eval_outputs[output_name].results)
        else:
            metrics[output_name] = collect_interaction_metrics_with_expected_latencies(
                eval_data.eval_outputs[output_name].results,
                eval_data.eval_outputs[output_name].logs,
                expected_latencies=eval_config.expected_latencies,
            )

        for k in metrics[output_name]:
            if k not in all_metrics:
                all_metrics.append(k)

    all_metrics = list(sorted(all_metrics))

    for metric in all_metrics:
        if "_seconds" in metric:
            table = latencies_table
        else:
            table = resource_usage_table

        table.append([metric] + [None for _ in range(len(output_names))])

    for i, output_name in enumerate(output_names):
        for metric, value in metrics[output_name].items():
            if "_seconds" in metric:
                # latencies_table.append([metric, value])
                _update_value(latencies_table, i, metric, value)
            else:
                # resource_usage_table.append([metric, value])
                _update_value(resource_usage_table, i, metric, value)

    return (
        DataFrame(resource_usage_table, columns=["Metric", *output_names]),
        DataFrame(latencies_table, columns=["Metric", *output_names]),
    )


def _render_resource_usage_and_latencies(
    output_names: List[str],
    eval_data: EvalData,
    eval_config: EvalConfig,
    short: bool = False,
):
    """Render the resource usage part."""
    df_resource_usage, df_latencies = _get_resource_usage_and_latencies_df(
        output_names,
        eval_data,
        eval_config=eval_config,
        use_expected_latencies=st.session_state.use_expected_latencies,
    )

    latency_type = "Expected" if st.session_state.use_expected_latencies else "Measured"

    if not short:
        st.header("Resource Usage")

        st.markdown(
            """
            *Resources* are divided into two categories: *LLMs* and *Actions*.
            For each resource, the number of calls and latencies are tracked.
            For LLM resources, the token usage is also tracked.
        """
        )

        if st.checkbox("Show raw resource usage data"):
            st.dataframe(
                df_resource_usage,
                use_container_width=True,
                hide_index=True,
            )

    st.header("LLM Usage")

    st.markdown(
        """
        **Total LLM Calls** represents the total number of LLM calls made
        when running the interactions in the evaluation dataset,
        regardless of the LLM model used.
    """
    )

    # 1. LLM Calls Count
    df_llm_calls_count = df_resource_usage[
        df_resource_usage["Metric"].str.startswith("llm_call_")
        & df_resource_usage["Metric"].str.endswith("_total")
        & ~df_resource_usage["Metric"].str.endswith("_tokens_total")
    ]
    # remove the `llm_call_` prefix and `_total` suffix.
    df_llm_calls_count["Metric"] = df_llm_calls_count["Metric"].str[9:-6]

    # We extract
    llm_models = df_llm_calls_count["Metric"].values

    # We also compute the total LLM calls, independent of the LLM
    df = df_llm_calls_count.copy()
    df["Metric"] = "LLM Calls"
    df = df.groupby(["Metric"]).sum().reset_index()
    df = df.transpose().reset_index().drop(0)
    df.columns = ["Guardrail Config", "LLM Calls"]
    plot_as_series(df, title="Total LLM Calls", include_table=True)

    if not short:
        if len(llm_models) > 1:
            plot_matrix_series(
                df_llm_calls_count,
                "Guardrail Config",
                "LLM Calls",
                include_table=True,
                title="Total LLM Calls by Model",
            )

    # Token Usage
    st.subheader("Token Usage")
    st.markdown(
        """
        **Total Token Usage** represents the total number of tokens used when running the interactions,
        regardless of the LLM model.
    """
    )

    # Detailed information about the token usage.
    df_llm_usage = df_resource_usage[
        (df_resource_usage["Metric"].str.startswith("llm_call_"))
        & (df_resource_usage["Metric"].str.endswith("_tokens_total"))
    ]
    df_llm_usage["Metric"] = df_llm_usage["Metric"].str[9:-13]

    # Detailed usage
    df_llm_usage_detailed = df_llm_usage.melt(id_vars=["Metric"], var_name="Guardrail Config", value_name="Value")[
        ["Guardrail Config", "Metric", "Value"]
    ]

    # Compute total token usage per category (Prompt, Completion, Total)
    df_total_tokens_per_category = df_llm_usage_detailed.copy()

    def _update_value(value):
        if "_prompt" in value:
            return "Prompt Tokens"
        elif "_completion" in value:
            return "Completion Tokens"
        else:
            return "Total Tokens"

    df_total_tokens_per_category["Metric"] = df_total_tokens_per_category["Metric"].apply(_update_value)
    df_total_tokens_per_category = (
        df_total_tokens_per_category.groupby(["Guardrail Config", "Metric"])["Value"].sum().reset_index()
    )
    df_total_tokens_per_category = df_total_tokens_per_category.rename(columns={"Value": "Tokens"})
    plot_bar_series(df_total_tokens_per_category, title="Total Token Usage", include_table=True)

    if not short:
        if len(llm_models) > 1:
            # Compute total tokens usage per LLM
            df_llm_total_tokens = df_llm_usage_detailed[
                ~df_llm_usage_detailed["Metric"].str.contains("completion")
                & ~df_llm_usage_detailed["Metric"].str.contains("prompt")
            ]
            df_llm_total_tokens = df_llm_total_tokens.rename(columns={"Value": "Total Tokens"})
            plot_bar_series(df_llm_total_tokens, title="Total Tokens per LLM", include_table=True)

            # st.dataframe(df_llm_usage, use_container_width=True)
            plot_bar_series(
                df_llm_usage_detailed,
                title="Detailed Token Usage per LLM",
                include_table=True,
            )
            # if st.checkbox("Show table", key="show-llm-usage-table"):
            #     st.dataframe(df_llm_usage)

    if not short:
        st.header(f"{latency_type} Latencies")

        st.markdown(
            f"""
            The *Total {latency_type} Latency* is the total duration of all interactions in the dataset.
        """
        )

        if st.checkbox("Show raw latency data"):
            st.dataframe(
                df_latencies,
                use_container_width=True,
                hide_index=True,
            )

    # Chart with Total Latency for the test
    st.subheader(f"{latency_type} Interaction Latencies")

    if not short:
        df = (
            df_latencies.set_index("Metric")
            .loc[["interaction_seconds_total"]]
            .reset_index()
            .transpose()
            .reset_index()
            .drop(0)
        )
        df.columns = ["Guardrail Config", "Total Latency"]
        plot_as_series(df, title=f"Total {latency_type} Interactions Latency", include_table=True)

    df = (
        df_latencies.set_index("Metric")
        .loc[["interaction_seconds_avg"]]
        .reset_index()
        .transpose()
        .reset_index()
        .drop(0)
    )
    df.columns = ["Guardrail Config", "Average Latency"]
    plot_as_series(df, title=f"Average {latency_type} Interaction Latency", include_table=True)

    if not short:
        # Total and Average latency per LLM Call
        st.subheader(f"LLM Call {latency_type} Latencies")
        df = df_latencies[
            df_latencies["Metric"].str.startswith("llm_call_") & df_latencies["Metric"].str.endswith("_seconds_total")
        ]
        df["Metric"] = df["Metric"].str[9:-14]
        plot_matrix_series(
            df,
            var_name="Guardrail Config",
            value_name="Total Latency",
            title=f"Total {latency_type} Latency per LLM",
            include_table=True,
        )

        df = df_latencies[
            df_latencies["Metric"].str.startswith("llm_call_") & df_latencies["Metric"].str.endswith("_seconds_avg")
        ]
        df["Metric"] = df["Metric"].str[9:-12]
        plot_matrix_series(
            df,
            var_name="Guardrail Config",
            value_name="Average Latency",
            title=f"Average {latency_type} Latency per LLM",
            include_table=True,
        )

        # Total and Average latency per action
        st.subheader(f"{latency_type} Action Latencies")

        st.info(
            """
            **Note**: The latency of an action also includes the latency of the LLM calls it makes.
        """
        )
        df = df_latencies[
            df_latencies["Metric"].str.startswith("action_") & df_latencies["Metric"].str.endswith("_seconds_total")
        ]
        df["Metric"] = df["Metric"].str[7:-14]
        plot_matrix_series(
            df,
            var_name="Guardrail Config",
            value_name="Total Latency",
            title=f"Total {latency_type} Latency per Action",
            include_table=True,
        )

        df = df_latencies[
            df_latencies["Metric"].str.startswith("action_") & df_latencies["Metric"].str.endswith("_seconds_avg")
        ]
        df["Metric"] = df["Metric"].str[7:-12]
        plot_matrix_series(
            df,
            var_name="Guardrail Config",
            value_name="Average Latency",
            title=f"Average {latency_type} Latency per Action",
            include_table=True,
        )


def render_summary(short: bool = False):
    """Show a summary of the evaluation results."""

    st.title("Evaluation Summary")

    # Load the evaluation data
    eval_data = load_eval_data().copy()
    eval_config = eval_data.eval_config

    # Extract the list of tags from the interactions
    all_tags = []
    for interaction_set in eval_config.interactions:
        for tag in interaction_set.tags:
            if tag not in all_tags:
                all_tags.append(tag)

    output_names = list(eval_data.eval_outputs.keys())
    policy_options = [policy.id for policy in eval_config.policies]

    # Sidebar
    output_names, policy_options, tags = _render_sidebar(output_names, policy_options, all_tags)

    # If all tags are selected, we don't do the filtering.
    # Like this, interactions without tags will also be included.
    if len(tags) != len(all_tags):
        # We filter the interactions to only those that have the right tags
        filtered_interaction_ids = []
        for interaction_set in eval_config.interactions:
            include = False
            for tag in tags:
                if tag in interaction_set.tags:
                    include = True
                    break
            if include:
                filtered_interaction_ids.append(interaction_set.id)

        new_eval_outputs = {}
        for output_name in output_names:
            eval_output = eval_data.eval_outputs[output_name]
            _results = []
            _logs = []
            for i in range(len(eval_output.results)):
                interaction_id = eval_output.results[i].id.split("/")[0]
                if interaction_id in filtered_interaction_ids:
                    _results.append(eval_output.results[i])
                    _logs.append(eval_output.logs[i])

            new_eval_outputs[output_name] = EvalOutput(results=_results, logs=_logs)

        eval_data.eval_outputs = new_eval_outputs

    # Compliance data
    _render_compliance_data(output_names, policy_options, eval_data, short=short)

    # Resource Usage and Latencies
    _render_resource_usage_and_latencies(output_names, eval_data, eval_config=eval_config, short=short)
