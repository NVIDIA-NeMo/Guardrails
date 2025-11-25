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

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nemoguardrails.eval.models import ComplianceCheckResult, InteractionOutput, Policy
from nemoguardrails.eval.ui.streamlit_utils import get_span_colors, load_eval_data
from nemoguardrails.eval.ui.utils import EvalData
from nemoguardrails.utils import new_uuid


def _render_policy(_policy: Policy, interaction_output: InteractionOutput, eval_data: EvalData):
    index = 0
    orig_option = ""
    if interaction_output.compliance[_policy.id] is True:
        index = 1
        orig_option = "Complies"
    elif interaction_output.compliance[_policy.id] is False:
        index = 2
        orig_option = "Does NOT comply"
    elif interaction_output.compliance[_policy.id] == "n/a":
        index = 3
        orig_option = "n/a"

    option = st.selectbox(
        _policy.id,
        (
            "",
            "Complies",
            "Does NOT comply",
            "n/a",
        ),
        index=index,
        placeholder="Complies?",
        help=_policy.description,
        key=f"{_policy.id}-{eval_data.selected_output_path}-{interaction_output.id}",
    )
    if orig_option != option:
        if option == "Complies":
            interaction_output.compliance[_policy.id] = True
        elif option == "Does NOT comply":
            interaction_output.compliance[_policy.id] = False
        elif option == "n/a":
            interaction_output.compliance[_policy.id] = "n/a"
        else:
            interaction_output.compliance[_policy.id] = None

        # We also need to record the change made by the human
        interaction_output.compliance_checks.append(
            ComplianceCheckResult(
                id=new_uuid(),
                created_at=datetime.now(timezone.utc).isoformat(),
                interaction_id=interaction_output.id,
                method="manual",
                compliance={_policy.id: interaction_output.compliance[_policy.id]},
                details="",
            )
        )

        # We need to save the output data
        st.session_state.executor.submit(eval_data.update_results)


def main():
    """Review the interaction and evaluation results one by one."""
    # Load the evaluation data
    eval_data = load_eval_data()

    if "executor" not in st.session_state:
        st.session_state.executor = ThreadPoolExecutor(max_workers=1)

    with st.sidebar:
        eval_data.selected_output_path = st.sidebar.selectbox(
            "Guardrail Configurations", options=eval_data.eval_outputs.keys(), index=0
        )
        eval_output = eval_data.eval_outputs[eval_data.selected_output_path]

        st.write("If you change the result files outside of the Eval UI, you must reload from disk. ")
        if st.button("Reload"):
            load_eval_data.clear()
            st.rerun()

        if "interactions_filter" not in st.session_state:
            st.session_state.interactions_filter = {}

        with st.sidebar.expander("Filter", expanded=True):
            st.session_state.non_compliant_filter = st.checkbox(
                "Non-compliant interactions",
                help="Interactions that have at least on policy violation.",
            )

            st.session_state.judge_inconsistency_filter = st.checkbox(
                "Compliance inconsistencies",
                help="An interaction has compliance inconsistencies if there are different "
                "compliance values for a policy (e.g., either between human and LLM-as-a-judge, "
                "or between multiple runs of the LLM-as-a-judge).",
            )

            for policy in eval_data.eval_config.policies:
                option = st.selectbox(
                    policy.id,
                    (
                        "",
                        "No value",
                        "Complies",
                        "Does NOT comply",
                        "n/a",
                    ),
                    placeholder="Complies?",
                    help=policy.description,
                    key=f"{policy.id}-filter",
                )
                if option == "Complies":
                    st.session_state.interactions_filter[policy.id] = True
                elif option == "Does NOT comply":
                    st.session_state.interactions_filter[policy.id] = False
                elif option == "n/a":
                    st.session_state.interactions_filter[policy.id] = "n/a"
                elif option == "No value":
                    st.session_state.interactions_filter[policy.id] = None
                else:
                    if policy.id in st.session_state.interactions_filter:
                        del st.session_state.interactions_filter[policy.id]

    if "idx" not in st.session_state:
        st.session_state.idx = 1
    if "slider_idx" not in st.session_state:
        st.session_state.slider_idx = 1
    if "idx_change" not in st.session_state:
        st.session_state.idx_change = None

    if st.session_state.idx != st.session_state.slider_idx and st.session_state.idx_change == "button":
        st.session_state.idx_change = None
        st.session_state.slider_idx = st.session_state.idx
    else:
        st.session_state.idx = st.session_state.slider_idx

    # Do the filtering

    filtered_results = []
    filtered_logs = []

    for i in range(len(eval_output.results)):
        item = eval_output.results[i]
        log = eval_output.logs[i]
        skip = False
        for policy_id, val in st.session_state.interactions_filter.items():
            if item.compliance.get(policy_id) != val:
                skip = True
                break

        if st.session_state.non_compliant_filter:
            if False not in item.compliance.values():
                skip = True

        if st.session_state.judge_inconsistency_filter:
            inconsistency = False
            for policy_id in item.compliance:
                val = "---"
                for _item in item.compliance_checks:
                    for _policy_id in _item.compliance:
                        if _policy_id == policy_id:
                            if val == "---":
                                val = _item.compliance[_policy_id]
                            elif val != _item.compliance[_policy_id]:
                                inconsistency = True

            if not inconsistency:
                skip = True

        if not skip:
            filtered_results.append(item)
            filtered_logs.append(log)

    # Make sure the index does not fall outside
    if st.session_state.idx >= len(filtered_results):
        st.session_state.idx = 0

    if st.session_state.slider_idx >= len(filtered_results):
        st.session_state.slider_idx = 0

    # Title
    st.title(f"Review {len(filtered_results)} interactions")

    if len(filtered_results) == 0:
        return

    st.slider(
        "Interaction Index",
        min_value=1,
        max_value=len(filtered_results),
        key="slider_idx",
    )

    interaction_output = filtered_results[st.session_state.idx - 1]
    interaction_id = interaction_output.id.split("/")[0]
    interaction_set = [_i for _i in eval_data.eval_config.interactions if _i.id == interaction_id][0]

    # Interaction history

    if isinstance(interaction_output.input, str):
        with st.chat_message("user"):
            st.write(interaction_output.input)
    else:
        assert isinstance(interaction_output.input, dict)
        for message in interaction_output.input["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if isinstance(interaction_output.output, str):
        with st.chat_message("assistant"):
            st.write(interaction_output.output)
    else:
        for message in interaction_output.output:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Expected output

    if interaction_set.expected_output:
        lines = ["**Expected**:"]
        for expected_output in interaction_set.expected_output:
            lines.append(f" - {str(expected_output)} ({expected_output.policy})")
        lines.append("---")

        st.markdown("\n".join(lines))

    # Violations

    violations = []
    for policy_id, val in interaction_output.compliance.items():
        if val is False:
            for check in reversed(interaction_output.compliance_checks):
                if check.compliance.get(policy_id) is False:
                    violations.append(f" - [{check.method}] **{policy_id}**: {check.details}")
                    break
    if violations:
        st.markdown("**Violations**:\n" + "\n".join(violations) + "\n---")

    st.write("Any changes to you make to the compliance statuses below are saved automatically to the result files. ")
    # Render the navigation buttons

    col1, col2, col3, col4 = st.columns([4, 2, 3, 5])
    with col1:
        if st.button('Mark all as "Complies"'):
            for policy_id in interaction_output.compliance:
                # We only change the status for the applicable policies
                if interaction_output.compliance[policy_id] != "n/a":
                    interaction_output.compliance[policy_id] = True

                    # We also need to record the change made by the human
                    interaction_output.compliance_checks.append(
                        ComplianceCheckResult(
                            id=new_uuid(),
                            created_at=datetime.now(timezone.utc).isoformat(),
                            interaction_id=interaction_output.id,
                            method="manual",
                            compliance={policy_id: interaction_output.compliance[policy_id]},
                            details="",
                        )
                    )

            # We need to save the output data
            st.session_state.executor.submit(eval_data.update_results)

            st.rerun()

    with col2:
        if st.button("Previous"):
            if st.session_state.idx > 1:
                st.session_state.idx -= 1
                st.session_state.idx_change = "button"
                st.rerun()

    with col3:
        if st.button("Next"):
            if st.session_state.idx < len(eval_output.results):
                st.session_state.idx += 1
                st.session_state.idx_change = "button"
                st.rerun()

    col1, col2 = st.columns([1, 1])

    # Render the policies in a two-column layout.
    with col1:
        for i, policy in enumerate(eval_data.eval_config.policies):
            if i % 2 == 1:
                continue

            _render_policy(policy, interaction_output, eval_data)

    with col2:
        for i, policy in enumerate(eval_data.eval_config.policies):
            if i % 2 == 0:
                continue

            _render_policy(policy, interaction_output, eval_data)

    if "show_compliance_check_details" not in st.session_state:
        st.session_state.show_compliance_check_details = False

    def _switch():
        st.session_state.show_compliance_check_details = not getattr(
            st.session_state, "show_compliance_check_details", False
        )

    # Compliance check details
    st.checkbox(
        "Show compliance check details",
        value=getattr(st.session_state, "show_compliance_check_details", False),
        on_change=_switch,
    )

    if st.session_state.show_compliance_check_details:
        rows = [["Policy", "Compliance", "Check", "Reason", "DateTime"]]
        for result in interaction_output.compliance_checks:
            for _policy_id in result.compliance:
                status_str = "n/a"
                status_val = result.compliance[_policy_id]
                if status_val is True:
                    status_str = "Complies"
                elif status_val is False:
                    status_str = "Does NOT comply"

                rows.append(
                    [
                        _policy_id,
                        status_str,
                        result.method,
                        result.details,
                        result.created_at,
                    ]
                )

        df = pd.DataFrame(rows[1:], columns=rows[0])
        st.dataframe(df)

    st.header("Details")

    interaction_log = eval_output.logs[st.session_state.idx - 1]
    spans = list(reversed(interaction_log.trace))
    # st.write(spans)

    data = {
        "start_time": [span.start_time for span in spans],
        "end_time": [span.end_time for span in spans],
        "span_id": [span.span_id for span in spans],
        "parent_id": [span.parent_id for span in spans],
        "name": [span.name for span in spans],
        "metrics": [json.dumps(span.metrics, indent=True).replace("\n", "<br>") for span in spans],
    }
    df = pd.DataFrame(data)
    df["duration"] = df["end_time"] - df["start_time"]

    # Initialize a figure with subplots
    fig = go.Figure()

    # Add bars for each span
    colors = get_span_colors(eval_output)
    for index, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["end_time"] - row["start_time"]],
                y=[row["name"]],
                orientation="h",
                base=[row["start_time"]],  # Starting point of each bar
                marker=dict(color=colors.get(row["name"], "#ff0000")),  # Use resource_id as color
                name=row["name"],  # Label each bar with span_id
                hovertext=f"{row['duration']:.3f} seconds\n{row['metrics']}",
            )
        )

    # Update layout for clarity
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Task",
        title="Execution Timeline",
        barmode="overlay",
        showlegend=False,
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    resource_usage_table = []
    latencies_table = []

    for metric, value in interaction_output.resource_usage.items():
        resource_usage_table.append([metric, value])

    for metric, value in interaction_output.latencies.items():
        latencies_table.append([metric, value])

    st.header("Resource Usage")
    st.dataframe(
        pd.DataFrame(resource_usage_table, columns=["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    # Latencies
    st.header("Latencies")
    st.dataframe(
        pd.DataFrame(latencies_table, columns=["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    # Compute token usage per rail and per action
    rail_token_usage = {}
    action_token_usage = {}
    spans = {}
    for span in interaction_log.trace:
        spans[span.span_id] = span

        for metric, value in span.metrics.items():
            if "_tokens" in metric:
                pass


if __name__ == "__main__":
    main()
