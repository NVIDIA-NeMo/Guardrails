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

from typing import Optional

import plotly.express as px
import streamlit as st
from pandas import DataFrame


def plot_as_series(
    df: DataFrame, title: Optional[str] = None, range_y=None, include_table=False
):
    """Helper to plot a dataframe as individual series."""
    df = df.copy()
    df[""] = ""

    fig = px.bar(
        df,
        x="",
        y=df.columns[1],
        range_y=range_y,
        color=df.columns[0],
        barmode="group",
        title=title or df.columns[1],
        hover_data={"": False},
    )
    st.plotly_chart(fig, use_container_width=True)

    if include_table:
        with st.expander("Table", expanded=False):
            st.dataframe(df, use_container_width=True)


def plot_bar_series(
    df: DataFrame,
    title: Optional[str] = None,
    range_y=None,
    include_table: bool = False,
):
    """Helper to plot a dataframe as bar chart."""
    fig = px.bar(
        df,
        x=df.columns[1],
        y=df.columns[2],
        range_y=range_y,
        color=df.columns[0],
        barmode="group",
        title=title or df.columns[2],
    )
    st.plotly_chart(fig, use_container_width=True)

    if include_table:
        with st.expander("Table", expanded=False):
            st.dataframe(df, use_container_width=True)


def plot_matrix_series(
    df: DataFrame,
    var_name: str,
    value_name: str,
    title: Optional[str] = None,
    range_y=None,
    include_table=False,
):
    df_melted = df.melt(id_vars=["Metric"], var_name=var_name, value_name=value_name)[
        [var_name, "Metric", value_name]
    ]

    plot_bar_series(df_melted, title=title, range_y=range_y)
    if include_table:
        with st.expander("Table", expanded=False):
            st.dataframe(df, use_container_width=True)
