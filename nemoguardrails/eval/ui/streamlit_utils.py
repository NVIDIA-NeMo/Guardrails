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

import argparse
import os
import random

import streamlit as st

from nemoguardrails.eval.models import EvalConfig, EvalOutput
from nemoguardrails.eval.ui.utils import EvalData
from nemoguardrails.eval.utils import get_output_paths


@st.cache_resource
def get_span_colors(_eval_output: EvalOutput):
    """Helper to get colors for the spans."""
    random.seed(4)
    colors = {}
    for log in _eval_output.logs:
        for span in reversed(log.trace):
            if span.name not in colors:
                colors[span.name] = "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    return colors


@st.cache_resource
def load_eval_data():
    """Loads the evaluation data"""
    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config-path", type=str, default="")
    parser.add_argument("--output-path", nargs="+", type=str, default=[])

    # Parse arguments
    args = parser.parse_args()

    # Load the evaluation configuration
    eval_config_path = os.path.abspath(args.eval_config_path)
    eval_config = EvalConfig.from_path(eval_config_path)

    # If no explicit output paths are provided, load all the output
    # dirs from the current folder
    if not args.output_path:
        args.output_path = get_output_paths()

    eval_outputs = {}
    for output_path in args.output_path:
        # We use relative paths to CWD to have them shorter in the UI
        output_path = os.path.relpath(output_path, os.getcwd())

        if os.path.basename(output_path).startswith("."):
            continue

        # Load the output
        eval_output = EvalOutput.from_path(output_path)
        eval_outputs[output_path] = eval_output

    return EvalData(
        eval_config_path=eval_config_path,
        output_paths=args.output_path,
        eval_config=eval_config,
        eval_outputs=eval_outputs,
    )
