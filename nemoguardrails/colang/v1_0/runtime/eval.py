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

import re

from simpleeval import simple_eval

from nemoguardrails.colang.v1_0.runtime.utils import AttributeDict


def eval_expression(expr, context):
    """
    Evaluates the provided expression in the given context.

    Args:
        expr (str): The expression to be evaluated.
        context (dict): The context containing variable values.

    Returns:
        The result of evaluating the expression.

    Raises:
        Exception: If there is an error evaluating the expression.
    """
    # If it's not a string, we should return it as such
    if expr is None:
        return None

    if not isinstance(expr, str):
        assert isinstance(expr, bool) or isinstance(expr, int)

        return expr

    # We search for all variable names starting with $, remove the $ and add
    # the value in the globals dict for eval
    var_names = re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", expr)
    updated_expr = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", r"var_\1", expr)
    expr_locals = {}

    for var_name in var_names:
        # if we've already computed the value, we skip
        if f"var_{var_name}" in expr_locals:
            continue

        val = context.get(var_name)

        # We transform dicts to AttributeDict so we can access their keys as attributes
        # e.g. write things like $speaker.name
        if isinstance(val, dict):
            val = AttributeDict(val)

        expr_locals[f"var_{var_name}"] = val

    # Finally, just evaluate the expression
    try:
        # TODO: replace this with something even more restrictive.
        return simple_eval(updated_expr, names=expr_locals, functions={"len": len})
    except Exception as ex:
        raise Exception(f"Error evaluating '{expr}': {str(ex)}")
