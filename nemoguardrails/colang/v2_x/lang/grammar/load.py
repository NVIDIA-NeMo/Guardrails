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
from functools import lru_cache

from lark import Lark
from lark.indenter import PythonIndenter


@lru_cache
def load_lark_parser(grammar_path: str):
    """Helper to load a Lark parser.

    The result is cached so that it's faster in subsequent times.

    Args:
        grammar_path: The path to the .lark file with the grammar.

    Returns:
        A Lark parser instance.
    """
    with open(grammar_path, "r") as f:
        grammar = f.read()

    return Lark(
        grammar,
        start="start",
        parser="lalr",
        lexer="contextual",
        postlex=PythonIndenter(),
        propagate_positions=True,
    )
