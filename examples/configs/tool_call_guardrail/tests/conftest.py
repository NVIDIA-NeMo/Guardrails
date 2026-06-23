# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Put the example directory on sys.path so the flat sibling imports the example
uses (`from policy import ...`, `from scanner.scan import ...`) resolve when these
tests are collected from the repo root. The suite is otherwise stdlib-only — it
needs no NeMo Guardrails install and makes no network or LLM calls."""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)
