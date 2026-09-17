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

"""Conduct Guard integration for NeMo Guardrails.

Runtime-policy enforcement for Colang input and output rails. Every
user turn (and every bot response) is evaluated against the workspace's
Conduct Guard policy; blocked prompts short-circuit before the model
is invoked and every decision lands in a hash-chained audit trail.

The wire client and response parser live in the ``conduct-nemo-guard``
PyPI package. This module is a thin adapter that wires that client into
the NeMo Guardrails rail manifest, config schema, and Colang flows.

Install::

    pip install conduct-nemo-guard

Docs: https://conductai.ai/guard
Source: https://github.com/sseshachala/conductai
"""
