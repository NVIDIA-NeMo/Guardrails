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

from nemoguardrails.colang.v2_x.lang.colang_ast import Spec, SpecType


def flow(name: str):
    """Convert a flow name to a FlowConfig/Spec? object."""

    flow_spec = Spec(name=name, spec_type=SpecType.FLOW, arguments={}, members=[])

    return flow_spec


def action(name: str):
    """Convert an action name to an ActionConfig?."""
    raise NotImplementedError()
