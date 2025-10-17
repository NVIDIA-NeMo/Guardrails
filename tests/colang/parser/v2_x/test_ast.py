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

from nemoguardrails.colang.v2_x.lang.colang_ast import (
    Flow,
    If,
    Return,
    Spec,
    SpecOp,
    SpecType,
)
from nemoguardrails.colang.v2_x.lang.utils import dataclass_to_dict


def test_basic():
    flow = Flow(
        name="test",
        elements=[
            SpecOp(
                op="match",
                spec=Spec(name="user express greeting", spec_type=SpecType.FLOW),
            ),
            If(
                expression="$a > 2",
                then_elements=[
                    SpecOp(
                        op="await",
                        spec=Spec(name="bot express greeting", spec_type=SpecType.FLOW),
                    ),
                ],
            ),
            Return(),
        ],
    )

    d = dataclass_to_dict(flow)
    assert d == {
        "_source": None,
        "_type": "flow",
        "decorators": [],
        "elements": [
            {
                "_source": None,
                "_type": "spec_op",
                "op": "match",
                "info": {},
                "return_var_name": None,
                "spec": {
                    "_source": None,
                    "_type": "spec",
                    "arguments": {},
                    "members": None,
                    "name": "user express greeting",
                    "spec_type": SpecType.FLOW,
                    "var_name": None,
                    "ref": None,
                },
            },
            {
                "_source": None,
                "_type": "if",
                "else_elements": None,
                "expression": "$a > 2",
                "then_elements": [
                    {
                        "_source": None,
                        "_type": "spec_op",
                        "op": "await",
                        "info": {},
                        "return_var_name": None,
                        "spec": {
                            "_source": None,
                            "_type": "spec",
                            "arguments": {},
                            "members": None,
                            "name": "bot express greeting",
                            "spec_type": SpecType.FLOW,
                            "var_name": None,
                            "ref": None,
                        },
                    }
                ],
            },
            {"_source": None, "_type": "return", "expression": None},
        ],
        "name": "test",
        "parameters": [],
        "return_members": [],
        "source_code": None,
        "file_info": {},
    }
