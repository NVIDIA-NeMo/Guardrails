# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass

from nemoguardrails.colang.v2_x.runtime.serialization import state_to_json


@dataclass
class Foo:
    bar: str
    baz: int


def test_state_to_json_unknown_dataclass_encodes_as_dict():
    js = state_to_json({"out": Foo("ok", 1)})
    d = json.loads(js)
    # We expect unknown dataclasses to be encoded as plain dicts
    assert d["__type"] == "dict"
    out = d["value"]["out"]
    assert out["__type"] == "dict"
    assert out["value"] == {"bar": "ok", "baz": 1}
