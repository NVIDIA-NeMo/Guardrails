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

import yaml

from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v1_0.lang.colang_parser import parse_coflows_to_yml_flows


def test_1():
    content = """
    define flow
      user express greeting
      bot express greeting
    """
    result = parse_colang_file(filename="", content=content)

    print(json.dumps(result, indent=True))

    assert result["flows"][0]["elements"] == [
        {
            "_type": "UserIntent",
            "intent_name": "express greeting",
            "intent_params": {},
            "_source_mapping": {
                "filename": "",
                "line_number": 3,
                "line_text": "user express greeting",
                "comment": None,
            },
        },
        {
            "_type": "run_action",
            "action_name": "utter",
            "action_params": {"value": "express greeting"},
            "_source_mapping": {
                "filename": "",
                "line_number": 4,
                "line_text": "bot express greeting",
                "comment": None,
            },
        },
    ]


def test_2():
    content = """
        define flow
          user express greeting
          bot express greeting
          if $name == "John"
            bot greet john
        """

    result = parse_coflows_to_yml_flows(filename="", content=content, snippets={}, include_source_mapping=False)

    print(yaml.dump(result))


def test_3():
    content = """
        define flow
          user express greeting
          bot express greeting
          execute log_greeting(name="dfdf")
        """

    result = parse_coflows_to_yml_flows(filename="", content=content, snippets={}, include_source_mapping=False)

    print(yaml.dump(result))

    result = parse_colang_file(filename="", content=content)

    print(json.dumps(result, indent=True))
