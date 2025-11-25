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

import yaml

from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v2_x.lang.colang_ast import SpecType
from nemoguardrails.utils import CustomDumper


def _flows(content):
    """Quick helper."""
    result = parse_colang_file(filename="", content=content, include_source_mapping=False, version="2.x")
    flows = [flow.to_dict() for flow in result["flows"]]

    print(yaml.dump(flows, sort_keys=False, Dumper=CustomDumper, width=1000))
    return flows


def test_1():
    flows = _flows(
        """
        flow test
          match UserIntent(intent="express greeting")
          bot express greeting
        """
    )

    assert flows == [
        {
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
                        "arguments": {"flow_id": '"test"'},
                        "members": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "var_name": None,
                        "ref": None,
                    },
                },
                {
                    "_source": None,
                    "_type": "spec_op",
                    "op": "match",
                    "info": {},
                    "return_var_name": None,
                    "spec": {
                        "_source": None,
                        "_type": "spec",
                        "arguments": {"intent": '"express greeting"'},
                        "members": None,
                        "name": "UserIntent",
                        "spec_type": SpecType.EVENT,
                        "var_name": None,
                        "ref": None,
                    },
                },
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
                },
            ],
            "name": "test",
            "parameters": [],
            "return_members": [],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        }
    ]


def test_2():
    flows = _flows(
        """
        flow test
          match user express greeting

          if $current_time < '12:00'
            bot express good morning
          else
            bot express good afternoon
        """
    )
    assert flows == [
        {
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
                        "arguments": {"flow_id": '"test"'},
                        "members": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "var_name": None,
                        "ref": None,
                    },
                },
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
                    "else_elements": [
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
                                "name": "bot express good afternoon",
                                "spec_type": SpecType.FLOW,
                                "var_name": None,
                                "ref": None,
                            },
                        }
                    ],
                    "expression": "$current_time < '12:00'",
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
                                "name": "bot express good morning",
                                "spec_type": SpecType.FLOW,
                                "var_name": None,
                                "ref": None,
                            },
                        }
                    ],
                },
            ],
            "name": "test",
            "parameters": [],
            "return_members": [],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        }
    ]


def test_3():
    flows = _flows(
        """
        flow test
          user silent $duration="5s" or user agrees
          user silent "5s"
          user ask $text="what?" $times=3
          user ask (text="what?", times=3)"""
    )
    assert flows == [
        {
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
                        "arguments": {"flow_id": '"test"'},
                        "members": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "var_name": None,
                        "ref": None,
                    },
                },
                {
                    "_source": None,
                    "_type": "spec_op",
                    "op": "await",
                    "info": {},
                    "return_var_name": None,
                    "spec": {
                        "_type": "spec_or",
                        "elements": [
                            {
                                "_source": None,
                                "_type": "spec",
                                "arguments": {"duration": '"5s"'},
                                "members": None,
                                "name": "user silent",
                                "spec_type": SpecType.FLOW,
                                "var_name": None,
                                "ref": None,
                            },
                            {
                                "_source": None,
                                "_type": "spec",
                                "arguments": {},
                                "members": None,
                                "name": "user agrees",
                                "spec_type": SpecType.FLOW,
                                "var_name": None,
                                "ref": None,
                            },
                        ],
                    },
                },
                {
                    "_source": None,
                    "_type": "spec_op",
                    "op": "await",
                    "info": {},
                    "return_var_name": None,
                    "spec": {
                        "_source": None,
                        "_type": "spec",
                        "arguments": {"$0": '"5s"'},
                        "members": None,
                        "name": "user silent",
                        "spec_type": SpecType.FLOW,
                        "var_name": None,
                        "ref": None,
                    },
                },
                {
                    "_source": None,
                    "_type": "spec_op",
                    "op": "await",
                    "info": {},
                    "return_var_name": None,
                    "spec": {
                        "_source": None,
                        "_type": "spec",
                        "arguments": {"text": '"what?"', "times": "3"},
                        "members": None,
                        "name": "user ask",
                        "spec_type": SpecType.FLOW,
                        "var_name": None,
                        "ref": None,
                    },
                },
                {
                    "_source": None,
                    "_type": "spec_op",
                    "op": "await",
                    "info": {},
                    "return_var_name": None,
                    "spec": {
                        "_source": None,
                        "_type": "spec",
                        "arguments": {"text": '"what?"', "times": "3"},
                        "members": None,
                        "name": "user ask",
                        "spec_type": SpecType.FLOW,
                        "var_name": None,
                        "ref": None,
                    },
                },
            ],
            "name": "test",
            "parameters": [],
            "return_members": [],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        }
    ]


def test_4():
    flows = _flows(
        """
        flow test
          match bot express greeting . Finished()
          match bot express greeting . Finished(status="success")
          match $action
          match $action.Finished()
          match $some_flow.some_action.Finished("success")
        """
    )
    assert flows == [
        {
            "_type": "flow",
            "decorators": [],
            "_source": None,
            "name": "test",
            "parameters": [],
            "return_members": [],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
            "elements": [
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "info": {},
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "arguments": {"flow_id": '"test"'},
                        "members": None,
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                },
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "info": {},
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "bot express greeting",
                        "spec_type": SpecType.FLOW,
                        "arguments": {},
                        "members": [
                            {
                                "_type": "spec",
                                "_source": None,
                                "name": "Finished",
                                "spec_type": SpecType.EVENT,
                                "arguments": {},
                                "members": None,
                                "var_name": None,
                                "ref": None,
                            }
                        ],
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                },
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "info": {},
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "bot express greeting",
                        "spec_type": SpecType.FLOW,
                        "arguments": {},
                        "members": [
                            {
                                "_type": "spec",
                                "_source": None,
                                "name": "Finished",
                                "spec_type": SpecType.EVENT,
                                "arguments": {"status": '"success"'},
                                "members": None,
                                "var_name": None,
                                "ref": None,
                            }
                        ],
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                },
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "info": {},
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": None,
                        "spec_type": SpecType.REFERENCE,
                        "arguments": {},
                        "members": None,
                        "var_name": "action",
                        "ref": None,
                    },
                    "return_var_name": None,
                },
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "info": {},
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": None,
                        "spec_type": SpecType.REFERENCE,
                        "arguments": {},
                        "members": [
                            {
                                "_type": "spec",
                                "_source": None,
                                "name": "Finished",
                                "spec_type": SpecType.EVENT,
                                "arguments": {},
                                "members": None,
                                "var_name": None,
                                "ref": None,
                            }
                        ],
                        "var_name": "action",
                        "ref": None,
                    },
                    "return_var_name": None,
                },
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "info": {},
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": None,
                        "spec_type": SpecType.REFERENCE,
                        "arguments": {},
                        "members": [
                            {
                                "_type": "spec",
                                "_source": None,
                                "name": "some_action",
                                "spec_type": SpecType.EVENT,
                                "arguments": {},
                                "members": None,
                                "var_name": None,
                                "ref": None,
                            },
                            {
                                "_type": "spec",
                                "_source": None,
                                "name": "Finished",
                                "spec_type": SpecType.EVENT,
                                "arguments": {"$0": '"success"'},
                                "members": None,
                                "var_name": None,
                                "ref": None,
                            },
                        ],
                        "var_name": "some_flow",
                        "ref": None,
                    },
                    "return_var_name": None,
                },
            ],
        }
    ]


def test_flow_param_defs():
    assert _flows(
        """
            flow test $name $price=2
                user express greeting
            """
    )[0]["parameters"] == [
        {"default_value_expr": None, "name": "name"},
        {"default_value_expr": "2", "name": "price"},
    ]

    assert _flows(
        """
            flow test $name
                user express greeting
            """
    )[0]["parameters"] == [
        {"default_value_expr": None, "name": "name"},
    ]

    assert _flows(
        """
            flow test($name)
                user express greeting
            """
    )[0]["parameters"] == [
        {"default_value_expr": None, "name": "name"},
    ]

    assert _flows(
        """
            flow test($name="John", $age)
                user express greeting
            """
    )[0]["parameters"] == [
        {"default_value_expr": '"John"', "name": "name"},
        {"default_value_expr": None, "name": "age"},
    ]


def test_flow_def():
    assert (
        len(
            _flows(
                """
flow main
  match UtteranceUserActionFinished()
  await UtteranceBotAction(script="Hello world!")"""
            )
        )
        > 0
    )


def test_flow_assignment_1():
    assert _flows(
        """
                flow main
                  $name = "John"
            """
    )[0]["elements"][1] == {
        "_source": None,
        "_type": "assignment",
        "expression": '"John"',
        "key": "name",
    }


def test_flow_assignment_2():
    assert _flows(
        """flow main
                  $name = $full_name"""
    )[0]["elements"][1] == {
        "_source": None,
        "_type": "assignment",
        "expression": "$full_name",
        "key": "name",
    }


def test_flow_if_1():
    assert _flows(
        """
                flow main
                  $name = $full_name
                  if $name == "John"
                    bot say "Hi, John!"
                  else
                    bot say "Hello!" """
    )[0]["elements"] == [
        {
            "_source": None,
            "_type": "spec_op",
            "op": "match",
            "info": {},
            "return_var_name": None,
            "spec": {
                "_source": None,
                "_type": "spec",
                "arguments": {"flow_id": '"main"'},
                "members": None,
                "name": "StartFlow",
                "spec_type": SpecType.EVENT,
                "var_name": None,
                "ref": None,
            },
        },
        {
            "_source": None,
            "_type": "assignment",
            "expression": "$full_name",
            "key": "name",
        },
        {
            "_source": None,
            "_type": "if",
            "else_elements": [
                {
                    "_source": None,
                    "_type": "spec_op",
                    "op": "await",
                    "info": {},
                    "return_var_name": None,
                    "spec": {
                        "_source": None,
                        "_type": "spec",
                        "arguments": {"$0": '"Hello!"'},
                        "members": None,
                        "name": "bot say",
                        "spec_type": SpecType.FLOW,
                        "var_name": None,
                        "ref": None,
                    },
                }
            ],
            "expression": '$name == "John"',
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
                        "arguments": {"$0": '"Hi, John!"'},
                        "members": None,
                        "name": "bot say",
                        "spec_type": SpecType.FLOW,
                        "var_name": None,
                        "ref": None,
                    },
                }
            ],
        },
    ]


def test_flow_if_2():
    assert _flows(
        """
                flow main
                  $name = $full_name
                  if $name == "John"
                    bot say "Hi, John!"
                  elif $name == "Michael"
                    bot say "Hi, Michael"
                  elif $name == "Mike"
                    bot say "Hi, Mike"
                  else
                    bot say "Hello!" """
    )[0]["elements"] == [
        {
            "_source": None,
            "_type": "spec_op",
            "op": "match",
            "info": {},
            "return_var_name": None,
            "spec": {
                "_source": None,
                "_type": "spec",
                "arguments": {"flow_id": '"main"'},
                "members": None,
                "name": "StartFlow",
                "spec_type": SpecType.EVENT,
                "var_name": None,
                "ref": None,
            },
        },
        {
            "_source": None,
            "_type": "assignment",
            "expression": "$full_name",
            "key": "name",
        },
        {
            "_source": None,
            "_type": "if",
            "else_elements": [
                {
                    "_source": None,
                    "_type": "if",
                    "else_elements": [
                        {
                            "_source": None,
                            "_type": "if",
                            "else_elements": [
                                {
                                    "_source": None,
                                    "_type": "spec_op",
                                    "op": "await",
                                    "info": {},
                                    "return_var_name": None,
                                    "spec": {
                                        "_source": None,
                                        "_type": "spec",
                                        "arguments": {"$0": '"Hello!"'},
                                        "members": None,
                                        "name": "bot say",
                                        "spec_type": SpecType.FLOW,
                                        "var_name": None,
                                        "ref": None,
                                    },
                                }
                            ],
                            "expression": '$name == "Mike"',
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
                                        "arguments": {"$0": '"Hi, Mike"'},
                                        "members": None,
                                        "name": "bot say",
                                        "spec_type": SpecType.FLOW,
                                        "var_name": None,
                                        "ref": None,
                                    },
                                }
                            ],
                        }
                    ],
                    "expression": '$name == "Michael"',
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
                                "arguments": {"$0": '"Hi, Michael"'},
                                "members": None,
                                "name": "bot say",
                                "spec_type": SpecType.FLOW,
                                "var_name": None,
                                "ref": None,
                            },
                        }
                    ],
                }
            ],
            "expression": '$name == "John"',
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
                        "arguments": {"$0": '"Hi, John!"'},
                        "members": None,
                        "name": "bot say",
                        "spec_type": SpecType.FLOW,
                        "var_name": None,
                        "ref": None,
                    },
                }
            ],
        },
    ]


def test_flow_assignment_3():
    assert _flows(
        """
flow main
  $user_message = $event_ref.arguments
"""
    )[0]["elements"] == [
        {
            "_source": None,
            "_type": "spec_op",
            "op": "match",
            "info": {},
            "return_var_name": None,
            "spec": {
                "_source": None,
                "_type": "spec",
                "arguments": {"flow_id": '"main"'},
                "members": None,
                "name": "StartFlow",
                "spec_type": SpecType.EVENT,
                "var_name": None,
                "ref": None,
            },
        },
        {
            "_source": None,
            "_type": "assignment",
            "expression": "$event_ref.arguments",
            "key": "user_message",
        },
    ]


def test_flow_return_values():
    """Test the different ways of defining public flow members."""
    flow = _flows(
        """
            flow a $param -> $val
              pass
            flow b -> $val = 1
              pass
            flow c -> $ret_1 = "test", $ret_2
              pass
            flow c -> $ret_1 = "test", $ret_2 = 13
              pass
            """
    )

    assert flow == [
        {
            "_type": "flow",
            "decorators": [],
            "_source": None,
            "name": "a",
            "parameters": [{"name": "param", "default_value_expr": None}],
            "return_members": [{"name": "val", "default_value_expr": None}],
            "elements": [
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "arguments": {"flow_id": '"a"'},
                        "members": None,
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                    "info": {},
                },
                {"_type": "pass_stmt", "elements": []},
            ],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        },
        {
            "_type": "flow",
            "decorators": [],
            "_source": None,
            "name": "b",
            "parameters": [],
            "return_members": [{"name": "val", "default_value_expr": "1"}],
            "elements": [
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "arguments": {"flow_id": '"b"'},
                        "members": None,
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                    "info": {},
                },
                {"_type": "pass_stmt", "elements": []},
            ],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        },
        {
            "_type": "flow",
            "decorators": [],
            "_source": None,
            "name": "c",
            "parameters": [],
            "return_members": [
                {"name": "ret_1", "default_value_expr": '"test"'},
                {"name": "ret_2", "default_value_expr": None},
            ],
            "elements": [
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "arguments": {"flow_id": '"c"'},
                        "members": None,
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                    "info": {},
                },
                {"_type": "pass_stmt", "elements": []},
            ],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        },
        {
            "_type": "flow",
            "decorators": [],
            "_source": None,
            "name": "c",
            "parameters": [],
            "return_members": [
                {"name": "ret_1", "default_value_expr": '"test"'},
                {"name": "ret_2", "default_value_expr": "13"},
            ],
            "elements": [
                {
                    "_type": "spec_op",
                    "_source": None,
                    "op": "match",
                    "spec": {
                        "_type": "spec",
                        "_source": None,
                        "name": "StartFlow",
                        "spec_type": SpecType.EVENT,
                        "arguments": {"flow_id": '"c"'},
                        "members": None,
                        "var_name": None,
                        "ref": None,
                    },
                    "return_var_name": None,
                    "info": {},
                },
                {"_type": "pass_stmt", "elements": []},
            ],
            "source_code": None,
            "file_info": {"exclude_from_llm": False, "name": ""},
        },
    ]
