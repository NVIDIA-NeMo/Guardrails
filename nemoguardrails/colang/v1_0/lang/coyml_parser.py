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

"""Module for converting CoYML to the CIL.

Converts the Conversations YAML format to the Common Intermediate Language that is used
by the coflows engine.

This also transpiles correctly to JS to be used on the client side.
"""

import json
import re
from ast import literal_eval
from typing import List

from .utils import get_stripped_tokens, split_args, split_max, word_split


def _to_value(s, remove_quotes: bool = False):
    """Helper that converts a str/dict to another value.

    It does the following:
    - if the value is "None" it is converted to None

    TODO: other useful value shorthands
    """
    if isinstance(s, str):
        # If it's a reference to a variable, we leave as is.
        if re.match(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", s):
            return s
        else:
            return literal_eval(s)
    else:
        return s


def _extract_inline_params(d_value, d_params):
    """Helper to extract inline parameters"""
    if isinstance(d_value, str) and "(" in d_value:
        d_value, params_str = get_stripped_tokens(split_max(d_value, "(", 1))

        assert params_str[-1] == ")", f"Incorrect params str: {params_str}"

        params_str = params_str[0:-1]
        param_pairs = get_stripped_tokens(split_args(params_str))

        for pair in param_pairs:
            # Skip empty pairs
            if pair == "":
                continue

            parts = word_split(pair, "=")

            if len(parts) > 1:
                assert len(parts) == 2
                param_name, param_value = parts

                d_params[param_name] = _to_value(param_value, remove_quotes=True)
            else:
                # we're dealing with an "exists" parameter
                param_name = _to_value(pair)

                d_params[param_name] = "<<IS NOT NONE>>"

    return d_value


def _dict_to_element(d):
    """Helper to turn a short-hand dictionary into an event structure.

    :param d: A dictionary in one of the supported formats
    :return:
    """
    # if there is any property that stars with ":" we transform it to "_"
    for _k in list(d.keys()):
        if _k[0] == ":":
            d["_" + _k[1:]] = d[_k]
            del d[_k]

    d_type = list(d.keys())[0]
    d_value = d[d_type]

    d_params = {}

    # if the value of the first key is a string, we see if there are any parameters,
    # but we skip for elements where it doesn't make sense
    if d_type not in ["set", "if", "while"]:
        if isinstance(d_value, str) and "(" in d_value:
            d_value = _extract_inline_params(d_value, d_params)

        elif isinstance(d_value, list):
            new_d_value = []
            for v in d_value:
                if isinstance(v, str) and "(" in v:
                    v = _extract_inline_params(v, d_params)
                new_d_value.append(v)
            d_value = new_d_value

    if d_type in ["user", "intent", "you"]:
        # remove <<IS NOT NONE>> parameters
        is_not_none_params = []
        for k in list(d_params.keys()):
            if d_params[k] == "<<IS NOT NONE>>":
                # we get rid of "$" if it exists
                del d_params[k]

                if k[0] == "$":
                    k = k[1:]

                is_not_none_params.append(k)

            elif k[0] == "$":
                # If a parameters name starts with "$" we remove it
                d_params[k[1:]] = d_params[k]
                del d_params[k]

        element = {
            "_type": "UserIntent",
            # We replace all spaces in intent names with "|"
            "intent_name": d_value,
            "intent_params": {
                # exclude the initial key and any meta properties
                # 1) **{k: _to_value(v) for k, v in d.items() if k != d_type and k[0] != "_"},
                # 2) **d_params
            },
            # Meta properties i.e. starting with "_" are added top level
            # 3) **{k: _to_value(v) for k, v in d.items() if k[0] == "_"}
        }

        # 1)
        for k in d.keys():
            if k != d_type and k[0] != "_":
                element["intent_params"][k] = _to_value(d[k])
        # 2)
        for k in d_params.keys():
            element["intent_params"][k] = d_params[k]
        # 3)
        for k in d.keys():
            if k[0] == "_":
                element[k] = _to_value(d[k])

        if is_not_none_params:
            _pp = []
            for p in is_not_none_params:
                _pp.append(f"$intent_params.{p if p[0] != '$' else p[1:]} is not None")
            element["_match"] = " and ".join(_pp)

    elif d_type in ["UtteranceUserActionFinished"]:
        element = {
            "_type": "UtteranceUserActionFinished",
            "final_transcript": d_value,
        }

    elif d_type in ["StartUtteranceBotAction"]:
        element = {
            "_type": "StartUtteranceBotAction",
            "content": d_value,
        }

    elif d_type in ["bot", "utter", "ask", "bot_ask"]:
        element = {
            "_type": "run_action",
            "action_name": "utter",
            "action_params": {
                "value": d_value,
                # 1) **{k: _to_value(v) for k, v in d.items() if k != d_type and k != "_source_mapping"},
                # 2) **d_params
            },
        }
        # 1)
        for k in d.keys():
            if k != d_type and k != "_source_mapping":
                element["action_params"][k] = _to_value(d[k])
        # 2)
        for k in d_params.keys():
            element["action_params"][k] = d_params[k]

    elif d_type in ["run", "action", "execute"]:
        #  if we have an "=" that means we're also dealing with an assignment
        action_name = d_value
        action_result_key = None

        # We extract positional parameters as "$1", "$2", etc.
        # It is a bit hackish, but we use the <<IS NOT NOT>> marker to figure out the params
        idx = 1
        positional_params = {}
        for k in list(d_params.keys()):
            if d_params[k] == "<<IS NOT NONE>>":
                positional_params[f"${idx}"] = k
                idx += 1
                del d_params[k]
        for k in positional_params.keys():
            d_params[k] = positional_params[k]

        if "=" in action_name:
            action_result_key, action_name = get_stripped_tokens(split_max(d_value, "=", 1))

            # if action_result starts with a $, which is recommended for clarity, we remove
            if action_result_key[0] == "$":
                action_result_key = action_result_key[1:]

        element = {
            "_type": "run_action",
            "action_name": action_name,
            "action_params": {
                # 1) **{k: _to_value(v) for k, v in d.items() if k != d_type and k != "_source_mapping"},
                # 2) **d_params
            },
            # The context key where the result should be stored, if any
            "action_result_key": action_result_key,
        }

        # 1)
        for k in d.keys():
            if k != d_type and k != "_source_mapping":
                element["action_params"][k] = _to_value(d[k])

        # 2)
        for k in d_params.keys():
            element["action_params"][k] = d_params[k]

    elif d_type in ["check"]:
        element = {"_type": "check", "expression": d_value}
    elif d_type in ["pass", "continue"]:
        element = {"_type": "continue"}
    elif d_type in ["stop", "abort"]:
        element = {"_type": "stop"}
    elif d_type in ["break"]:
        element = {"_type": "break"}
    elif d_type in ["return"]:
        element = {"_type": "jump", "_next": "-1", "_absolute": True}

        # Include the return values information
        if "_return_values" in d:
            element["_return_values"] = d["_return_values"]

    elif d_type in ["if"]:
        element = {
            "_type": "if",
            "expression": d_value,
            "then": d["then"],
            "else": d["else"] if "else" in d else [],
        }

    elif d_type in ["while"]:
        element = {"_type": "while", "expression": d_value, "do": d["do"]}

    elif d_type in ["set"]:
        key, expression = get_stripped_tokens(split_max(d_value, "=", 1))

        # if the key starts with a $, which is recommended for clarity, then
        # we remove it
        if key[0] == "$":
            key = key[1:]

        element = {
            "_type": "set",
            "key": key,
            "expression": expression,
        }
    elif d_type in ["checkpoint", "label"]:
        element = {"_type": "label", "name": d_value}

        # Propagate the value also
        if "value" in d:
            element["value"] = d["value"]

    elif d_type in ["goto"]:
        element = {"_type": "goto", "label": d_value}
    elif d_type in ["meta"]:
        element = {"_type": "meta", "meta": d_value}
    elif d_type in ["event"]:
        element = {
            "_type": d_value,
            # 1) **{k: _to_value(v) for k, v in d.items() if k != d_type and k != "_source_mapping"},
            # 2) **d_params
        }

        # 1)
        for k in d.keys():
            if k != d_type and k != "_source_mapping":
                element[k] = _to_value(d[k])

        # 2)
        for k in d_params.keys():
            element[k] = d_params[k]

    elif d_type in ["flow", "call", "activate"]:
        # We transform <<IS NOT NONE>> into positional parameters
        i = 0
        new_params = {}
        for k in list(d_params.keys()):
            if d_params[k] == "<<IS NOT NONE>>":
                new_params[f"${i}"] = k
            else:
                new_params[k] = d_params[k]
            i += 1

        element = {
            "_type": "flow",
            "flow_name": d_value,
            # The parameters are not used for now, but we pass them anyway
            "flow_parameters": {
                # 1) **{k: _to_value(v) for k, v in d.items() if k != d_type and k != "_source_mapping"
                #      and k != "_return_vars"},
                # 2) **new_params
            },
            "return_vars": d["_return_vars"] if "_return_vars" in d else [],
        }

        # 1)
        for k in d.keys():
            if k != d_type and k != "_source_mapping" and k != "_return_vars":
                element["flow_parameters"][k] = _to_value(d[k])
        # 2)
        for k in new_params.keys():
            element["flow_parameters"][k] = _to_value(new_params[k])

    # Element for inferring that when something happened, then something else also happened
    elif d_type in ["infer", "add", "new", "post"]:
        # currently we support only one infer
        # TODO: add support for more
        infer_event = d_value
        if isinstance(infer_event, list):
            infer_event = infer_event[0]

        # element = {
        #     "_type": "infer",
        #     "event": _dict_to_element(infer_event)
        # }

        element = {
            "_type": "run_action",
            "action_name": "create_event",
            "action_params": {
                "event": {
                    # 1)
                    # k: v for k, v in _dict_to_element(infer_event).items()
                    # if k != "_source_mapping"
                }
            },
        }
        # 1)
        dd = _dict_to_element(infer_event)
        for k in dd.keys():
            if k != "_source_mapping":
                element["action_params"]["event"][k] = dd[k]

    # For `any` element types, we first extract the elements and they will be later
    # included in the main flow
    elif d_type in ["any", "or"]:
        element = {
            "_type": "any",
            "count": len(d_value),
            "elements": [
                # 1) _dict_to_element(_d) for _d in d_value
            ],
        }
        # 1)
        for _d in d_value:
            element["elements"].append(_dict_to_element(_d))
    else:
        raise Exception(f"Unknown dict format for: {json.dumps(d)}")

    # Add the source mapping information if available
    if "_source_mapping" in d:
        element["_source_mapping"] = d["_source_mapping"]

    return element


def get_events(events_data: List):
    """Helper to convert a list of events data to 'full events'"""
    events = []

    for event in events_data:
        # Just a normalization
        if "type" in event:
            event["_type"] = event["type"]
            del event["type"]

        # if it's a dict, but without a "_type" that means it's a shorthand dict
        if "_type" not in event:
            event = _dict_to_element(event)

        events.append(event)

    return events


def _extract_elements(items: List) -> List[dict]:
    """Helper to convert a list of items data to flow elements"""
    elements = []

    i = 0
    while i < len(items):
        item = items[i]

        if isinstance(item, dict):
            # We're dealing with an element
            element = item

            # if it's a dict, but without a "_type" that means it's a shorthand dict
            if "_type" not in element:
                element = _dict_to_element(element)

            # for `if` flow elements, we have to go recursively
            if element["_type"] == "if":
                if_element = element
                then_elements = _extract_elements(if_element["then"])
                else_elements = _extract_elements(if_element["else"])

                # Remove the raw info
                del if_element["then"]
                del if_element["else"]

                if_element["_next_else"] = len(then_elements) + 1

                # Add the "if"
                elements.append(if_element)

                # Add the "then" elements
                elements.extend(then_elements)

                # if we have "else" elements, we need to adjust also add a jump
                if len(else_elements) > 0:
                    elements.append({"_type": "jump", "_next": len(else_elements) + 1})
                    if_element["_next_else"] += 1

                    # Add the "else" elements
                    elements.extend(else_elements)

            # WHILE
            elif element["_type"] == "while":
                while_element = element
                do_elements = _extract_elements(while_element["do"])
                n = len(do_elements)

                # Remove the raw info
                del while_element["do"]

                # On break we have to skip n elements and 1 jump, hence we go to n+2
                while_element["_next_on_break"] = n + 2

                # We need to compute the jumps on break and on continue for each element
                for j in range(n):
                    # however, we make sure we don't override an inner loop
                    if "_next_on_break" not in do_elements[j]:
                        do_elements[j]["_next_on_break"] = n + 1 - j
                        do_elements[j]["_next_on_continue"] = -1 * j - 1

                # Add the "while"
                elements.append(while_element)

                # Add the "do" elements
                elements.extend(do_elements)

                # Add the jump back to the while element to recheck the condition
                elements.append({"_type": "jump", "_next": -1 * (len(do_elements) + 1)})

            elif element["_type"] == "any":
                # We first append the `any` element, and then all the elements
                elements.append(element)
                elements.extend(element["elements"])

                # remove the elements array from the main element
                del element["elements"]

            else:
                elements.append(element)

        elif isinstance(item, list):
            # In this case we're dealing with a branch
            branches = [item]

            # We see if there are any more branches
            while i < len(items) - 1 and isinstance(items[i + 1], list):
                branches.append(items[i + 1])
                i += 1

            # Next, we parse the elements from each branch
            branch_path_elements = []
            for _branch in branches:
                branch_path_elements.append(_extract_elements(_branch))

            # Create the branch element and add it to the list
            branch_element = {
                "_type": "branch",
                # these are the relative positions to the current position
                "branch_heads": [],
            }
            branch_element_pos = len(elements)
            elements.append(branch_element)

            # And next, add each branch, together with a jump
            for branch_idx in range(len(branch_path_elements)):
                branch_path = branch_path_elements[branch_idx]
                # first, record the position of the branch head
                branch_element["branch_heads"].append(len(elements) - branch_element_pos)

                # Add the elements of the branch
                elements.extend(branch_path)

                # We copy the source mapping for the branch element from the first element of the firt branch
                if branch_idx == 0 and len(branch_path) > 0:
                    if "_source_mapping" in branch_path[0]:
                        branch_element["_source_mapping"] = branch_path[0]["_source_mapping"]

                # Create the jump element
                jump_element = {"_type": "jump", "_next": 1}

                # We compute how far we need to jump based on the remaining branches
                j = branch_idx + 1
                while j < len(branch_path_elements):
                    # we add +1 to the length to account for its corresponding jump
                    jump_element["_next"] += len(branch_path_elements[j]) + 1
                    j += 1

                # And finally, add the jump element
                elements.append(jump_element)
        else:
            raise Exception(f"Unknown element type: {item}")

        # Move to the next element
        i += 1

    return elements


def _resolve_gotos(elements: List[dict]) -> List[dict]:
    """Transforms all `goto` into simple jumps.

    It does two things:
    - all goto are converted to relative `jump` elements
    - all checkpoint elements are converted to `jump` elements to the next

    """
    checkpoint_idx = {}

    # First, we extract the position of the checkpoints and change them to jumps
    for i in range(len(elements)):
        element = elements[i]
        if element["_type"] == "label":
            name = element["name"]

            # just a sanity check
            if name in checkpoint_idx:
                raise Exception(f"Checkpoint {name} already defined")

            checkpoint_idx[name] = i

            element["_type"] = "jump"
            element["_next"] = 1
            element["_label"] = name
            if "value" in element and element["value"]:
                element["_label_value"] = element["value"]
                del element["value"]

            element["_debug"] = f"label: {name}"
            del element["name"]

    # Next, we resolve the goto
    for i in range(len(elements)):
        element = elements[i]
        if element["_type"] == "goto":
            checkpoint = element["label"]

            # sanity check
            if checkpoint not in checkpoint_idx:
                raise Exception(f"Checkpoint {checkpoint} not defined.")

            element["_type"] = "jump"
            element["_next"] = checkpoint_idx[checkpoint] - i
            element["_debug"] = f"goto {checkpoint}"
            del element["label"]

    return elements


def _process_ellipsis(elements):
    """Helper to process the "..." element.

    The "..." syntax is used as a syntactic sugar, to create more readable colang code.
    There will be multiple use cases for "...". The first one is for `generate_value` action.

    1. Generate Value

    When the value of a variable is assigned to "...", we use the comment right above
    as instructions to generate the value.

       ```
       # Extract the math query from the user's input.
       $math_query = ...
       ```

       will be replaced with

       ```
       $math_query = generate_value("Extract the math query from the user's input")
       ```
    """
    new_elements = []

    for i in range(len(elements)):
        element = elements[i]

        if element["_type"] == "set" and element["expression"] == "...":
            instructions = element.get("_source_mapping", {}).get("comment")
            var_name = element["key"]

            new_elements.append(
                {
                    "_type": "run_action",
                    "action_name": "generate_value",
                    "action_params": {
                        "instructions": instructions,
                    },
                    "action_result_key": var_name,
                }
            )
        else:
            new_elements.append(element)

    return new_elements


def parse_flow_elements(items):
    """Parses the flow elements from CoYML format to CIL format."""
    # Extract
    elements = _extract_elements(items)

    # And resolve goto's
    elements = _resolve_gotos(elements)

    # Finally, we proces the ellipsis syntax
    elements = _process_ellipsis(elements)

    return elements


__all__ = ["parse_flow_elements", "get_events"]
