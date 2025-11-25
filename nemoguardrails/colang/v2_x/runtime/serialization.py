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
"""Utilities for serializing and deserializing state objects to and from JSON."""

import functools
import json
from collections import deque
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Any, Dict

from nemoguardrails.colang.v2_x.lang import colang_ast as colang_ast_module
from nemoguardrails.colang.v2_x.runtime import flows as flows_module
from nemoguardrails.colang.v2_x.runtime.flows import Action, State
from nemoguardrails.colang.v2_x.runtime.statemachine import _flow_head_changed
from nemoguardrails.rails.llm.config import RailsConfig

# Load dynamically a map of all classes from the `colang_ast` and `flows` module.
# This is used on the decoding part.
name_to_class = {}
for module in [colang_ast_module, flows_module]:
    for attribute_name in dir(module):
        # Get the attribute
        attribute = getattr(module, attribute_name)
        # Check if the attribute is a class and defined in my_module (excluding built-in classes)
        if isinstance(attribute, type) and attribute.__module__ == module.__name__:
            # Map the class name to the class itself
            name_to_class[attribute_name] = attribute


def encode_to_dict(obj: Any, refs: Dict[int, Any]):
    """Helper to encode a hierarchy of objects to a dict.

    The encoding is able to mark correctly references to the same object. When an
    object is encountered a second time, only a reference marker will be added:

        {"__type": "ref", "__id": obj_id}

    Args:
        obj: The object that must be encoded.
        refs: An index with all the objects that have been encoded so far.
    """
    obj_id = id(obj)

    # If we've already encoded this particular object, we just make a reference
    if obj_id in refs:
        # Increase the reference count for that object
        refs[obj_id]["__ref_count"] = refs[obj_id].get("__ref_count", 0) + 1
        # And make sure the id is also present in the dict
        refs[obj_id]["__id"] = obj_id

        return {"__type": "ref", "__id": obj_id}

    # For primitive values and lists, we leave as is
    if isinstance(obj, list):
        return [encode_to_dict(v, refs) for v in obj]
    elif isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float) or obj is None:
        return obj
    elif isinstance(obj, functools.partial):
        # We don't encode the partial functions.
        # They will be re-created afterward.
        return None
    else:
        # Otherwise, we need custom encoding with support for references
        if isinstance(obj, dict):
            value = {
                "__type": "dict",
                "value": {k: encode_to_dict(v, refs) for k, v in obj.items()},
            }
        elif is_dataclass(obj):
            value = {
                "__type": type(obj).__name__,
                "value": {k: encode_to_dict(getattr(obj, k), refs) for k in obj.__dataclass_fields__.keys()},
            }
        elif isinstance(obj, RailsConfig):
            value = {
                "__type": "RailsConfig",
                "value": {k: encode_to_dict(v, refs) for k, v in obj.model_dump().items()},
            }
        elif isinstance(obj, colang_ast_module.SpecType):
            value = {"__type": "SpecType", "value": obj.value}
        elif isinstance(obj, Action):
            value = {"__type": "Action", "value": obj.to_dict()}
        elif isinstance(obj, datetime):
            value = {"__type": "datetime", "value": obj.isoformat()}
        elif isinstance(obj, Enum):
            value = {"__type": "enum", "__class": type(obj).__name__, "value": obj.name}
        elif isinstance(obj, deque):
            value = {"__type": "deque", "value": [encode_to_dict(v, refs) for v in obj]}
        elif isinstance(obj, tuple):
            value = {"__type": "tuple", "value": [encode_to_dict(v, refs) for v in obj]}
        elif isinstance(obj, set):
            value = {"__type": "set", "value": [encode_to_dict(v, refs) for v in obj]}
        else:
            raise Exception(f"Unhandled type in encode_to_dict: {type(obj)}")

        refs[obj_id] = value

        return value


def decode_from_dict(d: Any, refs: Dict[int, Any]):
    """Helper to decode a hierarchy of objects to a dict.

    The decoding is able to correctly restore references to the same object, using
    the markers with the form:

        {"__type": "ref", "__id": obj_id}

    Args:
        d: The dictionary that must be decoded.
        refs: An index with all the objects with references that have been decoded so far.
    """
    if isinstance(d, dict):
        if "__type" in d:
            d_type = d["__type"]

            if d_type == "ref":
                # If it's a reference, we use it.
                if d["__id"] not in refs:
                    raise Exception(f"Could not find reference {d['__id']}.")

                return refs[d["__id"]]

            elif d_type == "enum":
                value = name_to_class[d["__class"]][d["value"]]

            elif d_type == "RailsConfig":
                value = {k: decode_from_dict(v, refs) for k, v in d["value"].items()}
                value = RailsConfig.model_validate(value)

            elif d_type == "SpecType":
                value = colang_ast_module.SpecType(d["value"])

            elif d_type == "Action":
                value = Action.from_dict(decode_from_dict(d["value"], refs))

            elif d_type in name_to_class:
                args = decode_from_dict(d["value"], refs)

                # Attributes starting with "_" can't be passed to the constructor
                # for dataclasses, so we set them afterward.
                obj = name_to_class[d_type](**{k: v for k, v in args.items() if k[0] != "_"})
                for k in args:
                    if k[0] == "_":
                        setattr(obj, k, args[k])
                value = obj

            elif d_type == "datetime":
                value = datetime.fromisoformat(d["value"])

            elif d_type == "deque":
                value = deque(decode_from_dict(d["value"], refs))

            elif d_type == "tuple":
                value = tuple(decode_from_dict(d["value"], refs))

            elif d_type == "dict":
                value = {k: decode_from_dict(v, refs) for k, v in d["value"].items()}

            elif d_type == "set":
                value = set(decode_from_dict(d["value"], refs))

            else:
                raise Exception(f"Unknown d_type: {d_type}")

            # If we have an id, we also keep the reference
            if "__id" in d:
                refs[d["__id"]] = value

            return value
        else:
            return {k: decode_from_dict(v, refs) for k, v in d.items()}
    elif isinstance(d, list):
        return [decode_from_dict(v, refs) for v in d]
    else:
        return d


def state_to_json(state: State, indent: bool = False):
    """Helper to encode a State object to a JSON string.

    TODO: to make the size of the JSON even smaller, we can try to minify it.

    Args:
        state: The state that must be encoded.
        indent: Whether the JSON should be nicely indented.
    """
    refs = {}
    d = encode_to_dict(state, refs)

    result = json.dumps(d, indent=indent)

    return result


def json_to_state(s: str) -> State:
    """Helper to decode a State object from a JSON string."""
    data = json.loads(s)
    state = decode_from_dict(data, refs={})

    # Redo the callbacks.
    for flow_uid, flow_state in state.flow_states.items():
        for head_id, head in flow_state.heads.items():
            head.position_changed_callback = partial(_flow_head_changed, state, flow_state)
            head.status_changed_callback = partial(_flow_head_changed, state, flow_state)
    return state
