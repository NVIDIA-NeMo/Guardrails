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
import re
from typing import List, Sequence
from urllib.parse import quote

from nemoguardrails.actions.validation.filter_secrets import contains_secrets

MAX_LEN = 50


def validate_input(attribute: str, validators: Sequence[str] = (), **validation_args):
    """A generic decorator that can be used by any action (class method or function) for input validation.

    Supported validation choices are: length and quote.
    """

    def _validate_input(f):
        def wrapper(*args, **kwargs):
            obj = None

            if attribute in kwargs:
                attribute_value = kwargs.get(attribute)
            else:
                obj = args[0]
                attribute_value = getattr(obj, attribute)

            if not attribute_value:
                raise ValueError(f"Attribute {attribute} is empty.")

            if "length" in validators:
                max_len = validation_args["max_len"] if "max_len" in validation_args else MAX_LEN
                if len(attribute_value) > max_len:
                    raise ValueError(f"Attribute {attribute} is too long.")

            if "quote" in validators:
                if obj:
                    setattr(obj, attribute, quote(attribute_value))
                elif attribute in kwargs:
                    kwargs[attribute] = quote(attribute_value)

            return f(*args, **kwargs)

        return wrapper

    def decorator(obj):
        if isinstance(obj, type):
            if hasattr(obj, "run") and callable(getattr(obj, "run")):
                setattr(obj, "run", _validate_input(getattr(obj, "run")))
            return obj
        else:
            return _validate_input(obj)

    return decorator


def _is_default_resp(resp):
    """Helper for detecting a default response from LangChain tools."""
    pattern = re.compile(r"^No good.*result(?: was)? found$", re.IGNORECASE)
    match = pattern.search(resp)
    if match:
        return True
    return False


def validate_response(validators: List[str] = [], **validation_args):
    """A generic decorator that can be used by any action (class method or function) for response validation.

    Supported validation choices are: length, ip_filter, is_default_resp
    """

    def _validate_response(f):
        def wrapper(*args, **kwargs):
            def filter_ip(resp: str):
                """Filter out IP addresses from the response."""

                ip_regex = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
                return re.sub(ip_regex, "", resp)

            response_value = f(*args, **kwargs)

            if "length" in validators and len(response_value) > MAX_LEN:
                raise ValueError(f"Response Attribute {response_value} is too long.")

            if "ip_filter" in validators:
                if isinstance(response_value, str):
                    response_value = filter_ip(response_value)
                elif isinstance(response_value, dict):
                    for key, value in response_value:
                        response_value[key] = filter_ip(value)

            if "is_default_resp" in validators:
                if _is_default_resp(response_value):
                    raise ValueError("Default Response received from action")

            if "filter_secrets" in validators:
                if contains_secrets(json.dumps(response_value)):
                    raise ValueError("The response contains sensitive data.")

            return response_value

        return wrapper

    def decorator(obj):
        if isinstance(obj, type):
            if hasattr(obj, "run") and callable(getattr(obj, "run")):
                setattr(obj, "run", _validate_response(getattr(obj, "run")))
            return obj
        else:
            return _validate_response(obj)

    return decorator
