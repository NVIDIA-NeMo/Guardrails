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

"""Parsing and normalization for configured rail surface references."""

from typing import Dict, NoReturn, Tuple

_HORIZONTAL_WHITESPACE = " \t"
_QUOTES = "\"'"


def _is_surface_parameter_name_start(character: str) -> bool:
    return character == "_" or "A" <= character <= "Z" or "a" <= character <= "z"


def _is_surface_parameter_name_character(character: str) -> bool:
    return _is_surface_parameter_name_start(character) or "0" <= character <= "9"


def _surface_parse_error(message: str, position: int) -> NoReturn:
    raise ValueError(f"{message} at character {position}.")


def parse_configured_surface(flow_text: str) -> Tuple[str, Dict[str, str]]:
    """Parse one complete configured surface reference.

    The accepted syntax is a bare surface name followed by zero or more
    whitespace-separated `$name=value` parameters. Spaces around `=` and
    parenthesized parameters are not supported. Values remain strings; bare
    values end at whitespace, while single- or double-quoted values may contain
    whitespace and punctuation.

    Raises:
        ValueError: If the reference is empty, contains controls, or has malformed,
            blank, adjacent, or duplicate parameters.
    """
    if any(not character.isprintable() and character not in _HORIZONTAL_WHITESPACE for character in flow_text):
        raise ValueError("Configured surface references must not contain control characters.")
    flow_text = flow_text.strip(_HORIZONTAL_WHITESPACE)
    if not flow_text:
        raise ValueError("Configured surface must not be empty.")

    parameter_start = flow_text.find("$")
    if parameter_start < 0:
        return flow_text, {}
    if parameter_start == 0 or flow_text[parameter_start - 1] not in _HORIZONTAL_WHITESPACE:
        _surface_parse_error("Parameters must be separated from the surface name", parameter_start)

    name = flow_text[:parameter_start].rstrip(_HORIZONTAL_WHITESPACE)
    parameters: Dict[str, str] = {}
    position = parameter_start
    while True:
        position += 1
        key_start = position
        if position == len(flow_text) or not _is_surface_parameter_name_start(flow_text[position]):
            _surface_parse_error("Invalid surface parameter name", position)
        position += 1
        while position < len(flow_text) and _is_surface_parameter_name_character(flow_text[position]):
            position += 1
        if position == len(flow_text) or flow_text[position] != "=":
            _surface_parse_error("Parameters must use exact $name=value syntax", position)
        key = flow_text[key_start:position]
        position += 1

        if position < len(flow_text) and flow_text[position] in _QUOTES:
            quote = flow_text[position]
            value_start = position + 1
            position = flow_text.find(quote, value_start)
            if position < 0:
                _surface_parse_error("Unterminated quoted parameter value", len(flow_text))
            value = flow_text[value_start:position]
            position += 1
        else:
            value_start = position
            while position < len(flow_text) and flow_text[position] not in _HORIZONTAL_WHITESPACE:
                if flow_text[position] == "$":
                    _surface_parse_error("Adjacent parameters must be separated by whitespace", position)
                if flow_text[position] in _QUOTES:
                    _surface_parse_error("Quoted and bare parameter values cannot be concatenated", position)
                position += 1
            value = flow_text[value_start:position]

        if not value.strip():
            _surface_parse_error("Parameters must have a non-blank value", position)
        if key in parameters:
            _surface_parse_error(f"Duplicate surface parameter {key!r}", key_start)
        parameters[key] = value

        separator_start = position
        while position < len(flow_text) and flow_text[position] in _HORIZONTAL_WHITESPACE:
            position += 1
        if position == len(flow_text):
            return name, parameters
        if position == separator_start or flow_text[position] != "$":
            _surface_parse_error("Parameters must be separated by whitespace", position)


def normalize_configured_surface_name(flow_text: str) -> str:
    """Return the surface name prefix without validating its parameters.

    Malformed parameter syntax may still yield a normalized name.
    """
    flow_text = flow_text.strip()
    return flow_text.split("$", 1)[0].strip()
