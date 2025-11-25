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

import re
from typing import Optional, Union

import yaml

from .utils import get_stripped_tokens, split_max

SYMBOL_TYPES = ["intent", "object", "type", "property", "utterance", "sym", "lookup"]


def parse_pattern(pattern):
    """
    Parses a pattern from the Markdown-friendly format to an internal format.
    E.g. parse_pattern("show [me](user=CURRENT) the deals I've [won](deal__status)") =
    (show me the deals I've {deal__status}, {'user': 'CURRENT'}).

    For patters with "assignment patterns" like "show [me](user=CURRENT) the deals" we
    transform it into:
    "show {user=CURRENT} the deals" with the mapping:
    {
        "user=CURRENT": "me"
    }

    :param pattern: The pattern in Markdown-friendly format.
    :return: A tuple (pattern, params) where pattern is a pattern containing only
    text and {capture} tokens, params is a dict of 'implicit' parameter values.
    """
    params = {}
    for expr_param in re.findall(r"\[(.*?)\]\((.*?)\)", pattern):
        expr = expr_param[0]
        param = expr_param[1]
        to_replace = f"[{expr}]({param})"
        value = f"<CAPTURE>{expr}"
        if "=" in param:
            value = expr

        pattern = pattern.replace(to_replace, "{" + param + "}")
        params[param] = value

    return pattern, params


def parse_md_lang(file_name, content=None):
    """Returns the language of the .md file.

    The content can be also passed as a parameter to skip reading it.

    It searches for `lang: XX` in a yaml code block.
    By default it assumes the language is English.
    """
    if content is None:
        file = open(file_name, "r")
        content = file.read()
        file.close()

    yaml_lines = []
    in_yaml = False

    # We extract meta data and identify language
    for line in content.split("\n"):
        if not line.strip():
            continue

        # parse the meta data in yaml
        if line.startswith("```yaml"):
            in_yaml = True

        elif line.endswith("```"):
            try:
                yaml_data = yaml.safe_load("\n".join(yaml_lines))
            except Exception as ex:
                raise Exception(f"Error parsing {file_name}: {ex}")

            if "lang" in yaml_data:
                return yaml_data["lang"]
            else:
                return "en"

        elif in_yaml:
            yaml_lines.append(line)

    return "en"


def _get_param_type(type_str):
    """Helper to return the type of a parameter.

    For now we use a simple heuristic:
    1. If it's a primitive type, we leave it as such.

    2. If it already has some sort of prefix with ":", we leave it as such.

    3. If it starts with lower case and it's not one of the primitive types
       then we map it to "type:..."

    2. If it starts with an upper case, then we map it to an "object:..."

    :param type_str: A string representing the type.
    :return: The actual type.
    """
    if type_str.lower() in [
        "string",
        "str",
        "text",
        "bool",
        "boolean",
        "timedate",
        "datetime",
        "int",
        "number",
        "double",
        "currency",
    ]:
        return type_str

    if ":" in type_str:
        return type_str

    if type_str[0].islower():
        return f"type:{type_str}"
    else:
        return f"object:{type_str}"


def _get_symbol_type(sym):
    """Helper to determine if a symbol is prefixed with its type.

    :param sym: The name of the symbol.
    """

    for symbol_type in SYMBOL_TYPES:
        if sym.startswith(f"{symbol_type}:"):
            return symbol_type

    return None


def _get_typed_symbol_name(sym: str, symbol_type: str):
    """Returns the symbol name prefixed with the type, if not already."""
    if _get_symbol_type(sym):
        return sym

    return f"{symbol_type}:{sym}"


def _record_utterance(
    result: dict,
    sym: str,
    symbol_params: list,
    symbol_context: Optional[str],
    symbol_meta: dict,
    symbol_context_meta: dict,
    data: Union[str, dict],
):
    """Helper to record an utterance in the .md parsing result.

    It supports both string utterances and rich utterances.

    :param result: The result to append the utterance to.
    :param sym: The current symbol e.g. "utterance:welcome"
    :param symbol_params: Any additional symbol parameters.
      It is an array like ["$role=admin", "channel.type=messenger"]
    :param symbol_context: An additional contextual expression that must be evaluated to True/False.
    :param symbol_meta: Meta information for the symbol in general.
    :param symbol_meta: Meta information for the symbol in this context.
    :param data: The data for the utterance, either string or something "rich"
    :return:
    """
    utterance_id = split_max(sym, ":", 1)[1]

    if isinstance(data, str):
        text = data
        # We replace `field` with $field
        for param in re.findall(r"`(.*?)`", text):
            text = text.replace("`" + param + "`", f"${param}")

        utterance_data = {
            "text": text,
            "_context": {},
        }
    else:
        utterance_data = {
            "elements": data if isinstance(data, list) else [data],
            "_context": {},
        }

    # if we have symbol params that start with "$", then we record them as keys
    # that need to be matched in the context
    for param in symbol_params:
        if param.startswith("$") and "=" in param:
            key, value = get_stripped_tokens(param[1:].split("="))

            utterance_data["_context"][key] = value

    # If we have a generic contextual expression, we add it.
    # (special case for the 'None' value, which will allow us to reset the context during
    # the parsing of same symbol)
    if symbol_context and symbol_context.strip() != "None":
        utterance_data["_context"]["_expression"] = symbol_context

    meta = {}

    # If we have meta information, we add it
    if symbol_meta:
        for k in symbol_meta.keys():
            meta[k] = symbol_meta[k]

    if symbol_context_meta:
        for k in symbol_context_meta.keys():
            meta[k] = symbol_context_meta[k]

    if meta:
        utterance_data["_meta"] = meta

    # if we find more than one result, we make it an array
    if utterance_id in result["utterances"]:
        if not isinstance(result["utterances"][utterance_id], list):
            result["utterances"][utterance_id] = [result["utterances"][utterance_id]]
        result["utterances"][utterance_id].append(utterance_data)
    else:
        result["utterances"][utterance_id] = utterance_data


def parse_md_file(file_name, content=None):
    """Parse a Markdown file for patterns.

    The content can be also passed as a parameter to skip reading it.

    :param file_name: A markdown file
    :param content: The content of the file.
    :return: A list of patterns.
    """
    if content is None:
        file = open(file_name, "r")
        content = file.read()
        file.close()

    sym = None

    # First we extract the language
    file_lang = parse_md_lang(file_name, content)

    result: dict = {"patterns": [], "mappings": [], "utterances": {}}

    # The supported symbol types are: "intent", "object", "utterance"
    symbol_type = "intent"
    symbol_params = []
    symbol_context = None
    symbol_meta = {}
    symbol_context_meta = {}
    idx = 0
    lines = content.split("\n")

    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1

        # Skip blank lines
        if not line:
            continue

        if line == "### IGNORE BELOW ###":
            break

        if line.startswith("#") and not line.startswith("##"):
            _type = line[1:].lower().strip()
            if _type.startswith("intent"):
                symbol_type = "intent"
            elif _type.startswith("object"):
                symbol_type = "object"
            elif _type.startswith("utterance"):
                symbol_type = "utterance"
            elif _type.startswith("property") or _type.startswith("properties"):
                symbol_type = "property"
            elif _type.startswith("type"):
                symbol_type = "type"

        # Deal with intents part
        if line.startswith("##") and not line.startswith("###"):
            sym = line[2:].strip()

            if not sym:
                raise ValueError(f"sym cannot be empty at line: {idx + 1}")

            symbol_type = _get_symbol_type(sym) or symbol_type
            symbol_params = []
            symbol_context = None
            symbol_meta = {}
            symbol_context_meta = {}

        # TODO: remove this hack to ignore lines starting with ">   "
        #   it was added for the quick demo
        if line.startswith(">") and not line.startswith(">   "):
            sym = line[1:].strip()

            if not sym:
                raise ValueError(f"sym cannot be empty at line: {idx + 1}")

            # check if we have mappings as parameters
            # e.g. symbol(param1: type1, param2: type2, ...)
            symbol_params = []
            symbol_context = None

            if "(" in sym:
                sym, symbol_params = split_max(sym, "(", 1)
                symbol_params = get_stripped_tokens(symbol_params.split(")")[0].split(","))

            # Make sure we have the type of the symbol in the name of the symbol
            symbol_type = _get_symbol_type(sym) or symbol_type
            sym = _get_typed_symbol_name(sym, symbol_type)

            # append the mappings also
            for param in symbol_params:
                # It's a mapping only if it contains ":"
                if ":" in param:
                    name, value = get_stripped_tokens(split_max(param, ":", 1))
                    result["mappings"].append((f"{sym}:{name}", _get_param_type(value)))

        # Lines starting with ">   " represent a mapping for the current symbol
        # Record the mappings also
        if line.startswith(">   "):
            parts = get_stripped_tokens(split_max(line[4:], ":", 1))

            # We have a special case for the "_context" parameter, which marks the context
            # of the symbol. So, we record it separately and use it further down the line.
            if parts[0] == "_context":
                symbol_context = parts[1]

                # We also reset the symbol context meta on context change
                symbol_context_meta = {}
                continue

            # We have another special case for "_meta_*" parameters which mark parameters
            # that must be passed as meta information to the NLG and further
            if parts[0].startswith("_meta_"):
                var_name = parts[0][6:]
                var_expr = " ".join(parts[1:])

                # we put this either in the symbol meta, or symbol context meta
                if symbol_context:
                    symbol_context_meta[var_name] = var_expr
                else:
                    symbol_meta[var_name] = var_expr

                continue

            # Make sure we have the type of the symbol in the name of the symbol
            sym = _get_typed_symbol_name(sym, symbol_type)

            # For objects, we translate the "string" type to "kb:Object:prop|partial"
            param_type = _get_param_type(parts[1])
            if symbol_type == "object" and param_type in ["string", "text"]:
                object_name = split_max(sym, ":", 1)[1]
                param_type = f"kb:{object_name}:{parts[0]}|partial"

            # TODO: figure out a cleaner way to deal with this
            # For the "type:time" type, we transform it into "lookup:time"
            if param_type == "type:time":
                param_type = "lookup:time"

            result["mappings"].append((f"{sym}:{parts[0]}", param_type))
            symbol_params.append(parts[0])

        elif line.startswith("-") or line.startswith("*"):
            if sym is None:
                raise ValueError(f"sym is none at line: {idx + 1}")
            else:
                kind = line[0]
                pattern, params = parse_pattern(line[1:].strip())

                # If we have a context for the symbol, we record it here
                if symbol_context:
                    params["_context"] = symbol_context

                # Make sure we have the type of the symbol in the name of the symbol
                sym = _get_typed_symbol_name(sym, symbol_type)

                # For intent, objects, properties and types, we record the pattern
                if symbol_type in [
                    "intent",
                    "object",
                    "property",
                    "type",
                    "sym",
                    "lookup",
                ]:
                    # For "type" symbols, we need to make sure that the capture parameter
                    # (should be only one) is specified as [bla](type_name=value)
                    # So, we need to convert:
                    # - [bla](type_name) -> [bla](type_name=bla)
                    # - [bla](value) -> [bla](type_name=bla)
                    # - [bla](value=bla2) -> [bla](type_name=bla2)
                    #
                    # Also, we need to make sure we update the pattern itself
                    if symbol_type == "type":
                        symbol_name = split_max(sym, ":", 1)[1]

                        for k in list(params.keys()):
                            if (k == "value" or k == symbol_name) and k not in symbol_params:
                                value = params[k][9:]
                                new_k = f"{symbol_name}={value}"
                                params[new_k] = value
                                del params[k]

                                pattern = pattern.replace(f"{{{k}}}", f"{{{new_k}}}")

                            elif k.startswith("value="):
                                new_k = f"{symbol_name}{k[5:]}"
                                params[new_k] = params[k]
                                del params[k]

                                pattern = pattern.replace(f"{{{k}}}", f"{{{new_k}}}")

                    # if the symbol does not start with its type, we prepend it
                    pattern_config = dict(
                        lang=file_lang,
                        type="PATTERN" if kind == "-" else "ARG",
                        sym=sym,
                        body=pattern,
                        params=params,
                    )
                    result["patterns"].append(pattern_config)

                # For utterances, we record them in the separate dict
                elif symbol_type == "utterance":
                    _record_utterance(
                        result,
                        sym,
                        symbol_params,
                        symbol_context,
                        symbol_meta,
                        symbol_context_meta,
                        data=pattern,
                    )

        # Here we're dealing with a YAML block
        elif line.startswith("```"):
            block_lines = []
            # then we fetch the whole block
            line = lines[idx]
            idx += 1

            while not line.startswith("```"):
                block_lines.append(line)
                line = lines[idx]
                idx += 1

            # we also skip the last ``` line
            idx += 1

            # at this point we need to parse the yaml block
            d = yaml.safe_load("\n".join(block_lines))

            # If we don't have an active symbol, we skip
            # (maybe we're dealing with the `lang` tag)
            if not sym:
                continue

            sym = _get_typed_symbol_name(sym, symbol_type)

            # Currently we only support the YAML block for utterances
            if symbol_type == "utterance":
                _record_utterance(
                    result,
                    sym,
                    symbol_params,
                    symbol_context,
                    symbol_meta,
                    symbol_context_meta,
                    data=d,
                )
            else:
                raise Exception(f"YAML blocks for symbol {sym} not supported.")

    return result


__all__ = ["parse_md_file"]
