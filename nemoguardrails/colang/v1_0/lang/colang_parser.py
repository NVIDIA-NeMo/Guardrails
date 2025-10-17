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
from ast import literal_eval
from typing import List, Optional

import yaml

from .utils import (
    char_split,
    extract_main_token,
    extract_topic_object,
    get_first_key,
    get_numbered_lines,
    get_stripped_tokens,
    params_tokenize,
    parse_package_name,
    remove_token,
    split_max,
    string_hash,
    word_split,
    ws_tokenize,
)

# The full list of valid main tokens
VALID_MAIN_TOKENS = [
    # Global
    "import",
    "include",
    "use",
    "define",
    # Branching
    "when",
    "else when",
    # Elements
    "user",
    "bot",
    "event",
    "do",
    "flow",
    "allow",
    "accept",
    "disallow",
    "deny",
    "reject",
    "goto",
    "go to",
    "run",
    "set",
    "expect",
    "label",
    "checkpoint",
    # "check",
    "if",
    "else",
    "else if",
    "any",
    "infer",
    "pass",
    "continue",
    "break",
    "stop",
    "abort",
    "return",
    "done",
    "context",
    "meta",
    "log",
    "while",
    "for",
    "foreach",
]


class ColangParser:
    def __init__(
        self,
        filename: str,
        content: str,
        include_source_mapping: bool = False,
        snippets: Optional[dict] = None,
    ):
        """Parses a file in .co format to a YAML flows format

        Initializes the ColangParser.

        Args:
            filename (str): The name of the file.
            content (str): The content.
            include_source_mapping (bool): Whether to include source mapping into the flow elements.
            snippets (Optional[dict]): Snippets to use when parsing the file.
        """
        self.filename = filename
        self.content = content
        self.include_source_mapping = include_source_mapping
        self.snippets = snippets

        self.lines = get_numbered_lines(self.content)
        self.next_line = {}

        self.current_line_idx = 0
        self.current_line = {}
        self.current_indentation = 0
        self.current_namespace = ""
        self.current_namespaces = []
        self.current_indentations = []

        # What is the level of indentation required for params (1=next indentation, 2=second indentation)
        # In general, the params are included with an indentation level, however, in certain cases
        # like in a `when x`, if there are parameters required for `x`, they would need two indentation levels
        # to distinguish them from what would follow naturally after the current statement e.g. a branch
        self.current_params_indentation = 1

        # The current element i.e. user, bot, event, if ...
        self.current_element = None

        # The flows that have been parsed
        self.flows = {}

        # The imports that were found
        self.imports = []

        # This is where we gather intent/utterances content
        # TODO: replace this temporary solution with something more robust
        self.md_content = []

        # The stack of branches
        self.branches = []

        # The stack of ifs
        self.ifs = []

        # Used to keep track of the last flow interrupt/abort event
        self.last_event_flow_element = None

        self.symbol_name = None
        self.symbol_type = ""

        # Whether the current flow is an interruption flow. It influences how the
        # flow events are parsed i.e. whether they have `this` set to True or False.
        # For interruption flows it is set to `False`.
        self.is_interruption_flow: bool = False

        # If the current flow is a "response" flow then it influences how the "stop"
        # keyword is parsed.
        self.is_response_flow: bool = False

        # The current main token
        self.main_token = ""

        # The current text of the line
        self.text = ""

        # The current mode of the parser
        self.mode = "colang"

    def _normalize_line_text(self):
        """Applies a series of normalizations for a line of colang."""
        rules = []

        if self.mode == "colang":
            # First, we get rid of extra spaces
            self.text = " ".join(ws_tokenize(self.text))

            var_name = r"\$[\w.]+"

            # The label that should be used for "..." is decided dynamically, based
            # on what's on the next line
            ellipsis_label = "auto_resume"
            if self.next_line and (
                self.next_line["text"].startswith("bot ")
                or " bot " in self.next_line["text"]
            ):
                ellipsis_label = "force_interrupt"

            # Regex normalization rules
            rules = [
                # get rid of any ":" at end of line, for compatibility with python
                (r":$", r""),
                # Transform "infer x" into "infer event x" when infer is not followed by "user"/"bot"
                (r"^infer (?!user )(?!bot )(?!event )", "infer event "),
                # Transfer "(else )?when x" into "(else )?when event x"
                (
                    r"^when (?!user )(?!bot )(?!event )(?!flow )(?!no )(?!any)",
                    "when event ",
                ),
                (
                    r"^else when (?!user )(?!bot )(?!event )(?!flow )(?!no )(?!any)",
                    "else when event ",
                ),
                # replace def with define
                (r"^def ", r"define "),
                # Get rid of "says" after user/bot
                (r"^(user|bot) says(\b)", r"\1\2"),
                (r"^when (user|bot) says(\b)", r"when \1\2"),
                (r"^else when (user|bot) says(\b)", r"else when \1\2"),
                # Turn `user/bot something` into `user/bot *`
                (r"^(user|bot) something(?: else)?(\b)", r"\1 *\2"),
                (r"^when (user|bot) something(?: else)?(\b)", r"when \1 *\2"),
                (r"^else when (user|bot) something(?: else)?(\b)", r"else when \1 *\2"),
                # Turn any other "something" into "..."
                (
                    r'^((?:else )?(?:when )?(?:user|bot)\s+(?:[^"]*)\s+)something(?: else)?(\b)',
                    r"\1...\2",
                ),
                # Turn "execute"/"exec" into "run"
                (r"^execute ", r"run "),
                (r"^exec ", r"run "),
                # Alternative syntax for label
                (r"^set ([^$]+) label (?:to )?(\$.*)", r"label \1 \2"),
                # normalize running actions i.e. add the `run` in front
                (r"^(\$[\w.]+) = (?:run|execute|exec) ", r"run \1 = "),
                # normalize `set` instructions i.e. add the missing `set`
                (r"^(\$[\w.]+) = (?!run )(?!execute )(?!exec )", r"set \1 = "),
                # priority shorthand, from "priority 2" to 'meta {"priority": 2}'
                (r"^\s*priority\s*([\.\d]+)\s*$", r'meta {"priority": \1}'),
                # += operator
                (r"^(\$[\w.]+)\s*\+=", r"set \1 = \1 +"),
                # -= operator
                (r"^(\$[\w.]+)\s*\-=", r"set \1 = \1 -"),
                # Turn 'new' into 'infer'
                (r"^new( |$)", r"infer\1"),
                (r"^create( |$)", r"infer\1"),
            ]
        elif self.mode == "markdown":
            rules = [
                # Turn 'in the context of' into 'context'
                (r"^(?:in )?(?:the )?context(?: of)?(.*)$", r"context\1"),
                # Turn "expecting user x" into 'expecting("x")
                (r"expecting user (.*)$", r'expecting("\1")'),
                # Turn "context user x" into 'expecting("x")
                (r"context user (.*)$", r'context expecting("\1")'),
            ]

        for rule in rules:
            self.text = re.sub(rule[0], rule[1], self.text)

            # We have a specific case for anonymous flows.
            # We compute a hash from the content to infer the name
            if self.mode == "colang" and self.text.strip() == "define flow":
                # We add a hash computed from all the lines with a higher indentation level
                flow_text = ""
                ll = self.current_line_idx + 1
                while (
                    ll < len(self.lines)
                    and self.lines[ll]["indentation"] > self.current_line["indentation"]
                ):
                    flow_text += self.lines[ll]["text"]
                    ll += 1

                flow_hash = string_hash(flow_text)

                self.text += " anonymous-" + flow_hash

        # Below are some more advanced normalizations

        if self.mode == "colang":
            # TODO: this is a bit hackish, to think of a better way
            # if we have an "else" for a when, we turn it into "else when flow resuming"
            if self.main_token == "else":
                if (
                    len(self.ifs) == 0
                    or self.ifs[-1]["indentation"] <= self.current_indentation
                ):
                    self.text = "else when flow resuming"

    def _fetch_current_line(self):
        self.current_line = self.lines[self.current_line_idx]
        self.current_indentation = self.current_line["indentation"]
        self.current_params_indentation = 1
        self.next_line = (
            self.lines[self.current_line_idx + 1]
            if self.current_line_idx < len(self.lines) - 1
            else None
        )

        # Normalize the text of the line
        self.text = self.current_line["text"]

        # Extract the main token
        self.main_token = extract_main_token(self.text)

        # Apply the normalization step
        self._normalize_line_text()

        # Extract the main token again, in case the normalization changed the text
        self.main_token = extract_main_token(self.text)

    def _create_namespace(self, namespace):
        """create a namespace."""
        # First we need to pop all the namespaces at deeper indentation
        while (
            len(self.current_indentations) > 0
            and self.current_indentations[-1] > self.current_line["indentation"]
        ):
            self.current_indentations.pop()
            self.current_namespaces.pop()

        # Now, append the new one
        self.current_namespaces.append(namespace)
        self.current_namespace = ".".join(self.current_namespaces)
        self.current_indentation = self.next_line["indentation"]
        self.current_indentations.append(self.next_line["indentation"])

        # Reset the branches and the ifs on a new flow
        self.branches = []
        self.ifs = []

        self.current_line_idx += 1

    def _ignore_block_body(self):
        self.current_line_idx += 1

        # We also skip all indented lines i.e. the body of the snippet
        while self.current_line_idx < len(self.lines):
            if self.lines[self.current_line_idx]["indentation"] > 0:
                self.current_line_idx += 1
            else:
                break

    def _include_source_mappings(self):
        # Include the source mapping information if required
        if self.include_source_mapping:
            if self.current_element and "_source_mapping" not in self.current_element:
                self.current_element["_source_mapping"] = {
                    "filename": self.filename,
                    "line_number": self.current_line["number"],
                    "line_text": self.current_line["text"],
                    "comment": self.current_line.get("comment"),
                }

    def _record_import(self):
        self.text = remove_token(self.main_token, self.text)
        package_name = parse_package_name(self.text)

        if package_name not in self.imports:
            self.imports.append(package_name)

        self.current_line_idx += 1

    def _check_flow_exists(self):
        if self.main_token in [
            "user",
            "bot",
            "event",
            "if",
            "while",
            "for",
            "when",
            "any",
            "run",
            "label",
            "set",
            "goto",
            "go to",
            "do",
            "flow",
            "continue",
            "break",
            "stop",
            "abort",
            "done",
            "return",
            "check",
            "meta",
            "global",
            "var",
            "local",
            "param",
            "log",
        ]:
            # we make sure the current flow has been created
            if self.current_namespace not in self.flows:
                current_flow = []
                self.flows[self.current_namespace] = current_flow
                self.current_element = {}

                # initialize the branch also
                self.branches = [
                    {
                        # TODO: replace this with the elements array when migrating the
                        #   flow to a dict
                        "elements": current_flow,
                        "indentation": self.current_line["indentation"],
                    }
                ]

    def _check_ifs_and_branches(self):
        # If the current indentation is lower than the branch, we pop branches
        while (
            len(self.branches) > 0
            and self.current_indentation < self.branches[-1]["indentation"]
        ):
            self.branches.pop()

        # If the current indentation is lower than then the if, we pop the if
        while (
            len(self.ifs) > 0
            and self.current_indentation < self.ifs[-1]["indentation"]
            and (
                self.main_token != "else"
                and self.main_token != "else if"
                or self.current_indentation < self.ifs[-1]["keyword_indentation"]
            )
        ):
            self.ifs.pop()

    def _extract_markdown(self):
        """Helper to extract markdown content.

        The `self.symbol_type` and `self.symbol_name` must be set correctly before calling this.

        It will start with the next line, and use it as a reference for the indentation level.
        As long as the indentation is higher, it will keep parsing the lines as markdown.
        """
        yaml = False

        self.md_content.append(f"## {self.symbol_type}:{self.symbol_name}")
        self.current_line_idx += 1
        self.mode = "markdown"

        md_indentation = None

        # The indentation levels on which we have "if"s
        if_levels = []
        last_if_level = 0

        # The current array of context expressions, per if level.
        # all the ones up to the last one must be negated.
        expressions = {}

        while self.current_line_idx < len(self.lines):
            self._fetch_current_line()
            md_line = self.text.strip()

            # record the indentation on the first line
            if md_indentation is None:
                md_indentation = self.current_line["indentation"]

            tokens = word_split(md_line, " ")
            # Check if we're dealing with a parameter definition
            if tokens[0] in [
                "param",
                "parameter",
                "entity",
                "property",
                "attribute",
                "attr",
                "prop",
            ]:
                assert (
                    (len(tokens) == 4) or (len(tokens) == 5) and tokens[2] == "as"
                ), "Invalid parameters syntax."

                # If we have 5 tokens, we join the last two with ":".
                # This is for support for "define X as lookup Y"
                if len(tokens) == 5:
                    tokens[3] += ":" + tokens[4]
                    tokens = tokens[0:4]

                # We make sure we remove the "$" from the param name if it's used
                param_name = tokens[1]
                if param_name[0] == "$":
                    param_name = param_name[1:]

                self.md_content.append(f">   {param_name}: {tokens[3]}")

            elif tokens[0] == "set":
                var_name = tokens[1][1:]

                assert tokens[2] in ["=", "to"]

                self.md_content.append(f">   _meta_{var_name}: {' '.join(tokens[3:])}")

            elif tokens[0] in ["context"]:
                self.md_content.append(f">   _context: {' '.join(tokens[1:])}")

            elif tokens[0] in ["if", "else"] and self.symbol_type == "utterance":
                # If we were in yaml mode, we stop
                if yaml:
                    self.md_content.append("```")
                    yaml = False

                if_level = self.current_indentation
                last_if_level = if_level
                if if_level not in if_levels:
                    if_levels.append(if_level)

                # We turn if's into contexts
                if tokens[0] == "if" or (
                    len(tokens) > 1 and tokens[0] == "else" and tokens[1] == "if"
                ):
                    if tokens[0] == "if":
                        expr = " ".join(tokens[1:])

                        # We reset the expressions at a level when we get to an if
                        expressions[if_level] = [expr]
                    else:
                        expr = " ".join(tokens[2:])

                        if len(expressions[if_level]) > 0:
                            # We need to negate the last one before adding the new one
                            expressions[if_level][
                                -1
                            ] = f"not({expressions[if_level][-1]})"

                        expressions[if_level].append(expr)
                else:
                    # if we're dealing with a simple else, we just negate the last expression too
                    expressions[if_level][-1] = f"not({expressions[if_level][-1]})"

                # Extract all expressions that apply to this level
                all_expressions = []
                for _if_level in if_levels:
                    if _if_level <= if_level:
                        all_expressions.extend(expressions[_if_level])

                self.md_content.append(f">   _context: {' and '.join(all_expressions)}")

            elif tokens[0] in ["bot"]:
                # We need to start a new flow that maps one on one the intent with
                # what follows next. The easiest way, is to actually alter the input
                # and add the missing lines.
                # We create a flow with the name `{self.symbol_name}_direct`.
                self.lines.insert(
                    self.current_line_idx,
                    {
                        "text": f"{self.symbol_name}_direct:",
                        # We keep the line mapping the same
                        "number": self.current_line["number"],
                        "indentation": self.current_indentation - 2,
                    },
                )
                self.lines.insert(
                    self.current_line_idx + 1,
                    {
                        "text": f"user {self.symbol_name}",
                        "number": self.current_line["number"],
                        "indentation": self.current_indentation,
                    },
                )

                # We stop with the markdown parsing here
                self.mode = "colang"
                return
            else:
                # If we don't have strings, there are two cases:
                # 1. we have an yaml object (rich utterance)
                # 2. we're dealing with multi intents and we have a reference to another intent

                # To figure out if we're dealing with an yaml, we check if we have ":" in the text.
                # If it's yaml, it will be a key definition for sure on the first line.
                if not yaml and self.text[0] != '"':
                    # We use the word_split to avoid the ":" in potential strings
                    parts = word_split(self.text, ":")

                    if len(parts) > 1 or len(parts) == 1 and self.text.endswith(":"):
                        yaml = True
                        self.mode = "yaml"
                        self.md_content.append("```yaml")

                if yaml:
                    # we don't add the stripped version as we need the proper indentation
                    self.md_content.append(
                        f"{' ' * self.current_indentation}{self.text}"
                    )
                else:
                    # we split the line in multiple components separated by " and "
                    parts = word_split(md_line, " and ")

                    # Apply some transformations for each component:
                    # - remove double quotes
                    # - replace $xyz with [x](xyz)
                    # - replace references to other intents with {intent:x}
                    for i in range(len(parts)):
                        parts[i] = parts[i].strip()

                        # get rid of double quotes
                        if parts[i][0] == '"':
                            assert parts[i][-1] == '"', 'Invalid syntax, missing "'
                            parts[i] = parts[i][1:-1]

                            # We also transform "$xyz" into "[x](xyz)", but not for utterances
                            if self.symbol_type != "utterance":
                                replaced_params = {}
                                for param in re.findall(
                                    r"\$([^ \"'!?\-,;</]*(?:\w|]))", parts[i]
                                ):
                                    if param not in replaced_params:
                                        parts[i] = parts[i].replace(
                                            f"${param}", f"[x]({param})"
                                        )
                                        replaced_params[param] = True
                        else:
                            # We're dealing with another intent here, so we prefix with "sym:"
                            # and we're also replacing spaces with "|".
                            parts[i] = "{intent:" + parts[i].replace(" ", "|") + "}"

                    # Put it all back together by joining with the intent and
                    md_line = " {intent:and} ".join(parts)

                    # If we went below the last if indentation level, we need to issue
                    # a new _context line.
                    if self.current_indentation <= last_if_level:
                        all_expressions = []
                        for _if_level in if_levels:
                            if _if_level < self.current_indentation:
                                all_expressions.extend(expressions[_if_level])

                        # If we're left with nothing, we just set a simple "True" expression
                        if len(all_expressions) == 0:
                            self.md_content.append(f">   _context: True")
                        else:
                            self.md_content.append(
                                f">   _context: {' and '.join(all_expressions)}"
                            )

                    self.md_content.append(f" - {md_line}")

            self.current_line_idx += 1

            if self.current_line_idx < len(self.lines):
                self.next_line = self.lines[self.current_line_idx]
            else:
                self.next_line = None

            # Get out of the markdown mode
            if not self.next_line or self.next_line["indentation"] < md_indentation:
                if yaml:
                    self.md_content.append("```")

                self.md_content.append("")
                self.mode = "colang"
                return

    def _process_define(self):
        # TODO: deal with "when" after "else when"
        # If there is no next line, or it is not indented, and we have a multi-intent
        # definition, then we add a default line.
        # i.e. if we have "define user X and Y" we add a line with "X and Y"
        if (
            self.next_line is None
            or self.next_line["indentation"] <= self.current_line["indentation"]
            and self.text.startswith("define user")
        ):
            self.next_line = {
                "text": self.text.replace("define user", ""),
                # We keep the line mapping the same
                "number": self.current_line["number"],
                # We take the indentation of the flow elements that follow
                "indentation": self.current_line["indentation"] + 2,
            }
            self.lines.insert(self.current_line_idx + 1, self.next_line)

        assert (
            self.next_line["indentation"] > self.current_line["indentation"]
        ), "Expected indented block after define statement."

        self.text = remove_token("define", self.text)

        # Extract what we define
        define_token = extract_main_token(self.text)
        self.symbol_name = remove_token(define_token, self.text)

        allowed_tokens = [
            "bot",
            "user",
            "flow",
            "subflow",
            "action",
        ]

        # We extract the modifiers if they are present e.g. test, interruption
        modifiers = {}

        while define_token not in allowed_tokens:
            # For interruption flows i.e interruption handlers
            if define_token in ["interruption", "repair"]:
                modifiers["is_interruption_flow"] = True

            # For test flows
            elif define_token in ["test"]:
                modifiers["is_test"] = True

            # For non-interruptable flows
            elif define_token in [
                "non-interruptable",
                "noninterruptable",
                "continuous",
            ]:
                modifiers["interruptable"] = False

            # For recursive flows
            elif define_token in ["recursive", "parallel"]:
                modifiers["allow_multiple"] = True

            # For extension flows
            elif define_token in ["extension"]:
                modifiers["is_extension"] = True

            # For sample flows
            elif define_token in ["sample"]:
                modifiers["is_sample"] = True

            # For response flows
            elif define_token in ["response"]:
                modifiers["is_response"] = True

            else:
                raise Exception(f'Unknown token: "{define_token}"')

            # Remove the modifier token
            self.text = remove_token(define_token, self.text)
            define_token = extract_main_token(self.text)
            self.symbol_name = remove_token(define_token, self.text)

        # During normal parsing, we ignore the snippets
        if define_token == "snippet" or define_token == "action":
            self._ignore_block_body()
            return

        # For the define flow syntax, we transform it into the shorthand one, and reprocess
        if define_token in ["flow", "subflow"]:
            # We add a ":" in front, to make sure that even if it starts with a valid main token
            # e.g. "define" it will not be interpreted as such
            self.lines[self.current_line_idx]["text"] = f":{self.symbol_name}:"

            # if we're dealing with a subflow, we also add the meta information
            if define_token == "subflow":
                modifiers["subflow"] = True

            # If we have modifiers, we add them as the meta information
            if modifiers:
                # If we don't have a meta block, we add it
                if self.lines[self.current_line_idx + 1]["text"] != "meta":
                    self.lines.insert(
                        self.current_line_idx + 1,
                        {
                            "text": f"meta",
                            # We keep the line mapping the same
                            "number": self.current_line["number"],
                            # We take the indentation of the flow elements that follow
                            "indentation": self.next_line["indentation"],
                        },
                    )
                    meta_indentation = self.next_line["indentation"] + 2
                else:
                    meta_indentation = self.lines[self.current_line_idx + 2][
                        "indentation"
                    ]

                # We add all modifier information
                for modifier in modifiers.keys():
                    value = modifiers[modifier]
                    self.lines.insert(
                        self.current_line_idx + 2,
                        {
                            "text": f"{modifier}: {value}",
                            # We keep the line mapping the same
                            "number": self.current_line["number"],
                            # Increase the indentation a bit
                            "indentation": meta_indentation,
                        },
                    )

            # Record whether this is an interruption flow or not
            self.is_interruption_flow = False
            if "is_interruption_flow" in modifiers:
                self.is_interruption_flow = modifiers["is_interruption_flow"]

            self.is_response_flow = False
            if "is_response" in modifiers:
                self.is_response_flow = modifiers["is_response"]

            return

        # If we're dealing with a topic, then we expand the flow definition
        if define_token == "topic":
            self._insert_topic_flow_definition()
            return

        # Compute the symbol type
        if define_token == "user":
            self.symbol_type = "intent"

            # We also normalize the name and replace spaces with "|"
            self.symbol_name = "|".join(word_split(self.symbol_name, " "))
        elif define_token == "bot" or define_token == "template":
            self.symbol_type = "utterance"
        else:
            # For type, lookup, token, it's the same
            self.symbol_type = define_token

        # Finally, we parse the markdown content
        self._extract_markdown()

    def _extract_indentation_levels(self):
        """Helper to extract the indentation levels higher than the current line."""
        indentations = []
        p = self.current_line_idx + 1

        while (
            p < len(self.lines)
            and self.lines[p]["indentation"]
            > self.lines[self.current_line_idx]["indentation"]
        ):
            if self.lines[p]["indentation"] not in indentations:
                indentations.append(self.lines[p]["indentation"])
            p += 1

        indentations.sort()
        return indentations

    def _extract_indented_lines(self):
        """Helper to extract the indented lines, relative to the current line.

        It also needs to take into account if the params should be indented one level or two.
        """
        initial_line_idx = self.current_line_idx
        p = self.current_line_idx + 1

        indented_lines = []
        while (
            p < len(self.lines)
            and self.lines[p]["indentation"]
            > self.lines[self.current_line_idx]["indentation"]
        ):
            indented_lines.append(self.lines[p])
            p += 1

        # If the params should be on the second level of indentation,
        # we check if there is a lower indentation than the first one
        if len(indented_lines) > 0 and self.current_params_indentation == 2:
            # Take the indentation of the first line, and look for one lower than that
            params_indentation = indented_lines[0]["indentation"]
            i = 0
            while i < len(indented_lines):
                if indented_lines[i]["indentation"] < params_indentation:
                    break
                i += 1

            # If we did not reach the end, then we only take the first i lines as the ones
            # for the indentation
            if i < len(indented_lines):
                indented_lines = indented_lines[0:i]
                self.current_line_idx = initial_line_idx + i - 1
            else:
                # in this case, we actually didn't have indented lines
                indented_lines = []
                self.current_line_idx = initial_line_idx
        else:
            # Advance to the last process lined
            self.current_line_idx = p - 1

        return indented_lines

    def _extract_params(self, param_lines: Optional[List] = None):
        """Helper to parse additional parameters for an element.

        We transform the indented lines into valid YAML format. It should end up being a dict
        and not a list.

        :param param_lines: If provided, these lines will be used to extract the params.
        """
        # Fetch the param lines if not already provided
        if param_lines is None:
            param_lines = self._extract_indented_lines()

        if not param_lines:
            return

        # TODO: figure out a better heuristic
        # We need to know if advanced features are use, to skip certain transformations
        raw_yaml = "\n".join([line["text"] for line in param_lines])
        advanced_yaml = "{" in raw_yaml or "[" in raw_yaml

        # We also apply a series of transformations
        for i in range(len(param_lines)):
            param_line = param_lines[i]
            next_param_line = param_lines[i + 1] if i < len(param_lines) - 1 else None

            if not advanced_yaml:
                # First, we do some normalization using regex
                rules = [
                    # parameters set with "set" are meta parameters and we prefix them with "_"
                    (r"^set \$?([\w.]+) to", r"_\1:"),
                    (r"^set \$?([\w.]+) =", r"_\1:"),
                    (r'^(".*")', r"_text: \1"),
                ]

                line_text = param_line["text"]
                for rule in rules:
                    line_text = re.sub(rule[0], rule[1], line_text)

                tokens = params_tokenize(line_text)

                # inline list e.g. `quick_replies: "good", "bad"`
                if len(tokens) > 3 and tokens[1] == ":" and tokens[3] == ",":
                    tokens = [*tokens[0:2], "[", *tokens[2:], "]"]
                elif len(tokens) > 2 and tokens[1] != ":" and tokens[2] == ",":
                    tokens = [tokens[0], ":", "[", *tokens[1:], "]"]

                # add the missing ":"
                elif len(tokens) == 2 and tokens[1] != ":" and tokens[0] != "-":
                    tokens = [tokens[0], ":", tokens[1]]

                # turn "=" into ":"
                elif len(tokens) == 3 and tokens[1] == "=":
                    tokens[1] = ":"

                # turn single element into a key or a list element
                # TODO: add support for list of dicts as this is not yet supported
                elif len(tokens) == 1:
                    if (
                        next_param_line is None
                        or next_param_line["indentation"] <= param_line["indentation"]
                    ):
                        tokens = ["-", tokens[0]]
                    else:
                        tokens = [tokens[0], ":"]

                param_line["text"] = " ".join(tokens)

        # Next, we process all the lines and create a valid YAML block
        base_indentation = param_lines[0]["indentation"]

        # yaml_lines = [" " * (line["indentation"] - base_indentation) + line["text"] for line in param_lines]
        # More verbose way that transpiles correctly
        yaml_lines = []
        for line in param_lines:
            line_indent = ""
            for i in range(line["indentation"] - base_indentation):
                line_indent += " "
            yaml_lines.append(line_indent + line["text"])

        yaml_block = "\n".join(yaml_lines)
        yaml_value = yaml.safe_load(yaml_block)

        # NOTE: this is needed to parse the syntax that is used for training LLM
        # e.g.
        # user "Set the alarm for 6am"
        #   request set alarm
        if isinstance(yaml_value, str):
            yaml_value = {"$0": yaml_value}

        # self.current_element.update(yaml_value)
        for k in yaml_value.keys():
            # if the key tarts with $, we remove it
            param_name = k
            if param_name[0] == "$":
                param_name = param_name[1:]

            self.current_element[param_name] = yaml_value[k]

    def _is_test_flow(self):
        """Returns true if the current flow is a test one.

        NOTE:
        This will not work correctly if the current position is nested inside another branch
        like an "if". But currently, it is meant for test flows, which should be linear.
        """
        branch_elements = self.branches[-1]["elements"]

        if len(branch_elements) == 0 or get_first_key(branch_elements[0]) != "meta":
            return False

        if "is_test" in branch_elements[0]["meta"]:
            return branch_elements[0]["meta"]["is_test"]

        return False

    def _is_sample_flow(self):
        """Returns true if the current flow is a sample one.

        NOTE:
        This will not work correctly if the current position is nested inside another branch
        like an "if". But currently, it is meant for test flows, which should be linear.
        """
        branch_elements = self.branches[-1]["elements"]

        if len(branch_elements) == 0 or get_first_key(branch_elements[0]) != "meta":
            return False

        if "is_sample" in branch_elements[0]["meta"]:
            return branch_elements[0]["meta"]["is_sample"]

        return False

    # PARSE METHODS FOR SPECIFIC SYMBOLS

    def _parse_when(self):
        # TODO: deal with "when" after "else when"
        assert (
            self.next_line["indentation"] > self.current_line["indentation"]
        ), "Expected indented block after 'when' statement."

        # Create the new branch
        new_branch = {"elements": [], "indentation": self.next_line["indentation"]}

        # # on else, we need to pop the previous branch
        # if self.main_token == "else when":
        #     branches.pop()

        # Add the array of elements directly into the parent branch
        self.branches[-1]["elements"].append(new_branch["elements"])

        # And append it as the last one
        self.branches.append(new_branch)

        # A bit hackish, but we now change the text to get rid of the main token
        # Essentially, we make the following transformation
        #   when user greeting
        #     bot "hi"
        # -->
        #     user greeting
        #     bot "hi"

        if self.main_token == "when":
            self.text = remove_token("when", self.text)

            # if we have a "when no" then we transform it
            if self.text.startswith("no "):
                self.text = remove_token("no", self.text)

                # And we add the
                #   continue
                # else
                #   ...
                self.lines.insert(
                    self.current_line_idx + 1,
                    {
                        "text": f"continue",
                        # We keep the line mapping the same
                        "number": self.current_line["number"],
                        "indentation": self.next_line["indentation"],
                    },
                )
                self.lines.insert(
                    self.current_line_idx + 2,
                    {
                        "text": f"else",
                        # We keep the line mapping the same
                        "number": self.current_line["number"],
                        "indentation": self.current_indentation,
                    },
                )

                # refresh the next line as it was changed
                self.next_line = self.lines[self.current_line_idx + 1]
        else:
            self.text = remove_token("when", remove_token("else", self.text))

        # We extract all the indentation levels and set the branch at the first
        branch_indentations = self._extract_indentation_levels()
        self.current_indentation = branch_indentations[0]
        new_branch["indentation"] = branch_indentations[0]

        # Also, mark that the params should be on the second indentation level for this line
        self.current_params_indentation = 2
        self.main_token = extract_main_token(self.text)

    def _parse_user(self):
        # Check if we're dealing with a "or" of intents
        # in which case we transform it into a "any"
        or_intents = False
        if " or " in self.text:
            parts = word_split(self.text, " or ")
            if len(parts) > 1:
                or_intents = True
                p = self.current_line_idx + 1
                self.lines.insert(
                    p,
                    {
                        "text": f"any",
                        # We keep the line mapping the same
                        "number": self.current_line["number"],
                        "indentation": self.current_indentation,
                    },
                )
                p += 1

                for part in parts:
                    self.lines.insert(
                        p,
                        {
                            "text": part,
                            # We keep the line mapping the same
                            "number": self.current_line["number"],
                            # This just needs to be bigger then the next indentation
                            "indentation": self.current_indentation + 8,
                        },
                    )
                    p += 1

        # Otherwise, it's a normal intent
        if not or_intents:
            user_value = split_max(self.text, " ", 1)[1].strip()

            # Support for "user intent_name as $var" syntax
            re_as_variable = r"(?P<intent>.*?)(?: as \$(?P<var>.+)$)"
            as_var_match = re.match(re_as_variable, user_value)
            as_var = None

            # If we have a match, we save the info and update the intent
            if as_var_match:
                gd = as_var_match.groupdict()
                as_var = gd["var"]
                user_value = gd["intent"]

            # Check if the with syntax is used for parameters
            re_with_params_1 = r"(?P<intent>.*?)(?: (?:with|for) (?P<vars>\$.+)$)"
            re_with_params_2 = (
                r"(?P<intent>.*?)(?: (?:with|for) (?P<vars>\w+\s*=\s*.+)$)"
            )

            match = re.match(re_with_params_1, user_value) or re.match(
                re_with_params_2, user_value
            )
            if match:
                d = match.groupdict()
                # in this case we convert it to the canonical "(" ")" syntax
                user_value = f"{d['intent']}({d['vars']})"

            # Deal with arrays self.current_line_idx.e. multi intents
            if user_value[0] == "[":
                user_value = get_stripped_tokens(user_value[1:-1].split(","))

            self.current_element = {"user": user_value}

            # if it was a quoted text, we mark that we need to resolve the intent
            if user_value[0] in ["'", '"']:
                user_value = user_value[1:-1]
                self.current_element["user"] = user_value

                # This is the special marker that this is an example for an intent
                self.current_element["_is_example"] = True

            # parse additional parameters if it's the case
            if (
                self.next_line
                and self.next_line["indentation"] > self.current_indentation
            ):
                self._extract_params()

            # Add to current branch
            self.branches[-1]["elements"].append(self.current_element)

            # If we have an "as $var" statement, we record the info
            if as_var:
                self.current_element["_as_context_variable"] = as_var

    def _parse_bot(self):
        """Parser for the `bot X` syntax.

        The syntax is quite flexible, see example_45.co for various ways.

        A meta syntax is the following:

        bot <utterance_id>? "sample_utterance"? (with/for $param=value)?
          $?param (:|=) value

          "sample_utterance_1"?
          "sample_utterance_2"?

        When id and sample utterance is included, an utterance definition is also
        generated as markdown content.
        """
        self.current_element = {}

        # We're dealing with a rich self.current_element with parameters on the next self.lines
        if self.text.strip() == "bot":
            self._extract_params()

            if "_type" not in self.current_element:
                self.current_element["_type"] = "element"

            self.current_element = {"bot": self.current_element}
        else:
            # We put this first, so it is the first key
            self.current_element["bot"] = None

            text = self.text.strip()
            text = remove_token("bot", text)

            # If it's rich element, we don't do anything
            if text[0] == "{":
                self.current_element["bot"] = literal_eval(text)

            else:
                utterance_text = None
                utterance_id = None

                # re_params_at_end = r'^.* ((?:with|for) (?:,?\s*\$?[\w.]+\s*(?:=\s*(?:"[^"]*"|\$[\w.]+|[-\d.]+))?)*)$'
                re_param_def = r'\$?[\w.]+\s*(?:=\s*(?:"[^"]*"|\$[\w.]+|[-\d.]+))?'
                re_first_param_def_without_marker = (
                    r'\$?[\w.]+\s*=\s*(?:"[^"]*"|\$[\w.]+|[-\d.]+)'
                )
                re_first_param_def_just_variable = r"\$[\w.]+"
                re_first_param_def = rf"(?:(?:{re_first_param_def_just_variable})|(?:{re_first_param_def_without_marker}))"

                # IMPORTANT! We must not mix escapes with r"" formatted strings; they don't transpile correctly to js
                # Hence, why we've extracted re_comma_space separately
                re_comma_space = r"\s*,\s*"
                re_params_at_end = f"^.* ((?:with|for) {re_first_param_def}(?:{re_comma_space}{re_param_def})*)$"

                # Recognizes a parameter assignment
                re_param_value = r'\s*\$?([\w.]+)\s*[:=]\s*("[^"]*"|\$[\w.]+|[-\d.]+)'

                # First, we need to determine if we have parameters at the end
                if re.match(re_params_at_end, text):
                    # and if we do, we need to extract them
                    params_str = re.findall(re_params_at_end, text)

                    # Should be only one
                    assert (
                        len(params_str) == 1
                    ), f"Expected only 1 parameter assignment, got {len(params_str)}."
                    params_str = params_str[0]

                    # remove the parameters from the string
                    text = text[0 : -1 * len(params_str)].strip()

                    # now, get rid of the with/for
                    params_str = split_max(params_str, " ", 1)[1].strip()

                    param_parts = word_split(params_str, ",")
                    idx = 0
                    for param_part in param_parts:
                        k_vs = re.findall(re_param_value, param_part)

                        # If it's a parameter given with name and value, we use that
                        if len(k_vs) > 0:
                            # Should be only one
                            for item in k_vs:
                                k = item[0]
                                v = item[1]

                                if v[0] == '"':
                                    v = v[1:-1]

                                self.current_element[k] = v
                        else:
                            # Otherwise, we use it as the value and try to infer the name
                            # and if not, we just use it as a positional parameter
                            v = param_part

                            # TODO: should cross check if there is an actual variable for
                            #   the specified utterance
                            if v.startswith("$") and "." not in v:
                                k = v[1:]
                            else:
                                k = f"${idx}"

                            self.current_element[k] = v

                        idx += 1

                # Next we check if we have an utterance text
                results = re.findall(r'"[^"]*"', text)
                if len(results) > 0:
                    assert (
                        len(results) == 1
                    ), f"Expected only 1 parameter assignment, got {len(results)}."
                    utterance_text = results[0]

                    # And remove it from the text
                    text = text.replace(utterance_text, "").strip()

                # If we're left with something, it is the utterance id
                if len(text) > 0:
                    utterance_id = text

                initial_line_idx = self.current_line_idx

                # Next, we look at the indented lines, to decide if there are additional
                # parameters, or additional examples
                indented_lines = self._extract_indented_lines()

                # First, we expect to have param lines, and then example lines, so, we try
                # to detect the first example line
                i = 0
                while i < len(indented_lines):
                    line_text = indented_lines[i]["text"].strip()
                    tokens = line_text.split(" ")
                    if tokens[0] == "if" or tokens[0][0] == '"':
                        break
                    i += 1

                # If we have param lines, we extract the parameters
                if i > 0:
                    self._extract_params(indented_lines[0:i])

                # If we have an utterance id and at least one example, we need to parse markdown.
                # However, we only do this for non-test flows
                if utterance_id is not None and (
                    utterance_text is not None or i < len(indented_lines)
                ):
                    if not self._is_test_flow() and not self._is_sample_flow():
                        # We need to reposition the current line, before the first line we need to parse
                        self.current_line_idx = initial_line_idx + i

                        if utterance_text is not None:
                            self.lines.insert(
                                self.current_line_idx + 1,
                                {
                                    "text": f"{utterance_text}",
                                    # We keep the line mapping the same
                                    "number": self.current_line["number"],
                                    "indentation": self.current_indentation + 2
                                    if i == len(indented_lines)
                                    else indented_lines[i]["indentation"],
                                },
                            )

                        self.symbol_type = "utterance"
                        self.symbol_name = utterance_id

                        self._extract_markdown()

                        # The extract markdown will move the current line to the last processed one
                        # so we move back one position, as it will be advanced automatically in
                        # the main loop
                        self.current_line_idx -= 1
                    else:
                        # We need to skip the lines as if they were consumed by the markdown parser
                        self.current_line_idx = initial_line_idx + len(indented_lines)

                # Finally, decide what to include in the element
                if utterance_id is None:
                    self.current_element["bot"] = {
                        "_type": "element",
                        "text": utterance_text[1:-1],
                    }

                    # if we have quick_replies, we move them in the element
                    if "quick_replies" in self.current_element:
                        self.current_element["bot"][
                            "quick_replies"
                        ] = self.current_element["quick_replies"]
                        del self.current_element["quick_replies"]
                else:
                    self.current_element["bot"] = utterance_id

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

        # If there was a bot message with a snippet, we also add an expect
        # TODO: can this be handled better?
        try:
            if "snippet" in self.current_element["bot"]:
                self.branches[-1]["elements"].append(
                    {
                        "expect": "snippet",
                        "snippet": self.current_element["bot"]["snippet"],
                    }
                )
        # noinspection PyBroadException
        except:
            pass

    def _parse_event(self):
        text = split_max(self.text, " ", 1)[1]

        # Check if the with syntax is used for parameters
        re_with_params_1 = r"(?P<event_name>.*?)(?: (?:with|for) (?P<vars>\$.+)$)"
        re_with_params_2 = (
            r"(?P<event_name>.*?)(?: (?:with|for) (?P<vars>\w+\s*=\s*.+)$)"
        )

        match = re.match(re_with_params_1, text) or re.match(re_with_params_2, text)
        if match:
            d = match.groupdict()
            # in this case we convert it to the canonical "(" ")" syntax
            text = f"{d['event_name']}({d['vars']})"

        self.current_element = {"event": text}

        # parse additional parameters if it's the case
        if self.next_line and self.next_line["indentation"] > self.current_indentation:
            self._extract_params()

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

    def _split_inline_params(self, value):
        # Check if the "with/for" syntax is used for parameters
        re_with_params_1 = r"(?P<name>.*?)(?: (?:with|for) (?P<vars>\$.+)$)"
        re_with_params_2 = r"(?P<name>.*?)(?: (?:with|for) (?P<vars>\w+\s*=\s*.+)$)"

        match = re.match(re_with_params_1, value) or re.match(re_with_params_2, value)
        if match:
            d = match.groupdict()
            # in this case we convert it to the canonical "(" ")" syntax
            value = f"{d['name']}({d['vars']})"

        parts = split_max(value, "(", 1)
        if len(parts) > 1:
            name = parts[0]
            params = value[len(name) :]
        else:
            name = value
            params = ""

        return name, params

    def _parse_do(self):
        # Otherwise, it's a normal intent
        do_value = split_max(self.text, " ", 1)[1].strip()

        flow_name, flow_params = self._split_inline_params(do_value)

        # if we need to save the return values, we store the info
        if "=" in flow_name:
            return_vars, flow_name = get_stripped_tokens(split_max(flow_name, "=", 1))
        else:
            return_vars = None

        self.current_element = {"flow": f"{flow_name}{flow_params}"}

        # parse additional parameters if it's the case
        if self.next_line and self.next_line["indentation"] > self.current_indentation:
            self._extract_params()

        # record the name of the return vars, without the $ sign
        if return_vars:
            return_vars = get_stripped_tokens(return_vars.split(","))
            return_vars = [_var[1:] if _var[0] == "$" else _var for _var in return_vars]
            self.current_element["_return_vars"] = return_vars

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_goto(self):
        if self.main_token == "goto":
            value = split_max(self.text, " ", 1)[1]
        else:
            value = split_max(self.text, " ", 2)[2]
        self.current_element = {"goto": value}

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_meta(self):
        self.current_element = {}

        # Remove the text and check if we've got something else
        self.text = remove_token("meta", self.text)
        if self.text and self.text[0] == "{":
            try:
                self.current_element = json.loads(self.text)
            except Exception:
                raise Exception(f"Bad meta value: {self.text}")

        # parse additional parameters as the content of the meta
        if self.next_line and self.next_line["indentation"] > self.current_indentation:
            self._extract_params()

        # Add the meta element if it's missing
        branch_elements = self.branches[-1]["elements"]
        if len(branch_elements) == 0 or get_first_key(branch_elements[0]) != "meta":
            branch_elements.insert(0, {"meta": {}})

        # Update the elements coming from the parameters
        for k in self.current_element.keys():
            branch_elements[0]["meta"][k] = self.current_element[k]

    def _parse_generic(self):
        value = split_max(self.text, " ", 1)[1].strip()
        # if it's a quoted string, we remove the quotes
        if value[0] in ["'", '"']:
            value = value[1:-1]

        self.current_element = {self.main_token: value}

        # parse additional parameters if it's the case
        if self.next_line and self.next_line["indentation"] > self.current_indentation:
            self._extract_params()

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_run(self):
        value = split_max(self.text, " ", 1)[1].strip()
        action_name, action_params = self._split_inline_params(value)

        self.current_element = {"run": f"{action_name}{action_params}"}

        # parse additional parameters if it's the case
        if self.next_line and self.next_line["indentation"] > self.current_indentation:
            self._extract_params()

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_label(self):
        """Supports parsing labels, with or without a value.

        e.g.
            label bla
            label speech hints $hints
        """
        name = split_max(self.text, " ", 1)[1].strip()

        # We separate the name and the value
        parts = re.findall(r'([^$"]+)(\$.*|".*")?', name)

        assert len(parts) == 1, "Invalid label syntax."
        name = parts[0][0].strip()
        value = parts[0][1] or None

        self.current_element = {"label": name}

        if value:
            # Get rid of the quotes
            if value.startswith('"'):
                value = value[1:-1]

            self.current_element["value"] = value

        # parse additional parameters if it's the case
        if self.next_line and self.next_line["indentation"] > self.current_indentation:
            self._extract_params()

        # Add to current branch
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_if_branch(self, if_condition):
        self.current_element = {"if": if_condition, "then": []}
        self.branches[-1]["elements"].append(self.current_element)

        self.ifs.append(
            {
                "element": self.current_element,
                "indentation": self.next_line["indentation"],
                # We also record this to match it with the else
                "keyword_indentation": self.current_indentation,
            }
        )

        # Add a new branch for the then part
        self.branches.append(
            {
                "elements": self.ifs[-1]["element"]["then"],
                "indentation": self.ifs[-1]["indentation"],
            }
        )

    def _parse_if(self):
        if_condition = split_max(self.text, " ", 1)[1].strip()
        self._parse_if_branch(if_condition)

    def _parse_else_if(self):
        # Add a new branch for the then part
        if_element = self.ifs[-1]["element"]
        if_element["else"] = []
        self.branches.append(
            {
                "elements": self.ifs[-1]["element"]["else"],
                "indentation": self.ifs[-1]["indentation"],
            }
        )

        # If we have a second if, we need to create a new if block
        if self.main_token == "else if":
            if_condition = split_max(self.text, " ", 2)[2].strip()
            self._parse_if_branch(if_condition)

    def _parse_while(self):
        while_condition = split_max(self.text, " ", 1)[1].strip()
        self.current_element = {"while": while_condition, "do": []}
        self.branches[-1]["elements"].append(self.current_element)

        # Add a new branch for the then part
        self.branches.append(
            {
                "elements": self.current_element["do"],
                "indentation": self.next_line["indentation"],
            }
        )

    def _parse_any(self):
        self.current_element = {
            "any": [],
        }
        self.branches[-1]["elements"].append(self.current_element)

        # Add a new branch for the then part
        self.branches.append(
            {
                "elements": self.current_element["any"],
                "indentation": self.next_line["indentation"],
            }
        )

    def _parse_infer(self):
        self.text = remove_token("infer", self.text)

        # If we have the event right after the infer keyword, we move it to the next line
        if self.text:
            self.lines.insert(
                self.current_line_idx + 1,
                {
                    "text": self.text,
                    # We keep the line mapping the same
                    "number": self.current_line["number"],
                    "indentation": self.current_indentation + 1,
                },
            )
            self.next_line = self.lines[self.current_line_idx + 1]

        self.current_element = {
            "infer": [],
        }
        self.branches[-1]["elements"].append(self.current_element)

        # Add a new branch for the then part
        self.branches.append(
            {
                "elements": self.current_element["infer"],
                "indentation": self.next_line["indentation"],
            }
        )

    def _parse_continue(self):
        self.current_element = {
            "continue": True,
        }
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_stop(self):
        self.current_element = {
            "bot": "stop",
        }
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_break(self):
        self.current_element = {
            "break": True,
        }
        self.branches[-1]["elements"].append(self.current_element)

    def _parse_return(self):
        parts = split_max(self.text, " ", 1)
        if len(parts) > 1:
            return_values = get_stripped_tokens(parts[1].split(","))
        else:
            return_values = []

        self.current_element = {
            "return": True,
        }

        if return_values:
            self.current_element["_return_values"] = return_values

        self.branches[-1]["elements"].append(self.current_element)

    def parse(self):
        while self.current_line_idx < len(self.lines):
            self._fetch_current_line()

            try:
                # If we're importing another model, we just record the import
                if self.main_token in ["import", "include"]:
                    self._record_import()
                    continue

                # if we're dealing with a definition
                elif self.main_token in ["define", "def"]:
                    self._process_define()
                    continue

                # Make sure we get rid of the finished branches/ifs
                self._check_ifs_and_branches()

                # NAMESPACE
                # if it's not a main token that makes sense and if the next
                # line is indented, then it's a new namespace (which could be a flow)
                if (
                    self.main_token not in VALID_MAIN_TOKENS
                    and self.next_line
                    and self.next_line["indentation"] > self.current_line["indentation"]
                ):
                    # We can only create a namespace if there are no elements in the current branch
                    # or there is no current branch
                    if (
                        len(self.branches) == 0
                        or len(self.branches[-1]["elements"]) == 0
                    ):
                        namespace = self.text
                        # We make sure to remove the pre-pended ":" if it's the case
                        if namespace.startswith(":"):
                            namespace = namespace[1:]
                        self._create_namespace(namespace)
                        continue

                # Make sure we have an active flow at this point
                self._check_flow_exists()

                # Create a new branch on "when" or "else when".
                # This will alter the text of the current line to to processed further
                # after the new branch is created
                if self.main_token in ["when", "else when"]:
                    self._parse_when()

                # Now we parse the main content of the line, according to the main token
                if self.main_token == "user":
                    self._parse_user()
                elif self.main_token == "bot":
                    self._parse_bot()
                elif self.main_token == "event":
                    self._parse_event()
                elif self.main_token in ["do"]:
                    self._parse_do()
                elif self.main_token in ["goto", "go to"]:
                    self._parse_goto()
                elif self.main_token in ["meta"]:
                    self._parse_meta()
                elif self.main_token in ["set", "expect", "check"]:
                    self._parse_generic()
                elif self.main_token in ["run"]:
                    self._parse_run()
                elif self.main_token in ["label", "checkpoint"]:
                    self._parse_label()
                elif self.main_token == "if":
                    self._parse_if()
                elif self.main_token == "while":
                    self._parse_while()
                elif self.main_token in ["else", "else if"]:
                    self._parse_else_if()
                elif self.main_token == "any":
                    self._parse_any()
                elif self.main_token == "infer":
                    self._parse_infer()
                elif self.main_token in ["pass", "continue"]:
                    self._parse_continue()
                elif self.main_token in ["stop", "abort"]:
                    self._parse_stop()
                elif self.main_token in ["break"]:
                    self._parse_break()
                elif self.main_token in ["return", "done"]:
                    self._parse_return()
                else:
                    raise Exception(
                        f"Unknown main token '{self.main_token}' on line {self.current_line['number']}"
                    )

                # Include the source mappings if needed
                self._include_source_mappings()

            except Exception as ex:
                error = f"Error parsing line {self.current_line['number']} in {self.filename}: {ex}"
                exception = Exception(error)

                # Decorate the exception with where the parsing failed
                exception.filename = self.filename
                exception.line = self.current_line["number"]
                exception.error = str(ex)

                raise exception

            self.current_line_idx += 1

        result = {"flows": self.flows}

        if self.imports:
            result["imports"] = self.imports

        if self.md_content:
            result["markdown"] = "\n".join(self.md_content)

        return result

    def _extract_snippet_name(self):
        """Helper to extract the name of a snippet. Also updates self.text."""
        # we need to figure out when the parameters begin, so we can extract the name
        # of the snippet, which can have spaces in it
        snippet_params_start_pos = 0
        while snippet_params_start_pos < len(self.text):
            if (
                self.text[snippet_params_start_pos] == '"'
                or self.text[snippet_params_start_pos] == "<"
            ):
                break
            else:
                snippet_params_start_pos += 1

        snippet_name = self.text[0:snippet_params_start_pos].strip()
        self.text = self.text[snippet_params_start_pos:]

        return snippet_name

    def parse_snippets_and_imports(self):
        """Parses just the snippets and imports from the file.

        The data is returned in the format
        {
            "snippet_name": {
                "name": "snippet_name",
                "params": ["T", "A"],
                "lines": <numbered lines>
            }
        }, ["skill_1", ...]
        """
        snippets = {}
        imports = []
        snippet = None

        while self.current_line_idx < len(self.lines):
            self._fetch_current_line()

            # If we are in a snippet, we just record the line
            if snippet:
                if self.current_line["indentation"] == 0:
                    # this means the snippet just ended
                    snippet = None
                else:
                    d = {}
                    for k in self.current_line.keys():
                        d[k] = self.current_line[k]
                    d["filename"] = self.filename
                    snippet["lines"].append(d)

                    self.current_line_idx += 1
                    continue

            if self.main_token == "define":
                self.text = remove_token("define", self.text)
                define_token = extract_main_token(self.text)

                if define_token == "snippet":
                    self.text = remove_token(define_token, self.text)
                    snippet_name = self._extract_snippet_name()

                    # Extract the params and get rid of the surrounding tags
                    param_names = re.findall("(<[^>]+>)", self.text)
                    param_names = [param[1:-1] for param in param_names]

                    snippet = {"name": snippet_name, "params": param_names, "lines": []}
                    snippets[snippet["name"]] = snippet

            elif self.main_token in ["import", "include"]:
                self.text = remove_token(self.main_token, self.text)
                package_name = parse_package_name(self.text)

                if package_name not in imports:
                    imports.append(package_name)

            self.current_line_idx += 1

        return snippets, imports


def parse_coflows_to_yml_flows(
    filename: str,
    content: str,
    include_source_mapping: bool = False,
    snippets: Optional[dict] = None,
):
    """
    Parses a file in .co format to a YAML flows format.

    Parameters:
    - filename (str): The name of the file.
    - content (str): The content of the file.
    - include_source_mapping (bool): Whether to include source mapping into the flow elements.
    - snippets (Optional[dict]): Snippets to use when parsing the file.

    Returns:
    The parsed YAML flows.

    Example:
    ```python
    filename = "example.co"
    content = "..."
    flows = parse_coflows_to_yml_flows(filename, content, include_source_mapping=True)
    ```

    Note:
    - The function uses the ColangParser to convert .co format to YAML flows.
    """
    parser = ColangParser(filename, content, include_source_mapping, snippets)

    return parser.parse()


def parse_snippets_and_imports(filename: str, content: str):
    """
    Parses snippets and imports from a file.

    Parameters:
    - filename (str): The name of the file.
    - content (str): The content of the file.

    Returns:
    A dictionary of snippets and a list of skill names.

    Example:
    ```python
    filename = "example.co"
    content = "..."
    snippets, skills = parse_snippets_and_imports(filename, content)
    ```

    Note:
    - The function uses the ColangParser to extract snippets and imports.
    """
    parser = ColangParser(filename, content)

    return parser.parse_snippets_and_imports()


__all__ = ["ColangParser", "parse_coflows_to_yml_flows", "parse_snippets_and_imports"]
