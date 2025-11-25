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

import logging
import os
import re
from pathlib import Path
from typing import List, Literal, Optional

import yaml

from nemoguardrails import utils
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.utils import console

_LIBS_USED = set()
_FILES_TO_EXCLUDE_ALPHA = ["ccl.co", "core_flo_library.co"]


def migrate(
    path,
    include_main_flow=False,
    use_active_decorator: bool = True,
    from_version: Literal["1.0", "2.0-alpha"] = "1.0",
    validate: bool = True,
):
    """Migrates all .co files in a directory from the old format to the new format.

    Args:
        path (str): The path to the directory containing the files to migrate.
        include_main_flow (bool): Whether to add main flow to the files.
        use_active_decorator (bool): Whether to use the active decorator.
        from_version(str): The version of the colang files to convert from. Any of '1.0' or '2.0-alpha'.
        validate (bool): Whether to validate the files.
    """
    console.print(f"Starting migration for path: {path} from version {from_version} to latest version.")

    co_files_to_process = _get_co_files_to_process(path)
    config_files_to_process = _get_config_files_to_process(path)

    console.print("Starting migration for colang files.")
    total_files_changed = _process_co_files(
        co_files_to_process,
        from_version,
        include_main_flow,
        use_active_decorator,
        validate,
    )

    console.print("Starting migration for config files.")
    total_config_files_changed = _process_config_files(config_files_to_process)

    console.print(
        f"Finished migration for path: {path}. \nTotal files changed: {total_files_changed}, Total config files changed: {total_config_files_changed}"
    )


def convert_colang_2alpha_syntax(lines: List[str]) -> List[str]:
    """Convert a co file form v2-alpha to v2-beta

    Args:
        lines (List[str]): The lines of the file to convert
    Returns:
        List (str): The new lines of the file after successful migration.
    """

    new_lines = []
    flow_line_index = None
    meta_decorators = []
    stats = {"lines": 0, "changes": 0}

    for line in lines:
        stats["lines"] += 1
        original_line = line

        line = re.sub(r"orwhen", "or when", line)
        line = re.sub(r"flow_start_uid", "flow_instance_uid", line)
        # line = re.sub(r'r"(.*)"', r'regex("\1")', line)
        line = re.sub(r'r"(.*)"', r'regex("(\1)")', line)
        line = re.sub(r"r'(.*)'", r"regex('(\1)')", line)
        line = re.sub(r'"\{\{(.*)\}\}"', r'"{\1}"', line)
        line = re.sub(r"'\{\{(.*)\}\}'", r"'{\1}'", line)
        line = re.sub(r"findall", "find_all", line)

        # Convert triple quotes to ellipsis followed by double quotes for inline NLDs
        if line.strip().startswith("$"):
            if '"""' in line:
                line = re.sub(r'"""(.*)"""', r'..."\1"', line)
            elif "'''" in line:
                line = re.sub(r"'''(.*)'''", r"...'\1'", line)

        # Replace specific phrases based on the file
        # if "core.co" in file_path:
        line = line.replace("catch Colang errors", "notification of colang errors")
        line = line.replace("catch undefined flows", "notification of undefined flow start")
        line = line.replace(
            "catch unexpected user utterance",
            "notification of unexpected user utterance",
        )
        line = line.replace("track bot talking state", "tracking bot talking state")
        line = line.replace("track user talking state", "tracking user talking state")
        line = line.replace("track user utterance state", "tracking user talking state")
        line = line.replace("track bot utterance state", "tracking bot talking state")

        # we must import core library
        _confirm_and_tag_replace(line, original_line, "core")

        line = line.replace("poll llm request response", "polling llm request response")
        line = line.replace(
            "trigger user intent for unhandled user utterance",
            "generating user intent for unhandled user utterance",
        )
        line = line.replace("generate then continue interaction", "llm continue interaction")
        line = line.replace("track unhandled user intent state", "tracking unhandled user intent state")
        line = line.replace("respond to unhandled user intent", "continuation on unhandled user intent")

        # we must import llm library
        _confirm_and_tag_replace(line, original_line, "llm")

        line = line.replace("track visual choice selection state", "track visual choice selection state")
        line = line.replace("interruption handling bot talking", "handling bot talking interruption")
        line = line.replace("manage listening posture", "managing listening posture")
        line = line.replace("manage talking posture", "managing talking posture")
        line = line.replace("manage thinking posture", "managing thinking posture")
        line = line.replace("manage bot postures", "managing bot postures")

        # Flows that were deprecated and have no replacement
        line = line.replace(
            "manage attentive posture",
            "!!!! DEPRECATED FLOW (please remove): manage attentive posture",
        )
        line = line.replace(
            "track user presence state",
            "!!!! DEPRECATED FLOW (please remove): track user presence state",
        )
        line = line.replace(
            "user became no longer present",
            "!!!! DEPRECATED FLOW (please remove): user became no longer present",
        )

        # we must import avatar library
        _confirm_and_tag_replace(line, original_line, "avatars")

        # Apply decorators to flow definitions
        if line.strip().startswith("flow "):
            flow_line_index = len(new_lines)
            new_lines.append(line)
        elif line.strip().startswith("# meta"):
            if "loop_id" in line:
                meta_decorator = re.sub(r"#\s*meta:\s*loop_id=(.*)", r'@loop("\1")', line.lstrip())
            else:

                def replace_meta(m):
                    return "@meta(" + m.group(1).replace(" ", "_") + "=True)"

                meta_decorator = re.sub(
                    r"#\s*meta:\s*(.*)",
                    replace_meta,
                    line.lstrip(),
                )
            meta_decorators.append(meta_decorator)
        else:
            if meta_decorators and flow_line_index is not None:
                for meta_decorator in meta_decorators:
                    new_lines.insert(flow_line_index, meta_decorator)
                    flow_line_index += 1
                meta_decorators = []
            new_lines.append(line)

        if original_line != line:
            stats["changes"] += 1

    return new_lines


def convert_colang_1_syntax(lines: List[str]) -> List[str]:
    """Converts a co file from v1 format to v2.

    Args:
        lines (List[str]): The lines of the file to convert.
    Returns:
        List: The new lines of the file after successful migration.
    """

    new_lines = []
    prev_line = None
    stats = {"lines": 0, "changes": 0}

    for i, line in enumerate(lines):
        stats["lines"] += 1
        original_line = line

        # Check if the line matches the pattern $variable = ...
        # use of ellipsis in Colang 1.0
        # Based on https://github.com/NVIDIA/NeMo-Guardrails/blob/ff17a88efe70ed61580a36aaae5739f5aac6dccc/
        # nemoguardrails/colang/v1_0/lang/coyml_parser.py#L610C1-L617C84

        if i > 0 and re.match(r"\s*\$\s*.*\s*=\s*\.\.\.", line):
            # Extract the variable name
            variable_match = re.search(r"\s*\$\s*(.*?)\s*=", line)
            comment_match = re.search(r"# (.*)", lines[i - 1])
            if variable_match and comment_match:
                variable = variable_match.group(1)
                comment = comment_match.group(1) or ""
                # Extract the leading whitespace
                leading_whitespace_match = re.match(r"(\s*)", line)
                leading_whitespace = leading_whitespace_match.group(1) if leading_whitespace_match else ""
                # Replace the line, preserving the leading whitespace
                line = f'{leading_whitespace}${variable} = ... "{comment}"'

        line = _globalize_variable_assignment(line)

        # Get the next line if it exists
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        # Reset prev_line if a new block is reached
        if line.lstrip().startswith("define "):
            prev_line = None
        if "define flow" in line or "define subflow" in line:
            line = re.sub(r"[-']", " ", line)
            # We convert "define flow" to "flow"
            line = re.sub(r"define flow", "flow", line)
            # We convert "define subflow" to "flow"
            line = re.sub(r"define subflow", "flow", line)

        if line.lstrip().startswith("bot") or line.lstrip().startswith("user"):
            line = re.sub(r"[-']", " ", line)

        # Convert "create event ..." to "send  ..." while preserving indentation
        line = re.sub(r"(^\s*)create event", r"\1send", line)

        # Convert $config.* to $system.config.*
        line = re.sub(r"\$config\.", r"$system.config.", line)

        if _is_anonymous_flow(line):
            # warnings.warn("Using anonymous flow is deprecated in Colang 2.0.")
            line = _revise_anonymous_flow(line, next_line or "") + "\n"

        # We convert "define bot" to "flow bot" and set the flag
        if "define bot" in line:
            line = re.sub(r"define bot", "flow bot", re.sub(r"[-']", " ", line))
            prev_line = "bot"

        # we convert "define user" to "flow user" and set the flag
        elif "define user" in line:
            line = re.sub(r"define user", "flow user", re.sub(r"[-']", " ", line))
            prev_line = "user"

        # Convert "bot ..." to "match UtteranceBotActionFinished()"
        if line.lstrip().startswith("bot ..."):
            line = re.sub(r"(\s*)bot \.\.\.", r"\1bot said something", line)

        # Convert "user ..." to "match UtteranceUserActionFinished()"
        elif line.lstrip().startswith("user ..."):
            line = re.sub(r"(\s*)user \.\.\.", r"\1user said something", line)

        # if _is_flow(stripped_line):
        #     stripped_line = re.sub(r"[-+*/]", "", stripped_line)
        #
        # Replace "when/or when" with "when/else when"
        line = re.sub(r"else when", "or when", line)
        # converting "execute" to "await"
        line = re.sub(r"execute", "await", line)

        # convert snake_case after "await" to CamelCase only if it's a single word, i.e., it is an action not a flow.
        match = re.search(r"await ([\w\s]+)", line)
        if match:
            action_name = match.group(1)
            if "_" in action_name:
                snake_case = action_name
                camel_case = utils.snake_to_camelcase(snake_case)
                if not camel_case.endswith("Action"):
                    camel_case += "Action"
                line = line.replace(action_name, camel_case)

            elif " " in action_name:
                snake_case = action_name.replace(" ", "_")
            else:
                snake_case = action_name

            camel_case = utils.snake_to_camelcase(snake_case)
            if not camel_case.endswith("Action"):
                camel_case += "Action"
            line = line.replace(action_name, camel_case)

        # Convert "stop" to "abort"
        line = re.sub(r"stop", "abort", line)

        # Convert quoted strings after "bot" to "bot say" or "or bot say" based on the flag
        if prev_line == "bot" and re.search(r"\"(.+?)\"", line):
            line = re.sub(r"\"(.+?)\"", r'bot say "\1"', line)
            prev_line = "bot say"
        elif prev_line == "bot say" and re.search(r"\"(.+?)\"", line):
            line = re.sub(r"\"(.+?)\"", r'or bot say "\1"', line)

        # Convert quoted strings after "user" to "user said" or "or user said" based on the flag
        if prev_line == "user" and re.search(r"\"(.+?)\"", line):
            line = re.sub(r"\"(.+?)\"", r'user said "\1"', line)
            prev_line = "user said"
        elif prev_line == "user said" and re.search(r"\"(.+?)\"", line):
            line = re.sub(r"\"(.+?)\"", r'or user said "\1"', line)

        new_lines.append(line)

        if original_line != line:
            stats["changes"] += 1

    return new_lines


def _write_transformed_content_and_rename_original(file_path, new_lines, co_extension=".v1.co"):
    """Writes the transformed content to the file."""

    # set the name of the v1 file
    new_file_path_v1 = os.path.splitext(file_path)[0] + co_extension

    # rename the original file
    os.rename(file_path, new_file_path_v1)

    # write the transformed content to a new file with the original name
    return _write_to_file(file_path, new_lines)


def _write_to_file(file_path, new_lines):
    """Writes new lines to a file. If the file does not exist, it is created.

    Args:
        file_path (str): The path to the file to write to.
        new_lines (list): The new lines to write to the file.
    """
    try:
        with open(file_path, "w") as file:
            file.writelines(new_lines)
        return True
    except Exception as e:
        logging.error(f"Failed to write to file: {file_path}. Error: {str(e)}")
        return False


def _is_anonymous_flow(flow_content: str) -> bool:
    """Checks if the flow is an anonymous flow.

    an anonymous flow is a flow that does not start with a name. so it only has define flow without a name.

    Args:
        flow_content (str): The content of the flow.
    Returns:
        bool: True if the flow is anonymous, False otherwise.
    Examples:

        flow_content = "flow  "
        is_anonymous_flow(flow_content)
        # True
        flow_content = "flow my_flow"
        is_anonymous_flow(flow_content)
        # False
    """

    return re.match(r"flow\s*$", flow_content) is not None


def _is_flow(line: str) -> bool:
    """Checks if the line is a flow definition. it is a flow definition if it starts with "flow" and other words after flow."""

    return re.match(r"flow\s+\w+", line) is not None


def _globalize_variable_assignment(line):
    # split the line into tokens
    tokens = line.split()

    # here we have local scope variable assignment
    if "await" in line or "execute" in line:
        return line

    # Check if the line is a variable assignment
    if len(tokens) >= 2 and tokens[1] == "=":
        # Get the original indentation
        indentation = len(line) - len(line.lstrip())

        # Prepend the "global" keyword with the original indentation
        return " " * indentation + "global " + tokens[0] + "\n" + line
    else:
        # If not a variable assignment, return the line as is
        return line


def _revise_anonymous_flow(flow_content: str, first_message: str) -> str:
    """Revises an anonymous flow by giving it a unique name.
    Args:
        flow_content (str): The content of the anonymous flow.
        first_message (str): The first message in the flow.
    Returns:
        str: The revised flow content with a unique name.
    """

    # remove "bot" or "user" from first_message
    first_message = first_message.replace("bot", "").replace("user", "").strip()
    # Use the first few words from the user's message for the flow name
    first_message_words = first_message.split()

    # TODO: Adjust the number as needed

    flow_name = " ".join(first_message_words[:3]).replace("-", " ")

    return re.sub(r"flow\s*$", f"flow {flow_name}", flow_content)
    # return re.sub(r"flow\s*$", f"flow anonymous_{uuid.uuid4().hex}", flow_content)


def _get_flow_ids(content: str) -> List:
    """Returns the flow ids in the content.

    Args:
        content (str): The content to search for flow ids.
    Returns:
        list: A list of flow ids.

    Examples:
    content = "flow my_flow"
    get_flow_ids(content)
    # ['my_flow']
    content = "flow my_flow is better than\nflow another_flow"
    get_flow_ids(content)
    # ['my_flow is better than', 'another_flow']

    # flow user and flow bot are not considered as flow ids
    content = "flow user express greeting"
    get_flow_ids(content)
    # []

    """

    # Match any words (more than one) that comes after "flow " before new line and the first word after flow is not "user" or "bot"

    root_flow_pattern = re.compile(r"^flow\s+(?!user|bot)(.*?)$", re.IGNORECASE | re.MULTILINE)
    return root_flow_pattern.findall(content)


def _get_flow_ids_from_newlines(new_lines: List) -> List:
    """Returns the flow ids in the new lines.
    Args:
        new_lines (list): The new lines to search for flow ids.
    Returns:
        list: A list of flow ids.
    Examples:
    new_lines = ["flow my_flow is better than", "flow another_flow"]
    get_flow_ids_from_newlines(new_lines)
    # ['my_flow is better than', 'another_flow']
    """
    content = "\n".join(new_lines)
    return _get_flow_ids(content)


def _add_imports(new_lines: List, libraries: List[str]) -> List:
    for library in libraries:
        new_lines.insert(0, f"import {library}\n")
    return new_lines


def _add_main_co_file(file_path: str, libraries: Optional[List[str]] = None) -> bool:
    """Add the main co file to the given file path.
    Args:
        file_path (str): The path to the file to add the main co file to.
        libraries (list[str]): The list of libraries to import in the main co file.
    Returns:
        bool: True if the main co file was added successfully, False otherwise.
    """
    new_lines = _read_file_lines(file_path)
    if not libraries:
        libraries = list(_LIBS_USED)
    libraries.append("llm")
    new_lines = _add_imports(new_lines, libraries)

    # Add the main flow
    new_lines.append("\n")
    new_lines.append("flow main\n")
    new_lines.append("  activate llm continuation\n")

    return _write_to_file(file_path, new_lines)


def _generate_main_flow(new_lines: List) -> List:
    """Adds a 'main' flow to the new lines that activates all other flows.

    The 'main' flow is added at the beginning of the new lines. It includes an 'activate' command for each flow id found in the new lines.

    Args:
        new_lines (list): The new lines to add the main flow to.

    Returns:
        list: The new lines with the 'main' flow added at the beginning.

    Examples:
    new_lines = ["flow my_flow", "flow another_flow"]
    include_main_flow(new_lines)
    # '''flow main
    #  activate my_flow
    #  activate another_flow
    """
    # Create the 'main' flow
    main_flow = ["flow main"]

    # Add an 'activate' command for each flow id in the new lines
    flow_ids = _get_flow_ids_from_newlines(new_lines)
    # one level indentation
    _INDENT = "  "

    activation_commands = [_INDENT + "activate " + flow_id for flow_id in flow_ids]

    # add new line to the last activation command
    activation_commands[-1] += "\n\n"

    main_flow.extend(activation_commands)

    # Add the 'main' flow at the beginning of the new lines
    new_lines.insert(0, "\n".join(main_flow))

    return new_lines


def _add_active_decorator(new_lines: List) -> List:
    """Adds an '@active' decorator above each flow id in the new lines.

    Args:
        new_lines (List): The lines to add the decorators to.

    Returns:
        List: The lines with the decorators added.
    """
    decorated_lines = []

    _ACTIVE_DECORATOR = "@active"
    _NEWLINE = "\n"
    root_flow_pattern = re.compile(r"^flow\s+(?!bot)(.*?)$", re.IGNORECASE | re.MULTILINE)

    for line in new_lines:
        # if it is a root flow
        if root_flow_pattern.match(line):
            # Add the '@active' decorator above the flow id
            decorated_lines.append(_ACTIVE_DECORATOR + _NEWLINE)
        decorated_lines.append(line)
    return decorated_lines


def _get_raw_config(config_path: str):
    """read the yaml file and get rails key"""
    raw_config = None
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path) as f:
            raw_config = yaml.safe_load(f.read())

    return raw_config


def _get_rails_flows(raw_config):
    """Extracts the list of flows from the raw_config dictionary.

    Args:
        raw_config (dict): The raw configuration dictionary.

    Returns:
        list: The list of flows.
    """
    from collections import defaultdict

    flows = defaultdict(list)
    try:
        for key in raw_config["rails"]:
            if "flows" in raw_config["rails"][key]:
                flows[key].extend(raw_config["rails"][key]["flows"])
    except KeyError:
        pass
    return flows


def _generate_rails_flows(flows):
    """Generates flow definitions from the list of flows.
    Args:
        flows (dict): The dictionary of flows.
    Returns:
        str: The flow definitions.
    """
    _MAPPING = {
        "input": "flow input rails $input_text",
        "output": "flow output rails $output_text",
    }

    _GUARDRAILS_IMPORT = "import guardrails"
    _LIBRARY_IMPORT = "import nemoguardrails.library"

    flow_definitions = []
    _INDENT = "    "  # 4 spaces for indentation
    _NEWLINE = "\n"

    for key, value in flows.items():
        flow_definitions.append(_MAPPING[key] + _NEWLINE)
        for v in value:
            flow_definitions.append(_INDENT + v + _NEWLINE)
        flow_definitions.append(_NEWLINE)  # Add an empty line after each flow

    if flow_definitions:
        flow_definitions.insert(0, _GUARDRAILS_IMPORT + _NEWLINE)
        flow_definitions.insert(1, _LIBRARY_IMPORT + _NEWLINE * 2)

    return flow_definitions


def _write_rails_flows_to_file(file_path, rails_flows):
    """Writes the rails flows to a file.
    Args:
        file_path (str): The path to the file to write to.
        rails_flows (list): The rails flows to write to the file.
    """
    try:
        with open(file_path, "w") as file:
            file.writelines(rails_flows)
        return True
    except Exception as e:
        logging.error(f"Failed to write to file: {file_path}. Error: {str(e)}")
        return False


def _remove_rails_flows_from_config(raw_config):
    rails_config = raw_config.get("rails", {})
    rails_config.pop("input", None)
    rails_config.pop("output", None)
    raw_config["rails"] = rails_config

    return raw_config


def _append_colang_version(file_path):
    """Appends the Colang version "2.x" at the end of the config file."""
    with open(file_path, "a") as file:
        file.write('\ncolang_version: "2.x"\n')


def _comment_rails_flows_in_config(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    input_section = False
    output_section = False
    section_indent = None
    highest_indent = None
    section_indent = 0
    hit_rails_section = False
    for i, line in enumerate(lines):
        while not hit_rails_section:
            if "rails:" in line:
                hit_rails_section = True
                break
            i += 1
            line = lines[i]

        stripped_line = line.lstrip()
        indent = len(line) - len(stripped_line)
        if highest_indent is None and indent > 0:
            highest_indent = indent
        if "input:" in stripped_line and indent == highest_indent:
            input_section = True
            section_indent = indent
        if "output:" in stripped_line and indent == highest_indent:
            output_section = True
            section_indent = indent
        if (input_section or output_section) and indent >= section_indent:
            lines[i] = line[: line.index(stripped_line[0])] + "#" + stripped_line
        if indent < section_indent:
            input_section = False
            output_section = False

    with open(file_path, "w") as file:
        file.writelines(lines)

    with open(file_path, "r") as file:
        raw_config = yaml.safe_load(file.read())
        if not raw_config.get("rails"):
            # If 'rails' is empty, comment out the 'rails:' line
            for i, line in enumerate(lines):
                if "rails:" in line.lstrip():
                    lines[i] = "#" + line
                    break

    with open(file_path, "w") as file:
        file.writelines(lines)


def _get_config_files_to_process(path):
    """Get a list of .yaml or .yml files to process.

    Args:
        path (str): The path to the file or directory to process.

    Returns:
        list: The list of .yaml or .yml files to process.
    """

    config_files_to_process = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(path)
        for file in files
        if file.endswith(".yaml") or file.endswith(".yml")
    ]

    return config_files_to_process


def _get_co_files_to_process(path):
    """Get a list of .co files to process.

    Args:
        path (str): The path to the file or directory to process.

    Returns:
        list: The list of .co files to process.
    """

    if os.path.isfile(path) and path.endswith(".co"):
        return [path]

    co_files_to_process = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".co"):
                base_file = file.split(".", 1)[0]
                v1_file = os.path.join(root, base_file + ".v1.co")
                if not os.path.exists(v1_file):
                    co_file = os.path.join(root, file)
                    co_files_to_process.append(co_file)

    return co_files_to_process


def _process_co_files(
    co_files_to_process: List[str],
    from_version: Literal["1.0", "2.0-alpha"],
    include_main_flow: bool,
    use_active_decorator: bool,
    validate: bool,
) -> int:
    """Processes the Colang files in the given path.

    It converts the Colang files from the given version to the latest version and writes the changes to the files.

    Args:
        co_files_to_process (List[str]): The list of Colang files to process.
        from_version (str): The version of the Colang files to convert from.
        include_main_flow (bool): Whether to include the main flow.
        use_active_decorator (bool): Whether to use the active decorator.
        validate (bool): Whether to validate the files.
    Returns:
        int: The total number of files changed.
    """

    total_files_changed = 0
    checked_directories = set()

    converter = {
        "1.0": convert_colang_1_syntax,
        "2.0-alpha": convert_colang_2alpha_syntax,
    }

    for file_path in co_files_to_process:
        console.print(f" - converting Colang file: {file_path}")

        lines = _read_file_lines(file_path)

        new_lines = converter[from_version](lines)

        if validate:
            _validate_file(file_path, new_lines)

        # from 1.0 to latest
        if new_lines and from_version == "1.0":
            if include_main_flow:
                # we should not include main flow for standard library
                main_file_path = _create_main_co_if_not_exists(file_path)
                _add_main_co_file(main_file_path)

            if use_active_decorator:
                # we should not use active decorator for standard library?
                new_lines = _add_active_decorator(new_lines)

            if _write_transformed_content_and_rename_original(file_path, new_lines):
                total_files_changed += 1

        # from 2.0-alpha to latest
        if new_lines and from_version == "2.0-alpha":
            directory = os.path.dirname(file_path)
            if directory not in checked_directories:
                main_file_path = _create_main_co_if_not_exists(file_path)
                _add_main_co_file(main_file_path)
                checked_directories.add(directory)
            _remove_files_from_path(directory, _FILES_TO_EXCLUDE_ALPHA)
            if file_path not in _FILES_TO_EXCLUDE_ALPHA and _write_to_file(file_path, new_lines):
                total_files_changed += 1

    return total_files_changed


def _validate_file(file_path, new_lines):
    """Validates the file after the conversion.

    It validates the file by parsing it and checking for any errors.

    Args:
        file_path (str): The path to the file to validate.
        new_lines (list): The new lines of the file after the conversion.

    Raises:
        Exception: If the validation fails.
    """

    try:
        parse_colang_file(filename=file_path, content="\n".join(new_lines), version="2.x")
    except Exception as e:
        raise Exception(f"Validation failed for file: {file_path}. Error: {str(e)}")


def _process_config_files(config_files_to_process: List[str]) -> int:
    """Processes the config files in the given path.

    It extracts the rails flows from the config files and writes them to a new file.

    Args:
        config_files_to_process (List[str]): The list of config files to process.

    Returns:
        int: The total number of config files changed.
    """
    total_config_files_changed = 0

    for file_path in config_files_to_process:
        console.print(f" - converting config file: {file_path}")
        raw_config = _get_raw_config(file_path)

        rails_flows = _generate_rails_flows(_get_rails_flows(raw_config))

        if rails_flows:
            _rails_co_file_path = Path(file_path).parent / "rails.co"

            if _write_rails_flows_to_file(_rails_co_file_path, rails_flows):
                total_config_files_changed += 1

            _comment_rails_flows_in_config(file_path)

        # set colang version to 2.x
        _set_colang_version(version="2.x", file_path=file_path)

        # Process the sample_conversation section
        _process_sample_conversation_in_config(file_path)

    return total_config_files_changed


def _read_file_lines(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
    except Exception as e:
        logging.error(f"Failed to read file: {file_path}. Error: {str(e)}")
        return []
    return lines


def _confirm_and_tag_replace(line, original_line, name):
    if original_line != line:
        _LIBS_USED.add(name)


def _set_colang_version(file_path: str, version: str):
    """Sets the 'colang_version' in the given raw_config.

    Args:
        file_path (str): The path to the config file.
        version (str): The version to set.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Check if colang_version already exists
    for i, line in enumerate(lines):
        if "colang_version" in line:
            # If it exists, replace it
            lines[i] = f"colang_version: {version}\n"
            break
    else:
        # If it doesn't exist, insert it at the first line
        lines.insert(0, f"colang_version: {version}\n")

    # Write the modified lines back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)


def _create_main_co_if_not_exists(file_path):
    """Check if the main co file exists in the directory of the file.

    If the main co file does not exist, it creates an empty main co file in the directory of the given file.

    Args:
        file_path (str): The path to the file to check for the main co file.
    """
    directory = os.path.dirname(file_path)
    main_file_path = os.path.join(directory, "main.co")

    if not os.path.exists(main_file_path):
        # create an empty file
        open(main_file_path, "w").close()
    return main_file_path


def _remove_files_from_path(path, filenames: List[str]):
    """Remove files from the path.

    Args:
        path (str): The path to the directory to remove files from.
        filenames (list[str]): The list of filenames to remove.
    """
    for filename in filenames:
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)


def convert_sample_conversation_syntax(lines: List[str]) -> List[str]:
    """Converts the sample_conversation section from the old format to the new format.

    Args:
        lines (List[str]): The lines of the sample_conversation to convert.

    Returns:
        List[str]: The new lines of the sample_conversation after conversion.
    """
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        # skip empty lines
        if line.strip() == "":
            new_lines.append(line + "\n")
            i += 1
            continue

        # proccess 'user' lines
        if line.startswith("user "):
            # Check if line matches 'user "message"'
            m = re.match(r'user\s+"(.*)"', line)
            if m:
                message = m.group(1)
                new_lines.append(f'user action: user said "{message}"\n')
                # We know that the  next line is intent
                if i + 1 < len(lines):
                    intent_line = lines[i + 1].strip()
                    if intent_line:
                        # Include 'user' prefix in the intent
                        new_lines.append(f"user intent: user {intent_line}\n")
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif line.startswith("bot "):
            # Check wether line is 'bot intent'
            m = re.match(r"bot\s+(.*)", line)
            if m:
                intent = m.group(1)
                # include 'bot' prefix in the intent
                new_lines.append(f"bot intent: bot {intent}\n")
                # next line is message
                if i + 1 < len(lines):
                    message_line = lines[i + 1].strip()
                    m2 = re.match(r'"(.*)"', message_line)
                    if m2:
                        message = m2.group(1)
                        new_lines.append(f'bot action: bot say "{message}"\n')
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            # other lines remain as is
            new_lines.append(line + "\n")
            i += 1
    return new_lines


def _process_sample_conversation_in_config(file_path: str):
    """Processes the sample_conversation section in the config file.

    Args:
        file_path (str): The path to the config file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find 'sample_conversation:' line
    sample_conv_line_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s*sample_conversation:\s*\|", line):
            sample_conv_line_idx = idx
            break
    if sample_conv_line_idx is None:
        return  # No sample_conversation in file

    # get the base indentation
    base_indent = len(lines[sample_conv_line_idx]) - len(lines[sample_conv_line_idx].lstrip())
    sample_conv_indent = None

    # get sample_conversation lines
    sample_lines = []
    i = sample_conv_line_idx + 1
    while i < len(lines):
        line = lines[i]
        # Check if the line is indented more than base_indent
        line_indent = len(line) - len(line.lstrip())
        if line.strip() == "":
            sample_lines.append(line)
            i += 1
            continue
        if line_indent > base_indent:
            if sample_conv_indent is None:
                sample_conv_indent = line_indent
            sample_lines.append(line)
            i += 1
        else:
            # end of sample conversations lines
            break
    sample_conv_end_idx = i

    stripped_sample_lines = [line[sample_conv_indent:] for line in sample_lines]
    new_sample_lines = convert_sample_conversation_syntax(stripped_sample_lines)
    # revert the indentation
    indented_new_sample_lines = [" " * (sample_conv_indent or 0) + line for line in new_sample_lines]
    lines[sample_conv_line_idx + 1 : sample_conv_end_idx] = indented_new_sample_lines
    # Write back the modified lines
    with open(file_path, "w") as f:
        f.writelines(lines)
