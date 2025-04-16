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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import yara

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.injection_detection.yara_config import ActionOptions, Rules

YARA_DIR = Path(__file__).resolve().parent.joinpath("yara_rules")

log = logging.getLogger(__name__)


def _validate_unpack_config(config: RailsConfig) -> Tuple[str, Path, Tuple[str]]:
    """
    Validates and unpacks the injection detection configuration.

    Args:
        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Returns:
        Tuple[str, Path, Tuple[str]]: A tuple containing the action option, the YARA path,
        and the injection rules.

    Raises:
        FileNotFoundError: If the provided `yara_path` is not a directory.
        ValueError: If `yara_path` is not a string, the action option is invalid,
        or the injection rules contain invalid elements.
    """
    command_injection_config = config.rails.config.injection_detection

    if command_injection_config is None:
        msg = (
            "Injection detection configuration is missing in the provided RailsConfig."
        )
        log.error(msg)
        raise ValueError(msg)
    yara_path = command_injection_config.yara_path
    if not yara_path:
        yara_path = YARA_DIR
    elif isinstance(yara_path, str):
        yara_path = Path(yara_path)
        if not yara_path.exists() or not yara_path.is_dir():
            msg = (
                "Provided `yara_path` value in injection config %s is not a directory."
                % yara_path
            )
            log.error(msg)
            raise FileNotFoundError(msg)
    else:
        msg = "Expected a string value for `yara_path` but got %r instead." % type(
            yara_path
        )

        log.error(msg)
        raise ValueError(msg)
    action_option = command_injection_config.action
    if action_option not in ActionOptions:
        msg = (
            "Expected 'reject', 'omit', or 'sanitize' action in injection config but got %s"
            % action_option
        )
        log.error(msg)
        raise ValueError(msg)
    injection_rules = tuple(command_injection_config.injections)
    if not set(injection_rules) <= Rules:
        # Do the easy check above first. If they provide a custom dir or a custom rules file, check the filesystem
        if not all(
            [
                yara_path.joinpath(f"{module_name}.yara").is_file()
                for module_name in injection_rules
            ]
        ):
            default_rule_names = ", ".join([member.value for member in Rules])
            msg = (
                "Provided set of `injections` in injection config %r contains elements not in available rules. "
                "Provided rules are in %r."
            ) % (injection_rules, default_rule_names)
            log.error(msg)
            raise ValueError(msg)

    return action_option, yara_path, injection_rules


@lru_cache()
def load_rules(yara_path: Path, rule_names: Tuple) -> Union[yara.Rules, None]:
    """
    Loads and compiles YARA rules from the specified path and rule names.

    Args:
        yara_path (Path): The path to the directory containing YARA rule files.
        rule_names (Tuple): A tuple of YARA rule names to load.

    Returns:
        Union[yara.Rules, None]: The compiled YARA rules object if successful,
        or None if no rule names are provided.

    Raises:
        yara.SyntaxError: If there is a syntax error in the YARA rules.
    """
    if len(rule_names) == 0:
        log.warning(
            "Injection config was provided but no modules were specified. Returning None."
        )
        return None
    rules_to_load = {
        rule_name: str(yara_path.joinpath(f"{rule_name}.yara"))
        for rule_name in rule_names
    }
    try:
        rules = yara.compile(filepaths=rules_to_load)
    except yara.SyntaxError as e:
        msg = f"Encountered SyntaxError: {e}"
        log.error(msg)
        raise e
    return rules


def omit_injection(text: str, matches: list[yara.Match]) -> str:
    """
    Attempts to strip the offending injection attempts from the provided text.

    Note:
        This method may not be completely effective and could still result in
        malicious activity.

    Args:
        text (str): The text to check for command injection.
        matches (list[yara.Match]): A list of YARA rule matches.

    Returns:
        str: The text with the detected injections stripped out.
    """
    # Copy the text to a placeholder variable
    modified_text = text
    for match in matches:
        if match.strings:
            for match_string in match.strings:
                for instance in match_string.instances:
                    try:
                        plaintext = instance.plaintext().decode("utf-8")
                        if plaintext in modified_text:
                            modified_text = modified_text.replace(plaintext, "")
                    except (AttributeError, UnicodeDecodeError) as e:
                        log.warning(f"Error processing match: {e}")
    return modified_text


def sanitize_injection(text: str, matches: list[yara.Match]) -> str:
    """
    Attempts to sanitize the offending injection attempts in the provided text.
    This is done by 'de-fanging' the offending content, transforming it into a state that will not execute
    downstream commands.

    Note:
        This method may not be completely effective and could still result in
        malicious activity. Sanitizing malicious input instead of rejecting or
        omitting it is inherently risky and generally not recommended.

    Args:
        text (str): The text to check for command injection.
        matches (list[yara.Match]): A list of YARA rule matches.

    Returns:
        str: The text with the detected injections sanitized.

    Raises:
        NotImplementedError: If the sanitization logic is not implemented.
    """
    raise NotImplementedError(
        "Injection sanitization is not yet implemented. Please use 'reject' or 'omit'"
    )


def reject_injection(text: str, rules: yara.Rules) -> Tuple[bool, str]:
    """
    Detects whether the provided text contains potential injection attempts.

    This function is recommended as an output or execution guardrail. It loads
    all relevant YARA rules and compiles them according to the provided configuration.

    Args:
        text (str): The text to check for command injection.
        rules (yara.Rules): The loaded YARA rules.

    Returns:
        bool: True if attempted exploitation is detected, False otherwise.
        str: list of matches as a string

    Raises:
        ValueError: If the `action` parameter in the configuration is invalid.
    """
    if rules is None:
        log.warning(
            "reject_injection guardrail was invoked but no rules were specified in the InjectionDetection config."
        )
        return False, ""
    matches = rules.match(data=text)
    if matches:
        matches_string = ", ".join([match_name.rule for match_name in matches])
        log.info(f"Input matched on rule {matches_string}.")
        return True, matches_string
    else:
        return False, ""


@action()
async def injection_detection(text: str, config: RailsConfig) -> str:
    """
    Detects and mitigates potential injection attempts in the provided text.

    Depending on the configuration, this function can omit or sanitize the detected
    injection attempts. If the action is set to "reject", it delegates to the
    `reject_injection` function.

    Args:
        text (str): The text to check for command injection.
        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Returns:
        str: The sanitized or original text, depending on the action specified in the configuration.

    Raises:
        ValueError: If the `action` parameter in the configuration is invalid.
        NotImplementedError: If an unsupported action is encountered.
    """
    action_option, yara_path, rule_names = _validate_unpack_config(config)
    rules = load_rules(yara_path, rule_names)
    if action_option == "reject":
        verdict, detections = reject_injection(text, rules)
        if verdict:
            return f"I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of {detections}."
        else:
            return text
    if rules is None:
        log.warning(
            "injection detection guardrail was invoked but no rules were specified in the InjectionDetection config."
        )
        return text
    matches = rules.match(data=text)
    if matches:
        matches_string = ", ".join([match_name.rule for match_name in matches])
        log.info(f"Input matched on rule {matches_string}.")
        if action_option == "omit":
            return omit_injection(text, matches)
        elif action_option == "sanitize":
            return sanitize_injection(text, matches)
        else:
            # We should never ever hit this since we inspect the action option above, but putting an error here anyway.
            raise NotImplementedError(
                f"Expected `action` parameter to be 'omit' or 'sanitize' but got {action_option} instead."
            )
    else:
        return text
