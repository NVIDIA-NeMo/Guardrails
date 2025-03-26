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

import re
import logging
import yara
from pathlib import Path
from typing import Union, Tuple
from functools import lru_cache

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

YARA_DIR = Path(__file__).parent.joinpath("yara_rules")
PROVIDED_MODULES = ["sqli", "template", "code", "xss"]

log = logging.getLogger(__name__)


def _validate_unpack_config(config: RailsConfig) -> Tuple[str, Path, list[str]]:
    command_injection_config = config.rails.config.injection
    yara_path = command_injection_config.yara_path
    if not yara_path:
        yara_path = YARA_DIR
    elif isinstance(yara_path, str):
        yara_path = Path(yara_path)
        if not yara_path.exists() and yara_path.is_dir():
            msg = f"Provided `yara_path` value in injection config {yara_path} is not a directory."
            log.error(msg)
            raise FileNotFoundError(msg)
    else:
        msg = f"Expected a string value for `yara_path` but got {type(yara_path)} instead."
        log.error(msg)
        raise ValueError(msg)
    action_option = command_injection_config.action
    if action_option not in ["reject", "omit", "sanitize"]:
        msg = f"Expected 'reject', 'omit', or 'sanitize' action in injection config but got {action_option}"
        log.error(msg)
        raise ValueError(msg)
    injection_rules = command_injection_config.injections
    if not set(injection_rules) <= set(PROVIDED_MODULES):
        # Do the easy check above first. If they provide a custom dir or a custom rules file, check the filesystem
        if not all(
            [
                yara_path.joinpath(f"{module_name}.yara").is_file()
                for module_name in injection_rules
            ]
        ):
            msg = (
                f"Provided set of `injections` in injection config {injection_rules} contains elements not in "
                f"available rules. Provided rules are in {PROVIDED_MODULES}."
            )
            log.error(msg)
            raise ValueError(msg)

    return action_option, yara_path, injection_rules


@lru_cache()
def load_rules(config: RailsConfig) -> Tuple[str, Union[yara.Rules, None]]:
    """
    Take a RailsConfig object and load compiled yara rules

    Parameters
    ----------
    config : rails configuration object

    Returns
    -------
    the action option as a string
    compiled YARA rules object
    """
    action_option, yara_path, rule_names = _validate_unpack_config(config)
    if len(rule_names) == 0:
        log.warning(
            "Injection config was provided but no modules were specified. Returning None."
        )
        return action_option, None
    rules_to_load = {
        rule_name: yara_path.joinpath(f"{rule_name}.yara") for rule_name in rule_names
    }
    try:
        rules = yara.compile(filepaths=rules_to_load)
    except yara.SyntaxError as e:
        msg = f"Encountered SyntaxError: {e}"
        log.error(msg)
        raise e
    return action_option, rules


def detect_injection_mapping(result: bool) -> bool:
    """
    Mapping for detect_*_injection actions.

    The detect_*_injection functions return True when the relevant type of data is detected, and we take the prescribed
    action if the result is true.
    """
    return result


def omit_injection(text: str, matches: list[yara.Match]) -> str:
    """
    Attempt to strip the offending injection attempt.
    Note that this may not be completely effective and may still result in malicious activity.

    Parameters
    ----------
    text : text to check for command injection
    matches : YARA rule matches

    Returns
    -------
    the text to check for command injection with the detected injections stripped out.
    """
    # Copy the text to a placeholder variable
    modified_text = text
    for match in matches:
        if match.strings:
            for match_string in match.strings:
                for instance in match_string:
                    if instance in modified_text:
                        modified_text = modified_text.replace(
                            instance.plaintext().decode("utf-8"), ""
                        )
    return modified_text


def sanitize_injection(text: str, matches: list[yara.Match]) -> str:
    """
    Attempt to sanitize the offending injection attempt.
    Note that this may not be completely effective and may still result in malicious activity.
    Attempting to sanitize the malicious input but continuing to pass it instead of rejecting or omitting
    is inherently risky and generally not recommended.

    Parameters
    ----------
    text : text to check for command injection
    matches : YARA rule matches

    Returns
    -------
    the text to check for command injection with the detected injections stripped out.
    """
    raise NotImplementedError(
        "Injection sanitization is not yet implemented. Please use 'reject' or 'omit'"
    )


@action(is_system_action=True, output_mapping=detect_injection_mapping)
def reject_injection(text: str, config: RailsConfig) -> bool:
    """
    Detect whether the text contains potential injection.
    Recommended as an output or execution rail.
    Note that this will load all relevant YARA rules and compile them according to the provided config.

    Parameters
    ----------
    text : text to check for command injection
    config : rails configuration object

    Returns
    -------
    True if command injection is detected, False otherwise.
    """
    action_option, rules = load_rules(config)
    if action_option != "reject":
        log.warning(
            f"reject_injection guardrail expects config `action` parameter to be 'reject', but got '{action_option}' "
            f"instead. Proceeding with rejection rail. Please modify your config if you want a different action."
        )
    if rules is None:
        log.warning(
            "reject_injection guardrail was invoked but no rules were specified in the InjectionDetection config."
        )
        return False
    matches = rules.match(data=text)
    if matches:
        matches_string = ", ".join([match_name.rule for match_name in matches])
        log.info(f"Input matched on rule {matches_string}.")
        return True
    else:
        return False


@action(is_system_action=True)
def mitigate_injection(text: str, config: RailsConfig) -> str:
    """
    Detect whether the text contains potential injection.

    Parameters
    ----------
    text : text to check for command injection
    config : rails configuration object

    Returns
    -------
    String object with the detected injection attempt omitted or relatively sanitized.
    """
    action_option, rules = load_rules(config)
    if action_option == "reject":
        log.warning(
            "mitigate_injection expects config `action` parameter to be 'omit' or 'sanitize' but got 'reject' "
            "instead. Using reject_injection rail instead of mitigate_injection rail. "
            "Please modify your config if you want a different action"
        )
        if reject_injection(text, config):
            return "I'm sorry, I can't help you with that."
        else:
            return text
    if action_option not in ["omit", "sanitize"]:
        msg = f"Expected `action` parameter to be 'omit' or 'sanitize' but got {action_option} instead."
        log.error(msg)
        raise ValueError(msg)
    if rules is None:
        log.warning(
            "mitigate_injection guardrail was invoked but no rules were specified in the InjectionDetection config."
        )
        return text
    matches = rules.match(data=text)
    if matches:
        matches_string = ", ".join([match_name.rule for match_name in matches])
        log.info(f"Input matched on rule {matches_string}.")
        match action_option:
            case "omit":
                return omit_injection(text, matches)
            case "sanitize":
                return sanitize_injection(text, matches)
            # We should never ever hit this since we inspect the action option above, but putting an error here anyway.
            case _:
                raise NotImplementedError(
                    f"Expected `action` parameter to be 'omit' or 'sanitize' but got {action_option} instead."
                )
    else:
        return text
