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
from typing import Dict, Optional, Tuple, Union

yara = None
try:
    import yara
except ImportError:
    pass

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.library.injection_detection.yara_config import ActionOptions, Rules

YARA_DIR = Path(__file__).resolve().parent.joinpath("yara_rules")

log = logging.getLogger(__name__)


def _check_yara_available():
    if yara is None:
        raise ImportError(
            "The yara module is required for injection detection. "
            "Please install it using: pip install yara-python"
        )


def _validate_injection_config(config: RailsConfig) -> None:
    """
    Validates the injection detection configuration.

    Args:
        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Raises:
        ValueError: If the configuration is missing or invalid.
        FileNotFoundError: If the provided `yara_path` is not a directory.
    """
    command_injection_config = config.rails.config.injection_detection

    if command_injection_config is None:
        msg = (
            "Injection detection configuration is missing in the provided RailsConfig."
        )
        log.error(msg)
        raise ValueError(msg)

    # Validate action option
    action_option = command_injection_config.action
    if action_option not in ActionOptions:
        msg = (
            "Expected 'reject', 'omit', or 'sanitize' action in injection config but got %s"
            % action_option
        )
        log.error(msg)
        raise ValueError(msg)

    # Validate yara_path if no yara_rules provided
    if not command_injection_config.yara_rules:
        yara_path = command_injection_config.yara_path
        if yara_path and isinstance(yara_path, str):
            yara_path = Path(yara_path)
            if not yara_path.exists() or not yara_path.is_dir():
                msg = (
                    "Provided `yara_path` value in injection config %s is not a directory."
                    % yara_path
                )
                log.error(msg)
                raise FileNotFoundError(msg)
        elif yara_path and not isinstance(yara_path, str):
            msg = "Expected a string value for `yara_path` but got %r instead." % type(
                yara_path
            )
            log.error(msg)
            raise ValueError(msg)


def _extract_injection_config(
    config: RailsConfig,
) -> Tuple[str, Path, Tuple[str], Optional[Dict[str, str]]]:
    """
    Extracts and processes the injection detection configuration values.

    Args:
        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Returns:
        Tuple[str, Path, Tuple[str], Optional[Dict[str, str]]]: A tuple containing the action option,
        the YARA path, the injection rules, and optional yara_rules dictionary.

    Raises:
        ValueError: If the injection rules contain invalid elements.
    """
    command_injection_config = config.rails.config.injection_detection
    yara_rules = command_injection_config.yara_rules

    # Set yara_path
    if yara_rules:
        # we'll use this for validation only
        yara_path = YARA_DIR
    else:
        yara_path = command_injection_config.yara_path or YARA_DIR
        if isinstance(yara_path, str):
            yara_path = Path(yara_path)

    injection_rules = tuple(command_injection_config.injections)

    # only validate rule names against available rules if using yara_path
    if not yara_rules and not set(injection_rules) <= Rules:
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

    return command_injection_config.action, yara_path, injection_rules, yara_rules


def _load_rules(
    yara_path: Path, rule_names: Tuple, yara_rules: Optional[Dict[str, str]] = None
) -> Union["yara.Rules", None]:
    """
    Loads and compiles YARA rules from either file paths or direct rule strings.

    Args:
        yara_path (Path): The path to the directory containing YARA rule files.
        rule_names (Tuple): A tuple of YARA rule names to load.
        yara_rules (Optional[Dict[str, str]]): Dictionary mapping rule names to YARA rule strings.

    Returns:
        Union['yara.Rules', None]: The compiled YARA rules object if successful,
        or None if no rule names are provided.

    Raises:
        yara.SyntaxError: If there is a syntax error in the YARA rules.
        ImportError: If the yara module is not installed.
    """

    if len(rule_names) == 0:
        log.warning(
            "Injection config was provided but no modules were specified. Returning None."
        )
        return None

    try:
        if yara_rules:
            rules_source = {
                name: rule for name, rule in yara_rules.items() if name in rule_names
            }
            rules = yara.compile(
                sources={rule_name: rules_source[rule_name] for rule_name in rule_names}
            )
        else:
            rules_to_load = {
                rule_name: str(yara_path.joinpath(f"{rule_name}.yara"))
                for rule_name in rule_names
            }
            rules = yara.compile(filepaths=rules_to_load)
    except yara.SyntaxError as e:
        msg = f"Encountered SyntaxError: {e}"
        log.error(msg)
        raise e
    return rules


def _omit_injection(text: str, matches: list["yara.Match"]) -> str:
    """
    Attempts to strip the offending injection attempts from the provided text.

    Note:
        This method may not be completely effective and could still result in
        malicious activity.

    Args:
        text (str): The text to check for command injection.
        matches (list['yara.Match']): A list of YARA rule matches.

    Returns:
        str: The text with the detected injections stripped out.

    Raises:
        ImportError: If the yara module is not installed.
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


def _sanitize_injection(text: str, matches: list["yara.Match"]) -> str:
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
        matches (list['yara.Match']): A list of YARA rule matches.

    Returns:
        str: The text with the detected injections sanitized.

    Raises:
        NotImplementedError: If the sanitization logic is not implemented.
        ImportError: If the yara module is not installed.
    """

    raise NotImplementedError(
        "Injection sanitization is not yet implemented. Please use 'reject' or 'omit'"
    )


def _reject_injection(text: str, rules: "yara.Rules") -> Tuple[bool, str]:
    """
    Detects whether the provided text contains potential injection attempts.

    This function is recommended as an output or execution guardrail. It loads
    all relevant YARA rules and compiles them according to the provided configuration.

    Args:
        text (str): The text to check for command injection.
        rules ('yara.Rules'): The loaded YARA rules.

    Returns:
        bool: True if attempted exploitation is detected, False otherwise.
        str: list of matches as a string

    Raises:
        ValueError: If the `action` parameter in the configuration is invalid.
        ImportError: If the yara module is not installed.
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
    _check_yara_available()

    _validate_injection_config(config)
    action_option, yara_path, rule_names, yara_rules = _extract_injection_config(config)

    rules = _load_rules(yara_path, rule_names, yara_rules)

    if action_option == "reject":
        verdict, detections = _reject_injection(text, rules)
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
            return _omit_injection(text, matches)
        elif action_option == "sanitize":
            return _sanitize_injection(text, matches)
        else:
            # We should never ever hit this since we inspect the action option above, but putting an error here anyway.
            raise NotImplementedError(
                f"Expected `action` parameter to be 'omit' or 'sanitize' but got {action_option} instead."
            )
    else:
        return text
