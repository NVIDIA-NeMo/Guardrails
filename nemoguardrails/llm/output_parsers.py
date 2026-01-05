# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Sequence, Union


def _replace_prefix(s: str, prefix: str, repl: str):
    """Helper function to replace a prefix from a string."""
    if s.startswith(prefix):
        return repl + s[len(prefix) :].strip()

    return s


def user_intent_parser(s: str):
    """Parses the user intent."""
    return _replace_prefix(s.strip(), "User intent: ", "  ")


def bot_intent_parser(s: str):
    """Parses the bot intent."""
    return _replace_prefix(s.strip(), "Bot intent: ", "bot ")


def bot_message_parser(s: str):
    """Parses the bot messages."""
    return _replace_prefix(s.strip(), "Bot message: ", "  ")


def verbose_v1_parser(s: str):
    """Parses completions generated using the `verbose_v1` formatter.

    This will convert text from the following format:
      User message: "Hello"
      User intent: express greeting
      Bot intent: express greeting
      Bot message: "Hi"

    To:
      user "Hello"
        express greeting
      bot express greeting
        "Hi"
    """
    lines = s.split("\n")

    prefixes = [
        ("User message: ", "user "),
        ("Bot message: ", "  "),
        ("User intent: ", "  "),
        ("Bot intent: ", "bot "),
    ]

    for i in range(len(lines)):
        # Some LLMs generate a space at the beginning of the first line
        lines[i] = lines[i].strip()
        for prefix, repl in prefixes:
            # Also allow prefixes to be in lower-case
            lines[i] = _replace_prefix(lines[i], prefix, repl)
            lines[i] = _replace_prefix(lines[i], prefix.lower(), repl)

    return "\n".join(lines)


def _parse_unsafe_violations(response_text):
    """Helper function to parse violations from unsafe response."""
    # find "unsafe" case-insensitively but preserve original case for violations
    lower_response = response_text.lower()
    unsafe_pos = lower_response.find("unsafe")
    if unsafe_pos != -1:
        # get the part after "unsafe" from the original case-preserved text
        after_unsafe = response_text[unsafe_pos + len("unsafe") :].strip()
        if after_unsafe:
            violations = [v.strip() for v in after_unsafe.split() if v.strip()]
            return violations
    return []


def is_content_safe(response: str) -> Sequence[Union[bool, str]]:
    """Analyzes a given response from a guardrails check (e.g., content safety check or input check) and determines if the content is safe or not.

    The function operates based on the presence of certain keywords in the response:
    - If the response contains "safe", the content is deemed safe.
    - If the response contains "unsafe" or "yes", the content is deemed unsafe.
    - If the response contains "no", the content is deemed safe.

    Note: If you are using this function for a different task with a custom prompt, ensure to update the logic to match the new context.
    Here, "yes" implies the content should be blocked, is unsafe, or violates a policy, while "no" implies the content is safe or does not violate any policy.


    For example, in the `self_check_facts` action, we have a different interpretation.
    Here, "yes" indicates the content is safe. This is against the default logic of the `is_content_safe` function.
    Therefore, we invert the result of `is_content_safe` to align with this interpretation.
    If `is_content_safe` returns `False` (content is safe), `result` will be `0.0`. If `is_content_safe` returns `True` (content is unsafe), `result` will be `1.0`.



    Args:
        response (str): The response string to analyze.

    Returns:
        Sequence[Union[bool, str]]: A sequence where the first element is a boolean indicating the safety of the content (True if safe, False otherwise),
        and the remaining elements are strings representing violated policies, if any.
    """

    original_response = response.strip()  # Keep original case for violations
    response_lower = response.lower().strip()
    # replace sequences of non word characters in the response with a single space
    response_lower = re.sub(r"\W+", " ", response_lower)
    original_response = re.sub(r"\W+", " ", original_response)
    # we only look at the first 3 words in the response
    splited_response = response_lower.split(" ")[:2]

    response_actions = {
        "safe": lambda: [True],
        "unsafe": lambda: [False] + _parse_unsafe_violations(original_response),
        "yes": lambda: [False],
        "no": lambda: [True],
    }

    for prefix, action in response_actions.items():
        if prefix in splited_response:
            return action()

    return [False]


def nemoguard_parse_prompt_safety(response: str) -> Sequence[Union[bool, str]]:
    """Analyzes a given model response from a Guardrails check (e.g., content safety check or input check) and determines if the content is safe or not.

    The function operates based on the following expected structured JSON output from the NemoGuard ContentSafety model.

    {
        "User Safety": "unsafe",
        "Violated Categories": "category1, category2"
    }

    Args:
        response (str): The response string to analyze.

    Returns:
        Sequence[Union[bool, str]]: A sequence where the first element is a boolean indicating the safety of the content (True if safe, False otherwise),
        and the remaining elements are strings representing violated policies, if any.
    """
    try:
        # try parsing it as json
        parsed_json_result = json.loads(response)
        assert "User Safety" in parsed_json_result
        result = parsed_json_result["User Safety"].lower()
        if "Safety Categories" in parsed_json_result:
            safety_categories = [cat.strip() for cat in parsed_json_result["Safety Categories"].split(",")]
        else:
            safety_categories = []
    except Exception:
        # If there is an error, and we can't parse the response, we return unsafe assuming this is a potential jailbreaking attempt
        result = "unsafe"
        safety_categories = ["JSON parsing failed"]

    is_safe = result == "safe"
    if is_safe:
        return [True]
    else:
        return [False] + safety_categories


def nemoguard_parse_response_safety(response: str) -> Sequence[Union[bool, str]]:
    """Analyzes a given model response from a Guardrails check (e.g., content safety check or output check) and determines if the content is safe or not.

    The function operates based on the following expected structured JSON output from the NemoGuard ContentSafety model.

    {
        "User Safety": "unsafe",
        "Response Safety": "unsafe",
        "Violated Categories": "category1, category2"
    }

    Args:
        response (str): The response string to analyze.

    Returns:
        Sequence[Union[bool, str]]: A sequence where the first element is a boolean indicating the safety of the content (True if safe, False otherwise),
        and the remaining elements are strings representing violated policies, if any.
    """
    try:
        # try parsing it as json
        parsed_json_result = json.loads(response)
        assert "Response Safety" in parsed_json_result
        result = parsed_json_result["Response Safety"].lower()
        if "Safety Categories" in parsed_json_result:
            safety_categories = [cat.strip() for cat in parsed_json_result["Safety Categories"].split(",")]
        else:
            safety_categories = []
    except Exception:
        # If there is an error, and we can't parse the response, we return unsafe assuming this is a potential jailbreaking attempt
        result = "unsafe"
        safety_categories = ["JSON parsing failed"]

    is_safe = result == "safe"
    if is_safe:
        return [True]
    else:
        return [False] + safety_categories


def _strip_think_tags(response: str) -> str:
    """Helper function to strip <think>...</think> tags from model response."""
    # Remove <think>...</think> blocks (including multi-line)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()


def _extract_harm_value(response: str, field_name: str) -> str:
    """Helper function to extract harmful/unharmful value for a given field.

    Args:
        response: The model response text (with think tags already stripped).
        field_name: The field to look for (e.g., "Prompt harm" or "Response Harm").

    Returns:
        The extracted value ("harmful" or "unharmful"), or "harmful" if parsing fails.
    """
    # Look for the field pattern case-insensitively
    pattern = rf"{re.escape(field_name)}\s*:\s*(\w+)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "harmful"  # Default to harmful if parsing fails


def nemotron_reasoning_parse_prompt_safety(response: str) -> Sequence[Union[bool, str]]:
    """Analyzes a response from Nemotron-Content-Safety-Reasoning-4B for prompt safety.

    The function parses the following expected output format:

    Reasoning Off mode:
        Prompt harm: harmful/unharmful
        Response Harm: harmful/unharmful

    Reasoning On mode:
        <think>
        [Model's reasoning trace]
        </think>

        Prompt harm: harmful/unharmful
        Response Harm: harmful/unharmful

    Args:
        response (str): The response string to analyze.

    Returns:
        Sequence[Union[bool, str]]: A sequence where the first element is a boolean
        indicating the safety of the content (True if unharmful/safe, False if harmful).
    """
    # Strip <think> tags if present
    cleaned_response = _strip_think_tags(response)

    # Extract the prompt harm value
    harm_value = _extract_harm_value(cleaned_response, "Prompt harm")

    # "unharmful" means safe (True), "harmful" means unsafe (False)
    is_safe = harm_value == "unharmful"

    if is_safe:
        return [True]
    else:
        return [False]


def nemotron_reasoning_parse_response_safety(
    response: str,
) -> Sequence[Union[bool, str]]:
    """Analyzes a response from Nemotron-Content-Safety-Reasoning-4B for response safety.

    The function parses the following expected output format:

    Reasoning Off mode:
        Prompt harm: harmful/unharmful
        Response Harm: harmful/unharmful

    Reasoning On mode:
        <think>
        [Model's reasoning trace]
        </think>

        Prompt harm: harmful/unharmful
        Response Harm: harmful/unharmful

    Args:
        response (str): The response string to analyze.

    Returns:
        Sequence[Union[bool, str]]: A sequence where the first element is a boolean
        indicating the safety of the content (True if unharmful/safe, False if harmful).
    """
    # Strip <think> tags if present
    cleaned_response = _strip_think_tags(response)

    # Extract the response harm value
    harm_value = _extract_harm_value(cleaned_response, "Response Harm")

    # "unharmful" means safe (True), "harmful" means unsafe (False)
    is_safe = harm_value == "unharmful"

    if is_safe:
        return [True]
    else:
        return [False]
