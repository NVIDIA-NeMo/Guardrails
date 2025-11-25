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
import asyncio
import dataclasses
import fnmatch
import hashlib
import importlib.resources as pkg_resources
import json
import os
import random
import re
import uuid
from ast import literal_eval
from collections import namedtuple
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import yaml
from rich.console import Console

# Global console object to be used throughout the code base.
console = Console()

secure_random = random.SystemRandom()

_FIRST_CAP_PATTERN = re.compile("(.)([A-Z][a-z0-9]+)")
_ALL_CAP_PATTERN = re.compile("([a-z0-9])([A-Z])")


def init_random_seed(seed: int) -> None:
    """Init random generator with seed."""
    global secure_random
    random.seed(seed)
    secure_random = random


def new_uuid() -> str:
    """Helper to generate new UUID v4.

    In testing mode, it will generate a predictable set of UUIDs to help debugging if random seed was set dependent on
    the environment variable DEBUG_MODE.
    """
    random_bits = secure_random.getrandbits(128)
    return str(uuid.UUID(int=random_bits, version=4))


def new_readable_uuid(name: str) -> str:
    """Creates a new uuid with a human readable prefix."""
    return f"({name}){new_uuid()}"


def new_var_uuid() -> str:
    """Creates a new uuid that is compatible with variable names."""
    return new_uuid().replace("-", "_")


# Very basic event validation - will be replaced by validation based on pydantic models
Property = namedtuple("Property", ["name", "type"])
Validator = namedtuple("Validator", ["description", "function"])


def _has_property(e: Dict[str, Any], p: Property) -> bool:
    return p.name in e and isinstance(e[p.name], p.type)


_event_validators = [
    Validator("Events need to provide 'type'", lambda e: "type" in e),
    Validator("Events need to provide 'uid'", lambda e: _has_property(e, Property("uid", str))),
    Validator(
        "Events need to provide 'event_created_at' of type 'str'",
        lambda e: _has_property(e, Property("event_created_at", str)),
    ),
    Validator(
        "Events need to provide 'source_uid' of type 'str'",
        lambda e: _has_property(e, Property("source_uid", str)),
    ),
    Validator(
        "***Action events need to provide an 'action_uid' of type 'str'",
        lambda e: "Action" not in e["type"] or _has_property(e, Property("action_uid", str)),
    ),
    Validator(
        "***ActionFinished events require 'action_finished_at' field of type 'str'",
        lambda e: "ActionFinished" not in e["type"] or _has_property(e, Property("action_finished_at", str)),
    ),
    Validator(
        "***ActionFinished events require 'is_success' field of type 'bool'",
        lambda e: "ActionFinished" not in e["type"] or _has_property(e, Property("is_success", bool)),
    ),
    Validator(
        "Unsuccessful ***ActionFinished events need to provide 'failure_reason'.",
        lambda e: "ActionFinished" not in e["type"] or (e["is_success"] or "failure_reason" in e),
    ),
    Validator(
        "***StartUtteranceBotAction events need to provide 'script' of type 'str'",
        lambda e: e["type"] != "StartUtteranceBotAction" or _has_property(e, Property("script", str)),
    ),
    Validator(
        "***UtteranceBotActionScriptUpdated events need to provide 'interim_script' of type 'str'",
        lambda e: e["type"] != "UtteranceBotActionScriptUpdated " or _has_property(e, Property("interim_script", str)),
    ),
    Validator(
        "***UtteranceBotActionFinished events need to provide 'final_script' of type 'str'",
        lambda e: e["type"] != "UtteranceBotActionFinished" or _has_property(e, Property("final_script", str)),
    ),
    Validator(
        "***UtteranceUserActionTranscriptUpdated events need to provide 'interim_transcript' of type 'str'",
        lambda e: e["type"] != "UtteranceUserActionTranscriptUpdated"
        or _has_property(e, Property("interim_transcript", str)),
    ),
    Validator(
        "***UtteranceUserActionFinished events need to provide 'final_transcript' of type 'str'",
        lambda e: e["type"] != "UtteranceUserActionFinished" or _has_property(e, Property("final_transcript", str)),
    ),
]


_action_to_modality_info: Dict[str, Tuple[str, str]] = {
    "UtteranceBotAction": ("bot_speech", "replace"),
    "UtteranceUserAction": ("user_speech", "replace"),
    "TimerBotAction": ("time", "parallel"),
    "FacialGestureBotAction": ("bot_face", "replace"),
    "GestureBotAction": ("bot_gesture", "override"),
    "PostureBotAction": ("bot_posture", "override"),
    "VisualChoiceSceneAction": ("information", "override"),
    "VisualInformationSceneAction": ("information", "override"),
    "VisualFormSceneAction": ("information", "override"),
    "MotionEffectCameraAction": ("camera_motion_effect", "override"),
    "ShotCameraAction": ("camera_shot", "override"),
}


def _add_modality_info(event_dict: Dict[str, Any]) -> None:
    """Add modality related information to the action event"""
    for action_name, modality_info in _action_to_modality_info.items():
        modality_name, modality_policy = modality_info
        if action_name in event_dict["type"]:
            event_dict.setdefault("action_info_modality", modality_name)
            event_dict.setdefault("action_info_modality_policy", modality_policy)
            return


def _update_action_properties(event_dict: Dict[str, Any]) -> None:
    """Update action related even properties and ensure UMIM compliance (very basic)"""
    now = datetime.now(timezone.utc).isoformat()
    if "Started" in event_dict["type"]:
        event_dict.setdefault("action_started_at", now)
    elif "Start" in event_dict["type"]:
        event_dict.setdefault("action_uid", new_uuid())
    elif "Updated" in event_dict["type"]:
        event_dict.setdefault("action_updated_at", now)
    elif "Finished" in event_dict["type"]:
        event_dict.setdefault("action_finished_at", now)
        if "is_success" in event_dict and event_dict["is_success"] and "failure_reason" in event_dict:
            del event_dict["failure_reason"]


def ensure_valid_event(event: Dict[str, Any]) -> None:
    """Performs basic event validation and throws an AssertionError if any of the validators fail."""
    for validator in _event_validators:
        assert validator.function(event), validator.description


def is_valid_event(event: Dict[str, Any]) -> bool:
    """Performs a basic event validation and returns True if the event conforms."""
    for validator in _event_validators:
        if not validator.function(event):
            return False
    return True


def new_event_dict(event_type: str, **payload) -> Dict[str, Any]:
    """Helper to create a generic event structure."""

    event: Dict[str, Any] = {
        "type": event_type,
        "uid": new_uuid(),
        "event_created_at": datetime.now(timezone.utc).isoformat(),
        "source_uid": "NeMoGuardrails",
    }

    event = {**event, **payload}

    if "Action" in event_type:
        _add_modality_info(event)
        _update_action_properties(event)

    ensure_valid_event(event)
    return event


class CustomDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)

    def represent_data(self, data):
        if isinstance(data, Enum):
            return self.represent_data(data.value)
        return super().represent_data(data)


class EnhancedJsonEncoder(json.JSONEncoder):
    """Custom json encoder to handler dataclass and enum types"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.value
        try:
            return super().default(o)
        except Exception:
            return f"Type {type(o)} not serializable"


def get_or_create_event_loop():
    """Helper to return the current asyncio loop.

    If one does not exist, it will be created.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def get_data_path(package_name: str, file_path: str) -> str:
    """Helper to get the path to the data directory."""
    try:
        # Try to get the path from the package resources
        if hasattr(pkg_resources, "files"):
            path = pkg_resources.files(package_name).joinpath(file_path)
        else:
            # For Python 3.8 we need this approach
            with pkg_resources.path(package_name, "__init__.py") as path:
                path = path.parent.joinpath(file_path)

        if path.exists():
            return str(path)
    except FileNotFoundError:
        pass

    # If that fails, try to get the path from the local file system
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", file_path))
    if os.path.exists(path):
        return path

    raise FileNotFoundError(f"File not found: {path}")


def get_examples_data_path(file_path: str) -> str:
    """Helper to get the path to the examples data directory."""
    return get_data_path("nemoguardrails", f"examples/{file_path}")


def get_chat_ui_data_path(file_path: str) -> str:
    """Helper to get the path to the chat-ui data directory."""
    return get_data_path("nemoguardrails", f"chat-ui/{file_path}")


def camelcase_to_snakecase(name: str) -> str:
    """Converts a CamelCase string to snake_case.

    Args:
        name (str): The CamelCase string to convert.

    Returns:
        str: The converted snake_case string.
    """
    s1 = _FIRST_CAP_PATTERN.sub(r"\1_\2", name)
    return _ALL_CAP_PATTERN.sub(r"\1_\2", s1).lower()


def snake_to_camelcase(name: str) -> str:
    """Converts a snake_case string to CamelCase.

    Args:
        name (str): The snake_case string to convert.

    Returns:
        str: The converted CamelCase string.
    """
    return "".join(n.capitalize() for n in name.split("_"))


def get_railsignore_path(path: Optional[str] = None) -> Optional[Path]:
    """Get railsignore path.

    Args:
        path (Optional[str]): The starting path to search for the .railsignore file.

    Returns:
        Path: The .railsignore file path, if found.

    Raises:
        FileNotFoundError: If the .railsignore file is not found.
    """
    current_path = Path(path) if path else Path.cwd()

    while True:
        railsignore_file = current_path / ".railsignore"
        if railsignore_file.exists() and railsignore_file.is_file():
            return railsignore_file
        if current_path == current_path.parent:
            break
        current_path = current_path.parent

    return None


def get_railsignore_patterns(railsignore_path: Path) -> Set[str]:
    """Retrieve all specified patterns in railsignore.

    Returns:
        Set[str]: The set of filenames or glob patterns in railsignore
    """
    ignored_patterns = set()

    if railsignore_path is None:
        return ignored_patterns

    # File doesn't exist or is empty
    if not railsignore_path.exists() or not os.path.getsize(railsignore_path):
        return ignored_patterns

    try:
        with open(railsignore_path, "r") as f:
            railsignore_entries = f.readlines()

        # Remove comments and empty lines, and strip out any extra spaces/newlines
        railsignore_entries = [
            line.strip() for line in railsignore_entries if line.strip() and not line.startswith("#")
        ]

        ignored_patterns.update(railsignore_entries)
        return ignored_patterns

    except FileNotFoundError:
        print(f"No {railsignore_path} found in the current directory.")
        return ignored_patterns


def is_ignored_by_railsignore(filename: str, ignore_patterns: Set[str]) -> bool:
    """Verify if a filename should be ignored by a railsignore pattern"""

    ignore = False

    for pattern in ignore_patterns:
        if fnmatch.fnmatch(filename, pattern):
            ignore = True
            break

    return ignore


def safe_eval(input_value: str) -> str:
    """
    Safely evaluate a string to handle unescaped quotes or invalid syntax from the async generate_value action.

    Args:
        input_value (str): The input string to evaluate.

    Returns:
        str: The evaluated and properly formatted string.

    Raises:
        ValueError: If the input cannot be safely evaluated.
    """
    if input_value.startswith(("'", '"')) and input_value.endswith(("'", '"')):
        try:
            return literal_eval(input_value)
        except (ValueError, SyntaxError):
            pass
    escaped_value = input_value.replace("'", "\\'").replace('"', '\\"')
    input_value = f"'{escaped_value}'"
    return literal_eval(input_value)


def compute_hash(text: str) -> str:
    """
    Return the hash of the given text using MD5 if available,
    otherwise use SHA256.

    Args:
        text (str): The input text to hash.

    Returns:
        str: The hexadecimal digest of the hash.
    """
    try:
        # Attempt to use MD5 by doing a dummy call.
        hashlib.md5(b"")
        hash_func = hashlib.md5
    except (AttributeError, ValueError):
        # MD5 is not available use sha256 instead
        hash_func = hashlib.sha256

    return hash_func(text.encode("utf-8")).hexdigest()


MAX_ERROR_MESSAGE_SIZE = 400
MAX_JSON_SIZE = 500
MAX_PARSING_DEPTH = 2


def check_object_depth(obj: Any, current_depth: int = 0) -> None:
    """Check if an object's nesting depth exceeds the maximum allowed depths

    Args:
        obj: the object to check
        current_depth: current nesting depth

    Raises:
        ValueError: if the object is too deeply nested
    """

    if current_depth > MAX_PARSING_DEPTH:
        raise ValueError("Object too deeply nested")
    if isinstance(obj, dict):
        for v in obj.values():
            check_object_depth(v, current_depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            check_object_depth(item, current_depth + 1)


def extract_error_json(error_message: str) -> dict:
    """Safely extracts the JSON part from the exception message

    Args:
        error_message (str): The exception message.

    Returns:
        dict: The extracted JSON part as a dictionary, or an error dictionary if extraction fails.
    """

    if len(error_message) > MAX_ERROR_MESSAGE_SIZE:
        error_message = error_message[:MAX_ERROR_MESSAGE_SIZE] + "... (truncated)"

    # OpenAI error format typically has "Error code: XXX - {...}" format
    if " - " in error_message:
        json_part = error_message.split(" - ", 1)[1].strip()

        if not (json_part.startswith("{") and json_part.endswith("}")):
            return {"error": {"message": error_message}}

        try:
            try:
                return json.loads(json_part, parse_constant=lambda x: x)
            except json.JSONDecodeError:
                # validate before using ast.literal_eval
                if len(json_part) < MAX_JSON_SIZE:
                    try:
                        import ast
                        import re

                        # looking for suspicious patterns
                        # is it ok?
                        if re.search(r"__[\w]+__", json_part):
                            raise ValueError("Potentially unsafe content")

                        # parse & validate depth
                        parsed = ast.literal_eval(json_part)
                        check_object_depth(parsed)
                        return parsed
                    except (SyntaxError, ValueError) as e:
                        return {"error": {"message": f"Invalid error format: {str(e)}"}}

            if len(json_part) < MAX_JSON_SIZE:
                json_part = json_part.replace("'", '"')
                return json.loads(json_part, parse_constant=lambda x: x)
        except (json.JSONDecodeError, ValueError):
            pass

        error_dict = {"error": {"message": error_message}}
        if "Error code:" in error_message:
            try:
                code = error_message.split("Error code:", 1)[1].strip().split(" ", 1)[0]
                error_dict["error"]["code"] = code
            except (IndexError, ValueError):
                pass
        return error_dict
    else:
        return {"error": {"message": error_message}}
