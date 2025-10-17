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

"""Dynamic validator loading for Guardrails AI integration."""

import importlib
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Type

try:
    from guardrails import Guard
except ImportError:
    # Mock Guard class for when guardrails is not available
    class Guard:
        def __init__(self):
            pass

        def use(self, validator):
            return self

        def validate(self, text, metadata=None):
            return None


from nemoguardrails.actions import action
from nemoguardrails.library.guardrails_ai.errors import GuardrailsAIValidationError
from nemoguardrails.library.guardrails_ai.registry import get_validator_info
from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)


# cache for loaded validator classes and guard instances
_validator_class_cache: Dict[str, Type] = {}
_guard_cache: Dict[tuple, Guard] = {}


def guardrails_ai_validation_mapping(result: Dict[str, Any]) -> bool:
    """Map Guardrails AI validation result to NeMo Guardrails format."""
    # The Guardrails AI `validate` method returns a ValidationResult object.
    # On failure (PII found, Jailbreak detected, etc.), it's often a FailResult.
    # Both PassResult and FailResult have a `validation_passed` boolean attribute
    # which indicates if the validation criteria were met.
    # FailResult also often contains `fixed_value` if a fix like anonymization was applied.
    # We map `validation_passed=False` to `True` (block) and `validation_passed=True` to `False` (don't block).
    validation_result = result.get("validation_result", {})

    # Handle both dict and object formats
    if hasattr(validation_result, "validation_passed"):
        valid = validation_result.validation_passed
    else:
        valid = validation_result.get("validation_passed", False)

    return valid  # {"valid": valid, "validation_result": validation_result}


# TODO: we need to do this
# from guardrails.hub import RegexMatch, ValidLength
# from guardrails import Guard
#
# guard = Guard().use_many(
#     RegexMatch(regex="^[A-Z][a-z]*$"),
#     ValidLength(min=1, max=12)
# )
#
# print(guard.parse("Caesar").validation_passed)  # Guardrail Passes
# print(
#     guard.parse("Caesar Salad")
#     .validation_passed
# )  # Guardrail Fails due to regex match
# print(
#     guard.parse("Caesarisagreatleader")
#     .validation_passed
# )  # Guardrail Fails due to length


@action(
    name="validate_guardrails_ai_input",
    output_mapping=guardrails_ai_validation_mapping,
    is_system_action=False,
)
def validate_guardrails_ai_input(
    validator: str,
    config: RailsConfig,
    context: Optional[dict] = None,
    text: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Unified action for all Guardrails AI validators.

    Args:
        validator: Name of the validator to use (from VALIDATOR_REGISTRY)
        text: Text to validate
        context: Optional context dictionary

    Returns:
        Dict with validation_result
    """

    text = text or context.get("user_message", "")
    if not text:
        raise ValueError("Either 'text' or 'context' must be provided.")

    validator_config = config.rails.config.guardrails_ai.get_validator_config(validator)
    parameters = validator_config.parameters or {}
    metadata = validator_config.metadata or {}

    joined_parameters = {**parameters, **metadata}

    validation_result = validate_guardrails_ai(validator, text, **joined_parameters)

    # Transform to the expected format for Colang flows
    return validation_result


@action(
    name="validate_guardrails_ai_output",
    output_mapping=guardrails_ai_validation_mapping,
    is_system_action=False,
)
def validate_guardrails_ai_output(
    validator: str,
    context: Optional[dict] = None,
    text: Optional[str] = None,
    config: Optional[RailsConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Unified action for all Guardrails AI validators.

    Args:
        validator: Name of the validator to use (from VALIDATOR_REGISTRY)
        text: Text to validate
        context: Optional context dictionary

    Returns:
        Dict with validation_result
    """

    text = text or context.get("bot_message", "")
    if not text:
        raise ValueError("Either 'text' or 'context' must be provided.")

    validator_config = config.rails.config.guardrails_ai.get_validator_config(validator)
    parameters = validator_config.parameters or {}
    metadata = validator_config.metadata or {}

    # join parameters and metadata into a single dict
    joined_parameters = {**parameters, **metadata}

    validation_result = validate_guardrails_ai(validator, text, **joined_parameters)

    return validation_result


def validate_guardrails_ai(validator_name: str, text: str, **kwargs) -> Dict[str, Any]:
    """Unified action for all Guardrails AI validators.

    Args:
        validator: Name of the validator to use (from VALIDATOR_REGISTRY)
        text: Text to validate


    Returns:
        Dict with validation_result
    """

    try:
        # extract metadata if provided as a dict

        metadata = kwargs.pop("metadata", {})
        validator_params = kwargs

        validator_params = {k: v for k, v in validator_params.items() if v is not None}

        # get or create the guard with all non-metadata params
        guard = _get_guard(validator_name, **validator_params)

        try:
            validation_result = guard.validate(text, metadata=metadata)
            return {"validation_result": validation_result}
        except GuardrailsAIValidationError as e:
            # handle Guardrails validation errors (when on_fail="exception")
            # return a failed validation result instead of raising
            log.warning(f"Guardrails validation failed for {validator_name}: {str(e)}")

            # create a mock validation result for failed validations
            class FailedValidation:
                validation_passed = False
                error = str(e)

            return {"validation_result": FailedValidation()}

    except Exception as e:
        log.error(f"Error validating with {validator_name}: {str(e)}")
        raise GuardrailsAIValidationError(f"Validation failed: {str(e)}")


@lru_cache(maxsize=None)
def _load_validator_class(validator_name: str) -> Type:
    """Dynamically load a validator class."""
    cache_key = f"class_{validator_name}"

    if cache_key in _validator_class_cache:
        return _validator_class_cache[cache_key]

    try:
        validator_info = get_validator_info(validator_name)

        module_name = validator_info["module"]
        class_name = validator_info["class"]

        try:
            module = importlib.import_module(module_name)
            validator_class = getattr(module, class_name)
            _validator_class_cache[cache_key] = validator_class
            return validator_class
        except (ImportError, AttributeError):
            log.warning(
                f"Could not import {class_name} from {module_name}. "
                f"Make sure to install it first: guardrails hub install {validator_info['hub_path']}"
            )
            raise ImportError(
                f"Validator {validator_name} not installed. "
                f"Install with: guardrails hub install {validator_info['hub_path']}"
            )

    except Exception as e:
        raise ImportError(f"Failed to load validator {validator_name}: {str(e)}")


def _get_guard(validator_name: str, **validator_params) -> Guard:
    """Get or create a Guard instance for a validator."""

    # create a hashable cache key
    def make_hashable(obj):
        if isinstance(obj, list):
            return tuple(obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        return obj

    cache_items = [(k, make_hashable(v)) for k, v in validator_params.items()]
    cache_key = (validator_name, tuple(sorted(cache_items)))

    if cache_key not in _guard_cache:
        validator_class = _load_validator_class(validator_name)

        # TODO(@zayd): is this needed?
        # default handling for all validators
        if "on_fail" not in validator_params:
            validator_params["on_fail"] = "noop"

        try:
            validator_instance = validator_class(**validator_params)
        except TypeError as e:
            log.error(
                f"Failed to instantiate {validator_name} with params {validator_params}: {str(e)}"
            )
            raise

        guard = Guard().use(validator_instance)
        _guard_cache[cache_key] = guard

    return _guard_cache[cache_key]
