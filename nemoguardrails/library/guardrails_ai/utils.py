# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any
from nemoguardrails.actions import action
from guardrails.hub import ToxicLanguage
from guardrails import Guard

# Initialize the guard once as a module-level variable
toxicity_guard = Guard().use(ToxicLanguage)

def guardrails_ai_validation_mapping(result: dict) -> bool:
    """Mapping for guardrails_ai validation result.

    Returns True if validation failed (should block), False otherwise.
    Assumes the result object has a boolean `valid` attribute.
    """
    # The Guardrails AI `validate` method returns a ValidationResult object.
    # On failure (PII found, Jailbreak detected, etc.), it's often a FailResult.
    # Both PassResult and FailResult have a `validation_passed` boolean attribute
    # which indicates if the validation criteria were met.
    # FailResult also often contains `fixed_value` if a fix like anonymization was applied.
    # We map `validation_passed=False` to `True` (block) and `validation_passed=True` to `False` (don't block).

    # Check if 'validation_result' key exists and has the attribute
    validation_result = result.get("validation_result")
    if hasattr(validation_result, 'validation_passed'):
        return not validation_result.validation_passed
    elif hasattr(validation_result, 'valid'): # Fallback for older/different patterns
         return not validation_result.valid

    # Default to blocking if the structure is unexpected or validation failed implicitly
    # Or handle the case where only 'valid' boolean is returned directly in the dict
    return not result.get("valid", False) # Default to blocking if 'valid' is missing or False
