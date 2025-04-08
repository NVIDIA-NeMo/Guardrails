# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any
from nemoguardrails.actions import action
from guardrails.hub import ToxicLanguage
from guardrails import Guard

# Initialize the guard once as a module-level variable
toxicity_guard = Guard().use(ToxicLanguage)

def guardrails_ai_validation_mapping(result: dict) -> bool:
    """Mapping for guardrails_ai validation.

    Returns True if validation failed (should block), False otherwise.
    """
    return not result.get("valid", True)

@action(
    name="validate_guardrails_ai",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_guardrails_ai(
    text: str,
    context: Optional[dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI toxic language detection.

    Args:
        text: The text to validate
        context: Optional context information

    Returns:
        Dict containing validation results with keys:
        - valid: bool indicating if text passed validation
        - validation_result: raw validation result from Guardrails AI
    """
    try:
        validation_result = toxicity_guard.validate(text)
        return {
            "valid": validation_result.valid,
            "validation_result": validation_result
        }
    except Exception as e:
        # Log the error and return failed validation
        print(f"Error validating with Guardrails AI: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }