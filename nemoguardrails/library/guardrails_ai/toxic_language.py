# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any
from nemoguardrails.actions import action
from guardrails.hub import ToxicLanguage
from guardrails import Guard
from .utils import guardrails_ai_validation_mapping

# Initialize the guard once as a module-level variable
toxic_language_guard = Guard().use(ToxicLanguage(on_fail="noop")) # Specify on_fail for clarity

@action(
    name="validate_guardrails_ai_toxic_language",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_guardrails_ai_toxic_language(
    text: str,
    context: Optional[dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI toxic language detection.

    Args:
        text: The text to validate
        context: Optional context information

    Returns:
        Dict containing validation results. The `guardrails_ai_validation_mapping`
        interprets this for blocking decisions.
    """
    try:
        validation_result = toxic_language_guard.validate(text)
        # Return structure includes the raw result for potential logging/debugging
        return {
            "validation_result": validation_result
            # The 'valid' key might be implicitly derived by the mapping function
            # from validation_result.validation_passed
        }
    except Exception as e:
        print(f"Error validating toxic language with Guardrails AI: {str(e)}")
        # Return a structure indicating failure, mapping function should handle this
        return {
            "valid": False, # Explicitly False on error
            "error": str(e)
        }
