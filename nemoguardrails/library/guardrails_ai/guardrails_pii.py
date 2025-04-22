# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any, List
from nemoguardrails.actions import action
from guardrails.hub import GuardrailsPII
from guardrails import Guard
from .utils import guardrails_ai_validation_mapping

# Initialize the guard once.
# Note: Default entities are used here. Consider if specific entities
# should be default or always passed via the action call.
# on_fail="fix" might be desired for PII to anonymize, but the mapping function
# currently just blocks based on `validation_passed`. If fixed output is needed,
# the action/mapping needs adjustment. Using "noop" aligns with block/pass.
pii_guard = Guard().use(GuardrailsPII(on_fail="noop"))

@action(
    name="validate_guardrails_ai_guardrails_pii",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_guardrails_ai_guardrails_pii(
    text: str,
    context: Optional[dict] = None,
    entities: Optional[List[str]] = None, # Allow specifying entities
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI PII detection."""
    try:
        # Pass entities via metadata if provided
        metadata = {"entities": entities} if entities else {}
        validation_result = pii_guard.validate(text, metadata=metadata)
        return {
            "validation_result": validation_result
        }
    except Exception as e:
        print(f"Error validating PII with Guardrails AI: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }
