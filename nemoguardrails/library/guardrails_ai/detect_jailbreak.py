# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any
from nemoguardrails.actions import action
from guardrails.hub import DetectJailbreak
from guardrails import Guard
from .utils import guardrails_ai_validation_mapping

# Initialize the guard once as a module-level variable
# Check DetectJailbreak documentation for relevant init parameters like threshold
jailbreak_guard = Guard().use(DetectJailbreak(on_fail="noop")) # Specify on_fail

@action(
    name="validate_guardrails_ai_detect_jailbreak",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_guardrails_ai_jailbreak(
    text: str,
    context: Optional[dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI jailbreak detection."""
    try:
        validation_result = jailbreak_guard.validate(text)
        return {
            "validation_result": validation_result
        }
    except Exception as e:
        print(f"Error validating jailbreak with Guardrails AI: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }
