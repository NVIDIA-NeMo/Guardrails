# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any, List
from nemoguardrails.actions import action
# Assuming CompetitorCheck is the correct import name/path
from guardrails.hub import CompetitorCheck
from guardrails import Guard
from .utils import guardrails_ai_validation_mapping

# Initialize the guard once. Check CompetitorCheck docs for necessary init params.
competitor_guard = Guard().use(CompetitorCheck(on_fail="noop"))

@action(
    name="validate_guardrails_ai_competitor_check",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_guardrails_ai_competitor_check(
    text: str,
    context: Optional[dict] = None,
    competitors: Optional[List[str]] = None, # Allow specifying competitors
    # Other metadata might be needed depending on the validator
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI competitor check."""
    try:
        # Prepare metadata, merging specific args like competitors if provided
        run_metadata = metadata or {}
        if competitors:
             run_metadata["competitors"] = competitors

        validation_result = competitor_guard.validate(text, metadata=run_metadata)
        return {
            "validation_result": validation_result
        }
    except Exception as e:
        print(f"Error validating competitor check with Guardrails AI: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }
