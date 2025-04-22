# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any
from nemoguardrails.actions import action
# Assuming ProvenanceLLM is the correct import name/path
from guardrails.hub import ProvenanceLLM
from guardrails import Guard
from .utils import guardrails_ai_validation_mapping

# Initialize the guard once. Check ProvenanceLLM docs for necessary init params.
provenance_llm_guard = Guard().use(ProvenanceLLM(on_fail="noop"))

@action(
    name="validate_provenance_llm",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_provenance_llm(
    text: str,
    context: Optional[dict] = None,
    # ProvenanceLLM might require metadata e.g. sources
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI ProvenanceLLM detection."""
    try:
        # Pass metadata if provided, default to empty dict
        validation_result = provenance_llm_guard.validate(text, metadata=metadata or {})
        return {
            "validation_result": validation_result
        }
    except Exception as e:
        print(f"Error validating with Guardrails AI ProvenanceLLM: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }
