# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any, List
from nemoguardrails.actions import action
# Assuming RestrictToTopic is the correct import name/path
# May need adjustment e.g., from guardrails.hub import TryolabsRestrictToTopic
from guardrails.hub import RestrictToTopic
from guardrails import Guard
from .utils import guardrails_ai_validation_mapping

# Note: RestrictToTopic *requires* `valid_topics`.
# Initialization here without topics is not useful.
# The action handles dynamic initialization or expects topics via metadata.

@action(
    name="validate_guardrails_ai_restrict_to_topic",
    is_system_action=True,
    output_mapping=guardrails_ai_validation_mapping,
)
async def validate_guardrails_ai_restrict_to_topic(
    text: str,
    context: Optional[dict] = None,
    valid_topics: Optional[List[str]] = None, # Expect topics per call
    # Check validator docs if it supports topics via metadata instead
    metadata: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates text using Guardrails AI restrict to topic."""
    run_metadata = metadata or {}
    final_topics = valid_topics or run_metadata.get("valid_topics")

    if not final_topics:
         # Cannot validate without topics
         print("Error: RestrictToTopic validator requires 'valid_topics' list.")
         return {"valid": False, "error": "Missing valid_topics configuration"}
    try:
        # Option 1: Initialize guard dynamically (less efficient but clear dependency)
        dynamic_restrict_to_topic_guard = Guard().use(
            RestrictToTopic(valid_topics=final_topics, on_fail="noop")
        )
        validation_result = dynamic_restrict_to_topic_guard.validate(text)

        # Option 2: Pass via metadata (if validator supports it - check docs)
        # run_metadata["valid_topics"] = final_topics
        # Ensure a guard is initialized (maybe without topics?) if using this.
        # Needs a module-level guard initialized appropriately.
        # validation_result = restrict_topic_guard.validate(text, metadata=run_metadata)

        return {
            "validation_result": validation_result
        }
    except Exception as e:
        print(f"Error validating restrict_to_topic with Guardrails AI: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }
