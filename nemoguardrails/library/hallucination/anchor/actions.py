import os
import requests
from typing import Optional
from nemoguardrails.actions import action

@action(name="check_anchor_drift", is_system_action=False)
async def check_anchor_drift(
    context: Optional[dict] = None,
    threshold: float = 0.95
):
    """
    Checks the last bot message for intent drift using the anchor-engine API.
    
    Args:
        context: The current conversation context, containing the bot message and ground truth.
        threshold: The HHEM score threshold below which the action triggers a drift alert.
    """
    api_key = os.environ.get("ANCHOR_API_KEY")
    if not api_key:
        raise ValueError("Missing ANCHOR_API_KEY. Please sign up at https://anchor-app-one.vercel.app to get your free API key.")

    # Get the last bot message and the context/reference
    last_bot_message = context.get("last_bot_message")
    source_context = context.get("relevant_chunks", "") # Typical NeMo context key

    if not last_bot_message or not source_context:
        return True

    try:
        # Call the Anchor Scoring API
        # Target: https://anchor-app-one.vercel.app/api/score
        response = requests.post(
            "https://anchor-app-one.vercel.app/api/score",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "action": last_bot_message,
                "context": source_context,
                "threshold": threshold
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            # If 'allow' is False, drift was detected
            return result.get("allow", True)
            
    except Exception as e:
        # Log error or silently fail to maintain UX
        return True

    return True
