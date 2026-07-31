# Data exfiltration via send_message to unowned recipient

Source: https://example.org/advisories/2026-hermes-sendmessage-exfiltration

The Hermes `send_message` tool enables the agent to send messages to any user or
channel on a supported platform (Slack, email, Discord, Telegram). Without
recipient authorization, an attacker can use a crafted request to cause the agent
to forward sensitive data retrieved from internal tools to an external recipient
the principal does not control — a confused-deputy exfiltration (RT-4 data
exfiltration in the Nemotron Agentic Safety guide).

The mitigation is an ownership-bypass rule on the `recipient` argument: the
target recipient must belong to the principal's `approved_recipients` set. Any
message to a recipient not in that set is blocked before dispatch, regardless of
the platform or message content.

<!-- params: {"arg_name": "recipient", "owned_attr": "approved_recipients"} -->
