# Transfers to sanctioned destination accounts

Source: https://example.org/advisories/2026-sanctioned-destination

Agents exposing a `transfer_funds` tool have been steered into moving money to a
small set of known-bad destination accounts (sanctioned or fraud-associated).
The mitigation is a denylist on the destination argument: the call is blocked
when `to_account` is one of the forbidden values, regardless of the transfer
amount.

<!-- params: {"arg_name": "to_account", "denied": ["acct-9999"]} -->
