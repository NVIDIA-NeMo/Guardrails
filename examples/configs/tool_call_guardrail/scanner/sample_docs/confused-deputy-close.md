# Confused-deputy account closure

Source: https://example.org/cve/2026-confused-deputy-close

When an agent exposes a destructive operation such as a `close_account` tool but
the surrounding policy performs no ownership check, an attacker can use the agent
as a confused deputy: the agent holds the privilege, and a crafted request gets
it to close an account the acting principal does not own. The mitigation is to
require that the target account belongs to the principal before the call runs.

<!-- params: {"arg_name": "account_id"} -->
