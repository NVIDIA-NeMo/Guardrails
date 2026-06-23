# Identifier injection in account lookups

Source: https://example.org/research/2026-account-id-injection

When an agent forwards a free-form account identifier into a `read_account`
tool, a crafted value can carry injection or path-traversal metacharacters that
the downstream system mishandles. The mitigation is to require the `account_id`
argument to fully match a strict allowlisted pattern (e.g. `acct-` followed by
digits), rejecting any malformed value before the call runs.

<!-- params: {"arg_name": "account_id", "pattern": "acct-[0-9]+"} -->
