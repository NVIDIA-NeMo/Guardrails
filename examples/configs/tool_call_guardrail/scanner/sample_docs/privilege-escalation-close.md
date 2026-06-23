# Account closure without step-up authorization

Source: https://example.org/cve/2026-close-account-stepup

Destructive operations such as a `close_account` tool can be triggered by an
agent on behalf of a principal who has not completed step-up authorization. This
is a privilege escalation: a routine session is used to invoke an operation that
should demand elevated clearance. The mitigation is to require a verified
clearance attribute on the principal before the call is allowed.

<!-- params: {"attr_name": "mfa_verified", "expected": true} -->
