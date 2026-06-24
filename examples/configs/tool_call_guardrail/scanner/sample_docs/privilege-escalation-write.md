# Writing protected files without step-up clearance

Source: https://example.org/cve/2026-write-file-stepup

A sensitive `write_file` operation on protected paths can be triggered by an
agent on behalf of a principal who has not completed step-up authorization. This
is a privilege escalation: a routine session is used to perform an action that
should demand elevated clearance. The mitigation is to require a verified
clearance attribute on the principal before the call is allowed.

<!-- params: {"attr_name": "elevated", "expected": true} -->
