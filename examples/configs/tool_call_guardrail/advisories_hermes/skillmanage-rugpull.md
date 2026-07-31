# Skill rug-pull via skillmanage tool

Source: https://example.org/advisories/2026-hermes-skill-rugpull

The Hermes `skillmanage` tool allows an agent to create, patch, or delete
reusable procedural skills that encode best-practice workflows. An attacker who
can invoke `skillmanage` with `action=patch` or `action=delete` can silently
modify or remove safety-critical skills (for example, a PR-review checklist that
enforces code review), replacing them with versions that omit security controls.
This is the skill rug-pull pattern (RT-6 in the Nemotron Agentic Safety guide).

Two mitigations are required. First, modifying agent skills should be restricted
to principals with the `elevated` attribute — privilege-escalation check. Second,
the `delete` action should be blocked entirely via a disallowed-target rule; skill
deletion is irreversible and provides no recovery path if abused.

<!-- params: {"attr_name": "elevated", "expected": true} -->
<!-- params: {"arg_name": "action", "denied": ["delete"]} -->
