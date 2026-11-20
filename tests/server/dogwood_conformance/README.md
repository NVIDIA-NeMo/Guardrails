# Dogwood conformance examples

Each folder is copied unchanged from [dogwood-policy/dogwood](https://github.com/dogwood-policy/dogwood) at commit
`c7124710`, `dogwood-docs/examples/<name>/`: every example there that has an event trace, 56 in all. Each holds a
policy (`policy.dw`), its schemas, an event trace (`trace.log`) and the verdicts the reference `dogwood replay` CLI
gives for it (`expected.out`), which that project's CI checks. That project is licensed under Apache-2.0; its NOTICE
is in this folder.

`tests/server/test_hooks_dogwood_conformance.py` replays each trace through the `nemoguardrails-dogwood` binding, the
path the hooks server decides through, and compares the verdicts with `expected.out`.
