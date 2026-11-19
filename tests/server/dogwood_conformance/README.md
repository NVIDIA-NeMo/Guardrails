# Dogwood conformance examples

Each folder is copied unchanged (`policy.dw`, `trace.log`, `expected.out`) from
[dogwood-policy/dogwood](https://github.com/dogwood-policy/dogwood) at commit `c7124710`,
`dogwood-docs/examples/<name>/`. That project is licensed under Apache-2.0 (Copyright Amazon.com, Inc. or its
affiliates). Its own CI checks each `expected.out` against the reference `dogwood replay` CLI.

`tests/server/test_hooks_dogwood_conformance.py` replays each trace through the hooks server's Dogwood evaluator and
compares the verdicts with `expected.out`. These are the examples whose policies fall inside the subset the hooks
server evaluates, with the default event schema; examples with macros, providers, `previous`, plain `since`,
`output.*`, `callerPrincipal` or a custom event schema are not here.
