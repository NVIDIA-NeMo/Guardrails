# Egress-rail prototype (SPIKE)

> **Status: throwaway spike.** Built on branch `spike/schilton/egress-rail` to
> explore the layer the context-exfiltration finding was triaged to (see
> [`synthesis/TRIAGE.md`](synthesis/TRIAGE.md)). Not wired into `config.py` and
> not intended to merge as-is.

## What it is

`policy.py` authorizes **one** `(tool, args, principal)` call in isolation — it is
stateless by design. Some risks are invisible at that altitude: relaying context
to an attacker over a series of individually-benign `http_request` calls is a
property of the *sequence*, not any single call.

`egress.py` is a **session-aware** companion that watches outbound (egress) tool
calls within a session and vetoes aggregate behavior:

| Signal | Limit | Catches |
| --- | --- | --- |
| request count | `max_requests` | slow, sustained leak |
| cumulative volume | `max_cumulative_bytes` | drip exfiltration |
| distinct hosts | `max_distinct_hosts` | fan-out to many endpoints |
| burst rate | `max_requests_per_window` / `window_seconds` | spikes |

It is framework-free (no Guardrails import), like `policy.py`, so it runs offline
and could be hosted in a Guardrails action/output rail or an orchestrator.

## How it layers with the per-call guard

`authorize_with_egress(guard, monitor, session_id, call, principal)` composes the
two gates an orchestrator would apply in place of a raw tool:

1. the stateless per-call guard (`HARDENED_GUARD`) authorizes the single call;
2. only if it passes does the stateful `EgressMonitor` get to veto on aggregate
   behavior.

So a metadata-egress URL is still stopped by the per-call guard first; the
monitor only adds the cross-call dimension on top.

## Run it

```bash
python demo_egress.py                 # offline, four labelled scenarios
poetry run pytest tests/test_egress.py
```

## Honest caveats (why this is a spike, not a merge)

- **Heuristic proxy, not detection.** Static thresholds: a patient attacker stays
  under them; a legitimate data-sync agent trips them. The signal the source paper
  actually points at — sensitivity-aware data-flow / taint tracking — is a
  different discipline and is not attempted here.
- **State model.** In-memory, single-process, keyed by `(session_id, principal)`.
  A real deployment needs a shared store and an explicit session lifecycle
  (`reset()` is the stub).
- **Breaks the stateless contract.** This is deliberately a *separate* layer, not
  a `policy.Rule`, precisely so the per-call guard's thin, portable core stays
  intact. If a future decision wants this in-tree, that contract change is the
  thing to weigh.
- **Byte estimate is a stand-in** (`len` of string args), since the mock
  `http_request` carries no real body.
