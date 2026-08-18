# Community telemetry figures

These figures summarize anonymous NVIDIA NeMo Guardrails library telemetry for exact `0.22.0` and `0.23.0` release builds between `2026-05-22T11:19:23.257Z` and `2026-08-18T11:29:16.664Z`.

The source file contains cleaned aggregate counts only. It excludes session identifiers, user content, prompts, model names, endpoints, credentials, and internal data-source details.

The source aggregation uses these filters:

```text
eventName: "guardrails_usage_event"
parameters.nemoSource: "guardrails"
parameters.event: "startup"
parameters.nemoguardrailsVersion: ("0.22.0" or "0.23.0")
@timestamp: 2026-05-22T11:19:23.257Z through 2026-08-18T11:29:16.664Z
```

The rail-type and built-in-feature values use the cardinality of `client.sessionId` for each category. Elasticsearch cardinality values are estimates. The configured-rail values use startup-record counts so the buckets remain mutually exclusive.

Generate the SVG and PNG assets from the repository root:

```bash
uv run --locked python scripts/generate_community_telemetry_figures.py
```

The rail-type and built-in-feature figures use the estimated unique count of startup process sessions as their denominator. A process session can use multiple rail types or built-in features, so percentages in those figures do not sum to 100%.

The configured-rail figure uses startup records with at least one configured rail. It uses records instead of process sessions because multiple guardrails initializations in one process share a session identifier and can report different configurations. This makes the `1`, `2`, and `3+` buckets mutually exclusive and ensures their percentages sum to 100%. Dialog rails are represented by `railTypesInUse` and are not included in `numRailsConfigured`.
