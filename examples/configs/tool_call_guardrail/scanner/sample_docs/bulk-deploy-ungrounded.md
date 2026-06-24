# Bulk service deployment abuse

Source: https://example.org/advisories/2026-bulk-deploy

Agents with a `deploy_service` capability can be driven to spin up many services
at once, exhausting cloud quota and running up cost. This concerns a capability
that is not part of the currently guarded tool set, so there is nothing here for
the guard to act on.
