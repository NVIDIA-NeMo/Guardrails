# Prometheus and Grafana for the Guardrails server

A ready-to-run Prometheus + Grafana pair that scrapes the non-streaming
admission-queue metrics exposed by `nemoguardrails server --metrics-exporter prometheus`
and provisions a dashboard for them.

The full walkthrough, including how to start the server, lives in the docs:
[Prometheus and Grafana for the Guardrails Server](../../../docs/observability/metrics/server-prometheus-grafana.mdx).

## Quick start

1. Start a Guardrails server on the host with the exporter enabled. The config
   must set `metrics.enabled: true` and the IORails engine must be active:

   ```bash
   pip install "nemoguardrails[server]"
   export NEMO_GUARDRAILS_IORAILS_ENGINE=1
   nemoguardrails server --config ./config --metrics-exporter prometheus
   ```

2. Start Prometheus and Grafana from this directory. Pick the Grafana admin
   password yourself; Compose refuses to start without it:

   ```bash
   GRAFANA_ADMIN_PASSWORD='<choose a password>' docker compose up -d
   ```

3. Open Grafana at <http://localhost:3000> and sign in as `admin` with that
   password. The **NeMo Guardrails - Non-Streaming Admission Queue** dashboard
   is in the **NeMo Guardrails** folder, and the Prometheus data source is
   pre-configured.

Anonymous access is disabled, and both containers publish their ports on
`127.0.0.1` only, so they are not reachable from other machines.

Prometheus scrapes `host.docker.internal:9464`. Edit
`prometheus/prometheus.yml` if the server runs on another host or port.

## What is on the dashboard

A single time-series panel with the three admission-queue metrics:

| Series | Query |
|---|---|
| queued | `guardrails_nonstream_queued` |
| active | `guardrails_nonstream_active` |
| rejections/s | `rate(guardrails_nonstream_rejections_total[1m])` |

## Clean up

```bash
docker compose down
```
