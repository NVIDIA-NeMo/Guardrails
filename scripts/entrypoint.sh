#!/bin/bash

# Allow runtime overrides via env vars or args
CONFIG_ID="${CONFIG_ID:-${1:-nemo}}"
PORT="${PORT:-${2:-8000}}"

CONFIG_DIR="/app/config/${CONFIG_ID}"

echo "🚀 Starting NeMo Guardrails with config from: $CONFIG_DIR (port: $PORT)"

# Validate config exists
if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
  echo "❌ ERROR: config.yaml not found in $CONFIG_DIR"
  exit 1
fi

if [[ ! -f "$CONFIG_DIR/rails.co" ]]; then
  echo "❌ ERROR: rails.co not found in $CONFIG_DIR (ConfigMap is read-only, please provide it)"
  exit 1
fi

echo "✅ Configuration validated. Starting server..."

CMD=(/app/.venv/bin/nemoguardrails server \
  --config "/app/config" \
  --port "$PORT" \
  --default-config-id "$CONFIG_ID" \
  --disable-chat-ui)

# If OTEL_EXPORTER_OTLP_ENDPOINT is set and opentelemetry-instrument is available,
# wrap the server with auto-instrumentation
if [[ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ]] && command -v opentelemetry-instrument &>/dev/null; then
  echo "OpenTelemetry enabled: endpoint=$OTEL_EXPORTER_OTLP_ENDPOINT service=${OTEL_SERVICE_NAME:-nemo-guardrails}"
  export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-nemo-guardrails}"
  exec opentelemetry-instrument "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
