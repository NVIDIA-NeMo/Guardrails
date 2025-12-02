#!/bin/bash
# Live documentation server with auto-rebuild on file changes
# Usage: ./serve.sh [port]
# Default port: 8000

set -e

PORT=${1:-8000}
SOURCE_DIR="."
BUILD_DIR="_build/html"

echo "================================================"
echo "NeMo Guardrails Documentation Server"
echo "================================================"
echo ""
echo "Starting live documentation server on port ${PORT}..."
echo "Documentation will auto-rebuild on file changes."
echo ""
echo "Open your browser to: http://127.0.0.1:${PORT}"
echo ""
echo "Press Ctrl+C to stop the server."
echo "================================================"
echo ""

# Run sphinx-autobuild with the following options:
# --port: Port to serve on
# --host: Host to bind to (0.0.0.0 allows external access)
# --open-browser: Automatically open browser (optional, commented out by default)
# --ignore: Patterns to ignore for rebuilding
# --watch: Additional directories to watch (if needed)
# --delay: Delay in seconds before rebuilding (default: 0)

sphinx-autobuild \
    "${SOURCE_DIR}" \
    "${BUILD_DIR}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --ignore "*.swp" \
    --ignore "*.swo" \
    --ignore "*~" \
    --ignore ".DS_Store" \
    --ignore "_build/*" \
    --ignore "*.pyc" \
    --ignore "__pycache__/*" \
    --ignore ".git/*" \
    --delay 1 \
    --watch ../nemoguardrails \
    --re-ignore "_build/.*" \
    --re-ignore ".*\.egg-info.*"
