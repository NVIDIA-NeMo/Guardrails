# Brev Node Restart Runbook

Procedures for restarting NIMs, the Guardrails server, and benchmarking after a Brev node reboot.

**Node:** 8× H100 80 GB
**Repo:** `/ephemeral/Guardrails`
**NIM cache:** `/ephemeral/nim-cache`

---

## 1. Start the NIMs

Run both `docker run` commands from any directory. They start in the background (`-d`).

```bash
# Main LLM — Nemotron 3 Nano (GPUs 0,1 → port 8000)
docker run -d --gpus '"device=0,1"' \
  -e NGC_API_KEY=$NVIDIA_API_KEY \
  -v /ephemeral/nim-cache:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano:2.0.10

# Content Safety — Nemotron 3.5 (GPU 2 → port 8001)
docker run -d --gpus '"device=2"' \
  -e NGC_API_KEY=$NVIDIA_API_KEY \
  -v /ephemeral/nim-cache:/opt/nim/.cache \
  -p 8001:8001 \
  nvcr.io/nim/nvidia/nemotron-3.5-content-safety:2.0.5-variant
```

NIM startup takes several minutes while weights load. Poll until both respond:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

Both should return a model list (not a connection error) before proceeding.

---

## 2. Start the Guardrails server

Create a dedicated tmux session:

```bash
tmux rename-session guardrails_server
```

From `/ephemeral/Guardrails`, start the server in **one** of the two modes below.

### LLMRails mode

```bash
MAIN_MODEL_ENGINE=nim MAIN_MODEL_BASE_URL=http://localhost:8000/v1 \
  uv run nemoguardrails server \
  --config examples/configs/iorails_vs_llmrails_benchmark \
  --default-config-id iorails_vs_llmrails_benchmark \
  --port 9000 --verbose 2>&1 | tee /ephemeral/guardrails_llmrails.log
```

### IORails mode

```bash
MAIN_MODEL_ENGINE=nim MAIN_MODEL_BASE_URL=http://localhost:8000/v1 \
  NEMO_GUARDRAILS_IORAILS_ENGINE=1 \
  uv run nemoguardrails server \
  --config examples/configs/iorails_vs_llmrails_benchmark \
  --default-config-id iorails_vs_llmrails_benchmark \
  --port 9000 --verbose 2>&1 | tee /ephemeral/guardrails_iorails.log
```

Wait for `Uvicorn running on http://0.0.0.0:9000` in the log before running benchmarks.

**Note:** restart the Guardrails server between LLMRails and IORails runs — the engine mode is set at startup and cannot be changed without a restart.

---

## 3. Run benchmarks

Open a second tmux session:

```bash
tmux rename-session benchmarking
```

Change into the benchmark directory:

```bash
cd /ephemeral/Guardrails/benchmark
```

Run all four scenarios against the active engine (replace `iorails` with `llmrails` as appropriate):

```bash
./run_scenarios.sh iorails
```

Or run a single scenario to test a flag change before a full sweep:

```bash
PYTHONPATH=/ephemeral/Guardrails \
  /ephemeral/venv-aiperf/bin/python -m benchmark.aiperf \
  --config-file /ephemeral/Guardrails/benchmark/aiperf/configs/single_concurrency.yaml \
  --use-legacy-max-tokens
```

Results land in `benchmark/aiperf_results/`.

---

## 4. Data-Parallel Setup (Agent Scenario / Throughput Testing)

Tim Gasser's recommendation (2026-08-05): replicate Nano across 6 GPUs and Content Safety across 2 GPUs so the downstream models don't bottleneck before IORails/LLMRails overhead is visible. Percentile latency must stay stable as concurrency ramps.

**Layout:** 6 × Nano (GPUs 0–5, 1 GPU each) + 2 × Content Safety (GPUs 6–7, 1 GPU each).
**Why Nano fits on 1 GPU:** 30B-A3B-NVFP4 ≈ 15 GB weights — well within 80 GB H100.
**Load balancing:** Guardrails points at a single URL, so put nginx in round-robin in front of all Nano replicas (and optionally the CS replicas).

### Start 6 Nano replicas

```bash
for i in 0 1 2 3 4 5; do
  docker run -d --gpus "\"device=$i\"" \
    -e NGC_API_KEY=$NVIDIA_API_KEY \
    -v /ephemeral/nim-cache:/opt/nim/.cache \
    -p $((8010 + i)):8000 \
    nvcr.io/nim/nvidia/nemotron-3-nano:2.0.10
done
# Replicas on ports 8010–8015
```

### Start 2 Content Safety replicas

```bash
for i in 6 7; do
  docker run -d --gpus "\"device=$i\"" \
    -e NGC_API_KEY=$NVIDIA_API_KEY \
    -v /ephemeral/nim-cache:/opt/nim/.cache \
    -p $((8001 + i - 6)):8001 \
    nvcr.io/nim/nvidia/nemotron-3.5-content-safety:2.0.5-variant
done
# Replicas on ports 8001–8002
```

### nginx load balancer

Install nginx (`sudo apt-get install -y nginx`) and write `/etc/nginx/conf.d/nim_lb.conf`:

```nginx
upstream nano_lb {
    server localhost:8010;
    server localhost:8011;
    server localhost:8012;
    server localhost:8013;
    server localhost:8014;
    server localhost:8015;
}

upstream cs_lb {
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 8000;
    location / { proxy_pass http://nano_lb; }
}

server {
    listen 8003;
    location / { proxy_pass http://cs_lb; }
}
```

Reload nginx: `sudo nginx -s reload`

Guardrails then points at `http://localhost:8000/v1` (Nano, load-balanced) and `http://localhost:8003` (Content Safety, load-balanced) — same as the single-instance setup from Guardrails' perspective.

**Note:** the Guardrails config's content safety NIM URL may need updating from port 8001 → 8003 when using this layout.
