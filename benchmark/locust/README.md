# Locust Load Testing for NeMo Guardrails

This directory contains a Locust-based load testing framework for the NeMo Guardrails OpenAI-compatible server.

## Introduction

The [Locust](https://locust.io/) stress-testing tool ramps up concurrent users making API calls to the `/v1/chat/completions` endpoint of an OpenAI-compatible LLM with configurable parameters.
The focus is on ramping load up potentially beyond what a system can handle, and measuring how
gracefully it degrades under higher-than-expected load.

## Quickstart

Use this quickstart to load test the repository's local mock Guardrails stack.
It requires no GPU or external model credentials.

### 1. Start the mock stack

This mirrors
[1. Run Server-side components](../README.md#1-run-server-side-components-guardrails-openai-compatible-service-with-mock-llms-for-content-safety-and-application-llms)
and
[2. Validate services are running correctly](../README.md#2-validate-services-are-running-correctly)
in the benchmark README, with `locust` added. See those steps for the full
annotated walkthrough.

First raise the file descriptor limit. This matters more for Locust than for
steady-state benchmarks: it caps how many concurrent users you can ramp to
before the operating system, rather than Guardrails, becomes the bottleneck.

```shell
$ ulimit -n 65536
```

Install the dependencies. `honcho` reads the [`Procfile`](../Procfile) to bring
up the services, and `locust` drives the load test:

```shell
$ uv sync --locked --extra server
$ uv pip install honcho locust
```

Start the Guardrails server and the mock LLMs. Wait for all three `Uvicorn
running on ...` messages, which are typically not on consecutive lines:

```shell
$ cd benchmark
$ uv run honcho start
13:40:33 system    | gr.1 started (pid=93634)
13:40:33 system    | app_llm.1 started (pid=93635)
13:40:33 system    | cs_llm.1 started (pid=93636)
...
13:40:41 app_llm.1 | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
...
13:40:41 cs_llm.1  | INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
...
13:40:45 gr.1      | INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

In a second shell, validate that all three services are answering:

```shell
$ cd benchmark
$ scripts/validate_mocks.sh
Starting LLM endpoint health check...

--- Checking Port: 8000 ---
Health Check PASSED: Status is 'healthy'.
Model Check PASSED: Found 'meta/llama-3.3-70b-instruct' in model list.
--- Port 8000: ALL CHECKS PASSED ---
...
--- Final Summary ---
Port 8000 (meta/llama-3.3-70b-instruct): PASSED
Port 8001 (nvidia/llama-3.1-nemoguard-8b-content-safety): PASSED
Port 9000 (Rails Config): PASSED
---------------------
Overall Status: All endpoints are healthy!
```

The mock application model runs on port `8000`, the mock content-safety model on
port `8001`, and the Guardrails server on port `9000`.

### 2. Run a small load test

Run this from the repository root, so `cd ..` first if you are still in
`benchmark/` from the validation step. It starts one Locust user for 15
seconds, which is suitable for confirming the setup before increasing load.

```bash
mkdir -p locust_results
LOCUST_CONFIG_ID=content_safety_local \
LOCUST_MODEL=meta/llama-3.3-70b-instruct \
LOCUST_MESSAGE="Hello, what can you do?" \
uv run locust \
  -f benchmark/locust/locustfile.py \
  --host http://localhost:9000 \
  --users 1 \
  --spawn-rate 1 \
  --run-time 15s \
  --headless \
  --only-summary \
  --html locust_results/report.html
```

To use the Locust web UI instead, omit `--headless` and `--only-summary`, then
open `http://localhost:8089`.

Increase `--users`, `--spawn-rate`, and `--run-time` only after the smoke test
completes successfully. Stop the mock stack with `Ctrl-C` in its terminal.

For repeatable or tuned runs, use the YAML-driven CLI described below instead of
passing flags by hand.

## Running Benchmarks With The CLI

The `benchmark.locust` CLI wraps Locust and reads load-testing parameters from a
YAML configuration file, so a run can be version-controlled and repeated.
Set `headless: false` in your YAML config to use Locust's interactive web UI, then
open http://localhost:8089 to control the test and view real-time metrics.

```bash
uv run python -m benchmark.locust benchmark/locust/configs/local.yaml
```

Point `host` at the server you mean to test before running this. A `host` on
port `8000` targets the mock application LLM rather than Guardrails, and that
model ignores the `config_id` the load test sends — so the run reports success
while measuring the mock instead of the guardrails you meant to benchmark.

Note also that [`configs/local.yaml`](configs/local.yaml) ramps to 1024 users
over 120 seconds. Run the single-user smoke test above before using it.

### Concurrency sweeps

A single run measures one concurrency level. To see how the server degrades as
load rises, sweep over a list of user counts. Each level is an independent
Locust run:

```bash
uv run python -m benchmark.locust benchmark/locust/configs/sweep_concurrency.yaml
```

[`configs/sweep_concurrency.yaml`](configs/sweep_concurrency.yaml) uses the
batch form, which mirrors [the AIPerf runner](../aiperf/configs):

```yaml
batch_name: sweep_concurrency
output_base_dir: locust_results

base_config:
  host: "http://localhost:9000"
  config_id: content_safety_local
  model: meta/llama-3.3-70b-instruct
  spawn_rate: 16
  run_time: 120

sweeps:
  target_users: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
```

Sweeping more than one parameter runs every combination. Any `base_config`
field can be swept.

Three things make the levels comparable:

- **The ramp is excluded from the measurement.** `run_time` is the measured
  window, and the time spent spawning users is added on top of it. At 512 users
  and a spawn rate of 16/s Locust is asked for 152 seconds so that 120 of them
  are at full concurrency. Statistics gathered during the ramp are discarded.
- **Each level keeps its time series**, so a level that never reached a plateau
  can be told apart from one that did.
- **A level with failed requests does not stop the sweep.** Its exit code is
  recorded in that level's `run_metadata.json` and the next level starts. The
  CLI exits non-zero only when the benchmark could not run at all, such as a
  failed health check.

The ladder above takes about 25 minutes: eleven levels of 120 measured seconds
plus roughly 130 seconds of ramps. A longer `run_time` or more levels scales
that up accordingly.

Re-running a batch writes into the same level directories, overwriting the
previous results in place. This is what makes `--resume` possible, but it also
means a level from an earlier, longer ladder is left behind rather than
cleared. Use a different `batch_name` to keep two batches side by side.

Use `--resume` to skip levels that already hold a completed result:

```bash
uv run python -m benchmark.locust benchmark/locust/configs/sweep_concurrency.yaml --resume
```

A level counts as complete only once its exit code has been recorded, so an
interrupted level is re-run rather than kept.

### CLI Options

```bash
uv run python -m benchmark.locust [OPTIONS] CONFIG_FILE
```

**Arguments:**
- `CONFIG_FILE`: Path to YAML configuration file (required)

**Options:**
- `--dry-run`: Print commands without executing them
- `--verbose`: Enable verbose logging and debugging information
- `--resume`: Skip sweep levels that already hold a completed result

## Configuration Options

All CLI configuration is done via YAML files. Unknown fields are rejected. The
following fields are supported:

### Required Fields

- `config_id`: Guardrails configuration ID to use
- `model`: Model name to send in requests

### Optional Fields

- `host`: Server base URL (default: `http://localhost:8000`)
- `target_users`: Concurrency to ramp to and then hold (default: `256`, minimum: `1`). Also accepted as `users`, the name used before sweeps existed.
- `spawn_rate`: Users spawned per second while ramping (default: `10`, minimum: `0.1`)
- `run_time`: Measured seconds at the target, excluding the ramp (default: `60`, minimum: `1`)
- `message`: Message content to send (default: `"Hello, what can you do?"`)
- `headless`: Run without web UI (default: `true`)
- `output_base_dir`: Directory for test results (default: `"locust_results"`)

Together these describe one level: users ramp from 0 to `target_users` at
`spawn_rate` per second, taking `ceil(target_users / spawn_rate)` seconds, and
the test then runs for `run_time` seconds with `target_users` active. Locust is asked for the sum of the
two, and `--reset-stats` discards everything gathered during the ramp, so the
reported numbers cover only the held portion.

`host` defaults to port `8000`, which in the quickstart stack is the mock
application LLM rather than Guardrails, so set it explicitly to the server you
intend to benchmark.

### Batch Fields

These fields describe a batch of runs rather than a single one. A config file
that omits them is read as a single run, so existing config files keep working.

- `batch_name`: Directory name for this batch of runs (default: `"benchmark"`)
- `output_base_dir`: Base directory for batch results (default: taken from the run config)
- `base_config`: The run configuration applied to every level
- `sweeps`: Mapping of a `base_config` field to the list of values to run

## Load Test Behavior

- **Request Type**: 100% POST `/v1/chat/completions` requests
- **Wait Time**: Zero wait time between requests (continuous hammering)
- **Ramp-up**: Users spawn gradually at the specified `spawn_rate`
- **Message Content**: Static message content (configurable via `message` field)

## Output

### Headless Mode

The quickstart command writes a single HTML report to the path given by `--html`:

```text
locust_results/
└── report.html  # HTML report with charts
```

The CLI instead saves results to timestamped directories under
`output_base_dir`, so successive runs do not overwrite each other:

```text
locust_results/
└── YYYYMMDD_HHMMSS/
    ├── report.html               # HTML report with charts
    ├── run_metadata.json         # Test configuration metadata
    ├── stats_stats.csv           # Request statistics
    ├── stats_stats_history.csv   # Statistics over time
    ├── stats_failures.csv        # Failure statistics
    └── stats_exceptions.csv      # Exceptions raised during the run
```

A sweep instead writes one directory per level, named after the swept value,
so a level can be found again and resumed:

```text
locust_results/
└── sweep_concurrency/
    ├── target_users-1/
    │   ├── report.html
    │   ├── run_metadata.json    # includes the level's Locust exit code
    │   ├── stats_stats.csv
    │   ├── stats_stats_history.csv
    │   ├── stats_failures.csv
    │   └── stats_exceptions.csv
    ├── target_users-2/
    └── ...
```

### Web UI Mode

Real-time metrics are displayed in the web interface at http://localhost:8089, including:
- Requests per second (RPS)
- Response time percentiles (50th, 95th, 99th)
- Failure rate
- Number of users

### Troubleshooting

If the load test does not start:

- Run `./scripts/validate_mocks.sh` from `benchmark/` and resolve any failed
  service checks before starting Locust.
- Run the Locust command from the repository root so the `locustfile.py` path
  resolves correctly.
- Confirm that `honcho` and `locust` are installed, per the quickstart above.
- Start with the documented single-user smoke test before increasing load.

If you see validation errors from the CLI:

- Ensure all required fields (`config_id`, `model`) are present in your YAML config
- Check that the `config_id` matches a configuration on your server
- Verify that numeric values meet minimum requirements (e.g., `users >= 1`, `spawn_rate >= 0.1`)
- Ensure `host` starts with `http://` or `https://`
- Remove any unrecognized fields, which are rejected rather than ignored
