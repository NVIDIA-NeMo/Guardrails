# Locust Load Testing for NeMo Guardrails

This directory contains a Locust-based load testing framework for the NeMo Guardrails OpenAI-compatible server.

## Introduction

The [Locust](https://locust.io/) stress-testing tool ramps up concurrent users making API calls to the `/v1/chat/completions` endpoint of an OpenAI-compatible LLM with configurable parameters.
This complements [ai-perf](https://github.com/ai-dynamo/aiperf), which measures steady-state performance.  Locust instead focuses on ramping up load potentially beyond what a system can handle, and measure how gracefully it degrades under higher-than-expected load.

## Getting Started

### Prerequisites

These steps have been tested with Python 3.11.11.

1. **Create a virtual environment in which to install Locust and other benchmarking tools**

   ```bash
   $ mkdir ~/env
   $ python -m venv ~/env/benchmark_env
   ```

2. **Install dependencies in the virtual environment**

   ```bash
   $ pip install -r benchmark/requirements.txt
   ```

## Running Benchmarks

### Run with YAML Configuration (Required)

Create a YAML configuration file (see `benchmark/locust/configs/local.yaml` for a complete example):

```yaml
host: "http://localhost:8000"
config_id: "my-guardrails-config"
model: "mock-llm"
users: 256
spawn_rate: 10
run_time: 60
message: "Hello, what can you do?"
headless: true
output_base_dir: "locust_results"
```

Then run:

```bash
python -m benchmark.locust benchmark/locust/configs/local.yaml
```

### Web UI Mode

Set `headless: false` in your YAML config to use Locust's interactive web UI:

```yaml
host: "http://localhost:8000"
config_id: "my-config"
model: "mock-llm"
users: 100
spawn_rate: 10
headless: false
```

```bash
python -m benchmark.locust my-config.yaml
```

Then open http://localhost:8089 to control the test and view real-time metrics.

### CLI Options

The `benchmark.locust` CLI supports the following options:

```bash
python -m benchmark.locust [OPTIONS] CONFIG_FILE
```

**Arguments:**
- `CONFIG_FILE`: Path to YAML configuration file (required)

**Options:**
- `--dry-run`: Print commands without executing them
- `--verbose`: Enable verbose logging and debugging information

## Configuration Options

All configuration is done via YAML files. The following fields are supported:

### Required Fields

- `config_id`: Guardrails configuration ID to use
- `model`: Model name to send in requests

### Optional Fields

- `host`: Server base URL (default: `http://localhost:8000`)
- `users`: Maximum concurrent users (default: `256`, minimum: `1`)
- `spawn_rate`: Users spawned per second (default: `10`, minimum: `0.1`)
- `run_time`: Test duration in seconds (default: `60`, minimum: `1`, or `null` for unlimited)
- `message`: Message content to send (default: `"Hello, what can you do?"`)
- `headless`: Run without web UI (default: `true`)
- `output_base_dir`: Directory for test results (default: `"locust_results"`)

## Load Test Behavior

- **Request Type**: 100% POST `/v1/chat/completions` requests
- **Wait Time**: Zero wait time between requests (continuous hammering)
- **Ramp-up**: Users spawn gradually at the specified `spawn_rate`
- **Message Content**: Static message content (configurable via `message` field)

## Output

### Headless Mode

When run in headless mode, results are saved to timestamped directories:

```
locust_results/
└── YYYYMMDD_HHMMSS/
    ├── report.html          # HTML report with charts
    ├── run_metadata.json    # Test configuration metadata
    ├── stats.csv            # Request statistics
    ├── stats_failures.csv   # Failure statistics
    └── stats_history.csv    # Statistics over time
```

### Web UI Mode

Real-time metrics are displayed in the web interface at http://localhost:8089, including:
- Requests per second (RPS)
- Response time percentiles (50th, 95th, 99th)
- Failure rate
- Number of users

### Troubleshooting

If you see validation errors:
- Ensure all required fields (`config_id`, `model`) are present in your YAML config
- Check that the `config_id` matches a configuration on your server
- Verify that numeric values meet minimum requirements (e.g., `users >= 1`, `spawn_rate >= 0.1`)
- Ensure `host` starts with `http://` or `https://`
