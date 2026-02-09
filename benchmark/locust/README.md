# Locust Load Testing for NeMo Guardrails

This directory contains a Locust-based load testing framework for the NeMo Guardrails OpenAI-compatible server.

## Overview

The load tester simulates concurrent users making API calls to the `/v1/chat/completions` endpoint with configurable parameters. It's designed to test the performance of guardrails processing including input rails, generation, and output rails.

## Installation

Install the required dependencies:

```bash
pip install locust pyyaml httpx
```

Or install from the benchmark requirements:

```bash
pip install -r benchmark/requirements.txt
```

## Usage

### Option 1: Run with YAML Configuration (Recommended)

Create a YAML configuration file (see `configs/example.yaml`):

```yaml
batch_name: my_load_test
output_base_dir: locust_results

base_config:
  host: "http://localhost:8000"
  config_id: "my-config"
  model: "mock-llm"
  users: 256
  spawn_rate: 10
  run_time: 60
  message: "Hello, what can you do?"
  headless: true
```

Then run:

```bash
python -m benchmark.locust run --config-file benchmark/locust/configs/example.yaml
```

### Option 2: Run with CLI Arguments

```bash
python -m benchmark.locust run \
  --host http://localhost:8000 \
  --config-id my-config \
  --model mock-llm \
  --users 100 \
  --spawn-rate 10 \
  --run-time 60 \
  --headless
```

### Option 3: Web UI Mode

Run without `--headless` flag to use Locust's interactive web UI:

```bash
python -m benchmark.locust run \
  --config-id my-config \
  --model mock-llm \
  --users 100 \
  --spawn-rate 10
```

Then open http://localhost:8089 to control the test and view real-time metrics.

### Option 4: Direct Locust Invocation

You can also run the locustfile directly:

```bash
export LOCUST_CONFIG_ID=my-config
export LOCUST_MODEL=mock-llm
export LOCUST_MESSAGE="Hello, what can you do?"

locust -f benchmark/locust/locustfile.py \
  --host http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 60s \
  --headless
```

## Configuration Options

### Required Parameters

- `config_id`: Guardrails configuration ID to use
- `model`: Model name to send in requests

### Optional Parameters

- `host`: Server base URL (default: `http://localhost:8000`)
- `users`: Maximum concurrent users (default: `256`, max: `256`)
- `spawn_rate`: Users spawned per second (default: `10`)
- `run_time`: Test duration in seconds (default: `60`, `0` for unlimited)
- `message`: Message content to send (default: `"Hello, what can you do?"`)
- `headless`: Run without web UI (default: `false`)
- `verbose`: Enable verbose logging (default: `false`)

## Load Test Behavior

- **Request Distribution**: 99% chat completions, 1% models endpoint (currently only chat completions)
- **Wait Time**: Zero wait time between requests (continuous hammering)
- **Ramp-up**: Users spawn gradually at the specified `spawn_rate`
- **Message Content**: Static message content (configurable)

## Output

### Headless Mode

When run in headless mode, results are saved to timestamped directories:

```
locust_results/
└── YYYYMMDD_HHMMSS/
    ├── report.html          # HTML report with charts
    ├── stats.csv            # Request statistics
    ├── stats_history.csv    # Statistics over time
    ├── stats_failures.csv   # Failure statistics
    └── run_metadata.json    # Test configuration metadata
```

### Web UI Mode

Real-time metrics are displayed in the web interface at http://localhost:8089, including:
- Requests per second (RPS)
- Response time percentiles (50th, 95th, 99th)
- Failure rate
- Number of users

## Testing with Mock LLM Server

The load tests are designed to work with the Mock LLM server in `benchmark/mock_llm_server`. The mock server returns stock responses without actual LLM calls, allowing you to test guardrails processing performance.

1. Start the Mock LLM server (see `benchmark/mock_llm_server/README.md`)
2. Start the NeMo Guardrails server with a config that uses the mock LLM
3. Run the load test pointing to the Guardrails server

## Examples

### Quick Test (10 users, 30 seconds)

```bash
python -m benchmark.locust run \
  --config-id test-config \
  --model mock-llm \
  --users 10 \
  --spawn-rate 2 \
  --run-time 30 \
  --headless
```

### Full Load Test (256 users, 5 minutes)

```bash
python -m benchmark.locust run \
  --config-id production-config \
  --model mock-llm \
  --users 256 \
  --spawn-rate 10 \
  --run-time 300 \
  --headless
```

### Interactive Testing

```bash
python -m benchmark.locust run \
  --config-id my-config \
  --model mock-llm
```

Then navigate to http://localhost:8089 to start and control the test interactively.

## Troubleshooting

### Server Connection Issues

If you see "Can't connect to http://localhost:8000":
- Ensure the NeMo Guardrails server is running
- Verify the host URL is correct
- Check that the config_id exists on the server

### Import Errors

If you see import errors for `locust`:
```bash
pip install locust
```

### Configuration Errors

If you see "config_id is required":
- Ensure you've provided `--config-id` via CLI or in the YAML config
- Check that the config_id matches a configuration on your server

## Architecture

The load tester consists of:

- `locust_models.py`: Pydantic models for configuration validation
- `locustfile.py`: Locust user behavior definition (can be run standalone)
- `run_locust.py`: Typer CLI wrapper with YAML support
- `__main__.py`: Module entry point

This follows the same pattern as `benchmark/aiperf` for consistency.
