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

### Run with YAML Configuration (Required)

Create a YAML configuration file (see `configs/local.yaml` for a complete example):

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
python -m benchmark.locust my-config.yaml
```

Or use the provided example:

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

### Direct Locust Invocation

You can also run the locustfile directly with Locust CLI:

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

Create `quick-test.yaml`:
```yaml
host: "http://localhost:8000"
config_id: "test-config"
model: "mock-llm"
users: 10
spawn_rate: 2
run_time: 30
headless: true
```

Run:
```bash
python -m benchmark.locust quick-test.yaml
```

### Full Load Test (256 users, 5 minutes)

Create `full-load-test.yaml`:
```yaml
host: "http://localhost:8000"
config_id: "production-config"
model: "mock-llm"
users: 256
spawn_rate: 10
run_time: 300
headless: true
```

Run:
```bash
python -m benchmark.locust full-load-test.yaml
```

### Interactive Testing

Create `interactive-test.yaml`:
```yaml
host: "http://localhost:8000"
config_id: "my-config"
model: "mock-llm"
headless: false
```

Run:
```bash
python -m benchmark.locust interactive-test.yaml
```

Then navigate to http://localhost:8089 to start and control the test interactively.

## Troubleshooting

### Server Connection Issues

If you see "Can't connect to http://localhost:8000":
- Ensure the NeMo Guardrails server is running
- Verify the `host` URL is correct in your YAML config
- Check that the `config_id` exists on the server

### Import Errors

If you see import errors for `locust`:
```bash
pip install locust
```

### Configuration Errors

If you see validation errors:
- Ensure all required fields (`config_id`, `model`) are present in your YAML config
- Check that the `config_id` matches a configuration on your server
- Verify that numeric values meet minimum requirements (e.g., `users >= 1`, `spawn_rate >= 0.1`)
- Ensure `host` starts with `http://` or `https://`

## Architecture

The load tester consists of:

- **`locust_models.py`**: Pydantic `LocustConfig` model for YAML configuration validation with field validators for host, users, spawn_rate, and run_time
- **`locustfile.py`**: Locust `GuardrailsUser` class defining load test behavior (can be run standalone with environment variables)
- **`run_locust.py`**: Typer CLI application (`LocustRunner` class) that loads YAML configs, validates server connectivity, builds Locust commands, and manages test execution
- **`__main__.py`**: Module entry point for `python -m benchmark.locust`

This follows the same YAML-first configuration pattern as `benchmark/aiperf` for consistency.

### Key Design Choices

- **YAML-first**: All configuration through YAML files rather than CLI flags for reproducibility
- **Pydantic validation**: Configuration validated at load time with clear error messages
- **Environment variables**: The locustfile receives configuration via environment variables, allowing standalone execution
- **Service health check**: Validates server connectivity before starting load tests
- **Timestamped results**: Headless mode saves results to timestamped directories with metadata
