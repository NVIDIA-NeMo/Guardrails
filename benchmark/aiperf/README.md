# AIPerf Benchmarking for NeMo Guardrails

## Introduction

[AIPerf](https://github.com/ai-dynamo/aiperf) is NVIDIA's latest benchmarking tool for LLMs. It supports any OpenAI-compatible inference service and generates synthetic data loads, benchmarks, and all the metrics needed for performance comparison and analysis.

The [`run_aiperf.py`](run_aiperf.py) script enhances AIPerf's capabilities by providing:

- **Batch Execution**: Run multiple benchmarks in sequence with a single command
- **Parameter Sweeps**: Automatically generate and run benchmarks across different parameter combinations (e.g., sweeping concurrency levels, token counts, etc.)
- **Organized Results**: Automatically organizes benchmark results in timestamped directories with clear naming conventions
- **YAML Configuration**: Simple, declarative configuration files for reproducible benchmark runs
- **Run Metadata**: Saves complete metadata about each run (configuration, command, timestamp) for future analysis and reproduction
- **Service Health Checks**: Validates that the target service is available before starting benchmarks

Instead of manually running AIPerf multiple times with different parameters, you can define a sweep in a YAML file and let the script handle the rest.

## Getting Started

### Prerequisites

These steps have been tested with Python 3.11.11.
To use the provided configurations, you need to create accounts at https://build.nvidia.com/ and [Huggingface](https://huggingface.co/).
* The provided configurations use models hosted at https://build.nvidia.com/, you'll need to create a Personal API Key to access the models.
* The provided AIperf configurations require the [Meta Llama 3.3 70B Instruct tokenizer](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to calculate token-counts.

1. **Create a virtual environment in which to install AIPerf**

   ```bash
   $ mkdir ~/env
   $ python -m venv ~/env/aiperf
   ```

2. **Install dependencies in the virtual environment**

   ```bash
   $ pip install aiperf huggingface_hub typer
   ```

3. ** Login to Hugging Face:**

   ```bash
   huggingface-cli login
   ```

4. ** Set NVIDIA API Key:**

   The provided configs use models hosted on [build.nvidia.com](https://build.nvidia.com/).
   To access these, [create an account](https://build.nvidia.com/), and create a Personal API Key.
   After creating a Personal API key, set the `NVIDIA_API_KEY` variable as below.

   ```bash
   $ export NVIDIA_API_KEY="your-api-key-here"
   ```

## Running Benchmarks

Each benchmark is configured using the `AIPerfConfig` Pydantic model in [aiperf_models.py](aiperf_models.py).
The configs are stored in YAML files, and converted to an `AIPerfConfig` object.
There are two example configs included which can be extended for your use-cases. These both use Nvidia-hosted models, :

- [`single_concurrency.yaml`](aiperf_configs/single_concurrency.yaml): Example single-run benchmark with a single concurrency value.
- [`sweep_concurrency.yaml`](aiperf_configs/sweep_concurrency.yaml): Example multiple-run benchmark to sweep concurency values and run a new benchmark for each.

To run a benchmark, use the following command:

```bash
$ python -m benchmark.aiperf --config-file <path-to-config.yaml>
```

### Running a Single Benchmark

To run a single benchmark with fixed parameters, use the `single_concurrency.yaml` configuration:

```bash
$ python -m benchmark.aiperf --config-file aiperf/configs/single_concurrency.yaml
```

**Example output:**

```text
2025-12-01 10:35:17 INFO: Running AIPerf with configuration: aiperf/configs/single_concurrency.yaml
2025-12-01 10:35:17 INFO: Results root directory: aiperf_results/single_concurrency/20251201_103517
2025-12-01 10:35:17 INFO: Sweeping parameters: None
2025-12-01 10:35:17 INFO: Running AIPerf with configuration: aiperf/configs/single_concurrency.yaml
2025-12-01 10:35:17 INFO: Output directory: aiperf_results/single_concurrency/20251201_103517
2025-12-01 10:35:17 INFO: Single Run
2025-12-01 10:36:54 INFO: Run completed successfully
2025-12-01 10:36:54 INFO: SUMMARY
2025-12-01 10:36:54 INFO: Total runs : 1
2025-12-01 10:36:54 INFO: Completed  : 1
2025-12-01 10:36:54 INFO: Failed     : 0
```

### Running a Concurrency Sweep

To run multiple benchmarks with different concurrency levels, use the `sweep_concurrency.yaml` configuration as below:

```bash
$ python -m benchmark.aiperf --config-file aiperf/configs/sweep_concurrency.yaml
```

**Example output:**

```text
2025-11-14 14:02:54 INFO: Running AIPerf with configuration: nemoguardrails/benchmark/aiperf/aiperf_configs/sweep_concurrency.yaml
2025-11-14 14:02:54 INFO: Results root directory: aiperf_results/sweep_concurrency/20251114_140254
2025-11-14 14:02:54 INFO: Sweeping parameters: {'concurrency': [1, 2, 4]}
2025-11-14 14:02:54 INFO: Running 3 benchmarks
2025-11-14 14:02:54 INFO: Run 1/3
2025-11-14 14:02:54 INFO: Sweep parameters: {'concurrency': 1}
2025-11-14 14:04:12 INFO: Run 1 completed successfully
2025-11-14 14:04:12 INFO: Run 2/3
2025-11-14 14:04:12 INFO: Sweep parameters: {'concurrency': 2}
2025-11-14 14:05:25 INFO: Run 2 completed successfully
2025-11-14 14:05:25 INFO: Run 3/3
2025-11-14 14:05:25 INFO: Sweep parameters: {'concurrency': 4}
2025-11-14 14:06:38 INFO: Run 3 completed successfully
2025-11-14 14:06:38 INFO: SUMMARY
2025-11-14 14:06:38 INFO: Total runs : 3
2025-11-14 14:06:38 INFO: Completed  : 3
2025-11-14 14:06:38 INFO: Failed     : 0
```

## Additional Options

### AIPerf run options

The `--dry-run` option allows you to preview all benchmark commands without executing them. This is useful for:

- Validating your configuration file
- Checking which parameter combinations will be generated
- Estimating total execution time before committing to a long-running sweep
- Debugging configuration issues

```bash
$ python -m benchmark.aiperf --config-file aiperf/configs/sweep_concurrency.yaml --dry-run
```

When in dry-run mode, the script will:

- Load and validate your configuration
- Check service connectivity
- Generate all sweep combinations
- Display what would be executed
- Exit without running any benchmarks

### Verbose Mode

The `--verbose` option outputs more detailed debugging information to understand each step of the benchmarking process.

```bash
$ python -m benchmark.aiperf --config-file <config.yaml> --verbose
```

Verbose mode provides:

- Complete command-line arguments passed to AIPerf
- Detailed parameter merging logic (base config + sweep params)
- Output directory creation details
- Real-time AIPerf output (normally captured to files)
- Full stack traces for errors

**Tip:** Use verbose mode when debugging configuration issues or when you want to see live progress of the benchmark execution.

## Configuration Files

Configuration files are YAML files located in [aiperf_configs](aiperf_configs). The configuration is validated using Pydantic models to catch errors early.

### Top-Level Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `batch_name` | string | Yes | Name for this batch of benchmarks. Used in output directory naming (e.g., `aiperf_results/batch_name/timestamp/`) |
| `output_base_dir` | string | Yes | Base directory where all benchmark results will be stored |
| `base_config` | object | Yes | Base configuration parameters applied to all benchmark runs (see below) |
| `sweeps` | object | No | Optional parameter sweeps for running multiple benchmarks with different values |

### Base Configuration Parameters

The `base_config` section contains parameters that are passed to AIPerf. Any of these can be overridden by sweep parameters.

#### Model and Service Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., `meta/llama-3.3-70b-instruct`) |
| `tokenizer` | string | No | Tokenizer name from Hugging Face or local path. If not provided, AIPerf will attempt to use the model name |
| `url` | string | Yes | Base URL of the inference service (e.g., `https://integrate.api.nvidia.com`) |
| `endpoint` | string | No | API endpoint path (default: `/v1/chat/completions`) |
| `endpoint_type` | string | No | Type of endpoint: `chat` or `completions` (default: `chat`) |
| `api_key_env_var` | string | No | Name of environment variable containing API key (e.g., `NVIDIA_API_KEY`) |
| `streaming` | boolean | No | Whether to use streaming mode (default: `false`) |

#### Load Generation Settings

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `warmup_request_count` | integer | Yes | Number of warmup requests to send before starting the benchmark |
| `benchmark_duration` | integer | Yes | Duration of the benchmark in seconds |
| `concurrency` | integer | Yes | Number of concurrent requests to maintain during the benchmark |
| `request_rate` | float | No | Target request rate in requests/second. If not provided, calculated from concurrency |
| `request_rate_mode` | string | No | Distribution mode: `constant` or `poisson` (default: `constant`) |

#### Synthetic Data Generation

These parameters control the generation of synthetic prompts for benchmarking:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `random_seed` | integer | No | Random seed for reproducible synthetic data generation |
| `prompt_input_tokens_mean` | integer | No | Mean number of input tokens per prompt |
| `prompt_input_tokens_stddev` | integer | No | Standard deviation of input token count |
| `prompt_output_tokens_mean` | integer | No | Mean number of expected output tokens |
| `prompt_output_tokens_stddev` | integer | No | Standard deviation of output token count |

### Parameter Sweeps

The `sweeps` section allows you to run multiple benchmarks with different parameter values. The script generates a **Cartesian product** of all sweep values, running a separate benchmark for each combination.

#### Basic Sweep Example

```yaml
sweeps:
  concurrency: [1, 2, 4, 8, 16]
```

This will run 5 benchmarks, one for each concurrency level.

#### Multi-Parameter Sweep Example

```yaml
sweeps:
  concurrency: [1, 4, 16]
  prompt_input_tokens_mean: [100, 500, 1000]
```

This will run **9 benchmarks**, one for each value of `concurrency` and `prompt_input_tokens_mean`.

Each sweep combination creates a subdirectory named with the parameter values:

```text
aiperf_results/
└── my_benchmark/
    └── 20251114_140254/
        ├── concurrency1_prompt_input_tokens_mean100/
        ├── concurrency1_prompt_input_tokens_mean500/
        ├── concurrency4_prompt_input_tokens_mean100/
        └── ...
```

### Complete Configuration Example

```yaml
# Name for this batch of benchmarks
batch_name: my_benchmark

# Base directory where all benchmark results will be stored
output_base_dir: aiperf_results

# Base configuration applied to all benchmark runs
base_config:
  # Model and service configuration
  model: meta/llama-3.3-70b-instruct
  tokenizer: meta-llama/Llama-3.3-70B-Instruct
  url: "https://integrate.api.nvidia.com"
  endpoint: "/v1/chat/completions"
  endpoint_type: chat
  api_key_env_var: NVIDIA_API_KEY
  streaming: true

  # Load generation settings
  warmup_request_count: 20
  benchmark_duration: 60
  concurrency: 1
  request_rate_mode: "constant"

  # Synthetic data generation
  random_seed: 12345
  prompt_input_tokens_mean: 100
  prompt_input_tokens_stddev: 10
  prompt_output_tokens_mean: 50
  prompt_output_tokens_stddev: 5

# Optional: parameter sweeps (Cartesian product)
sweeps:
  concurrency: [1, 2, 4, 8, 16]
  prompt_input_tokens_mean: [100, 500, 1000]
```

### Common Sweep Patterns

#### Concurrency Scaling Test

```yaml
sweeps:
  concurrency: [1, 2, 4, 8, 16, 32, 64]
```

Useful for finding optimal concurrency levels and throughput limits.

#### Token Length Impact Test

```yaml
sweeps:
  prompt_input_tokens_mean: [50, 100, 500, 1000, 2000]
  prompt_output_tokens_mean: [50, 100, 500, 1000]
```

Useful for understanding how token counts affect latency and throughput.

#### Request Rate Comparison

```yaml
sweeps:
  request_rate_mode: ["constant", "poisson"]
  concurrency: [4, 8, 16]
```

Useful for comparing different load patterns.

## Output Structure

Results are organized in timestamped directories:

```text
aiperf_results/
├── <batch_name>/
│   └── <timestamp>/
│       ├── run_metadata.json          # Single run
│       ├── process_result.json
│       └── <aiperf_outputs>
│       # OR for sweeps:
│       ├── concurrency1/
│       │   ├── run_metadata.json
│       │   ├── process_result.json
│       │   └── <aiperf_outputs>
│       ├── concurrency2/
│       │   └── ...
│       └── concurrency4/
│           └── ...
```

### Output Files

Each run directory contains multiple files with benchmark results and metadata. A summary of these is shown below:

#### Benchmark runner files

- **`run_metadata.json`**: Contains complete metadata about the benchmark run for reproducibility.
- **`process_result.json`**: Contains the subprocess execution results.

#### Files Generated by AIPerf

- **`inputs.json`**: Synthetic prompt data generated for the benchmark.
- **`profile_export_aiperf.json`**: Main metrics file in JSON format containing aggregated statistics.
- **`profile_export_aiperf.csv`**: Same metrics as the JSON file, but in CSV format for easy import into spreadsheet tools or data analysis libraries.
- **`profile_export.jsonl`**: JSON Lines format file containing per-request metrics. Each line is a complete JSON object for one request with:
- **`logs/aiperf.log`**: Detailed log file from AIPerf execution containing:

## Resources

- [AIPerf GitHub Repository](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf)
- [AIPerf Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/README.html)
- [NVIDIA API Catalog](https://build.nvidia.com/)
