# Guardrails Benchmarking

NeMo Guardrails includes benchmarking tools to help users capacity-test their Guardrails applications.
Adding guardrails to an LLM-based application improves safety and security, while adding some latency.
These benchmarks allow users to quantify the tradeoff between security and latency, to make data-driven decisions.
We currently have a simple testbench, which runs the Guardrails server and mock LLMs for Content-Safety Guardrail and Application models.
This can be used for performance-testing on a laptop without any GPUs, and run in a few minutes.

-----

## Quickstart: Running Guardrails with Mock LLMs

This benchmark measures the performance of the Guardrails application, running on CPU-only laptop or instance.
It doesn't require GPUs on which to run local models, or access to the internet to use models hosted by providers.
All models use the [Mock LLM Server](mock_llm_server), which is a simplified model of an LLM used for inference.
The aim of this benchmark is to detect performance-regressions as quickly as running unit-tests.

To run benchmarks, two terminals or tmux panes are needed, one for the server-side and client-side processes.
The first shell runs the Guardrails OpenAI-compatible server and Mock LLMs for a content-safety and application LLM.
This uses the same poetry environment used to develop code, with extra packages added using `pip install`.
The second shell uses [AIPerf](https://github.com/ai-dynamo/aiperf) to issue client requests and measure latency.

### 1. Run Server-side components: Guardrails OpenAI-compatible service with Mock LLMs for Content-Safety and Application LLMs

You'll first increase the file descriptor limit to 65,536.
This is needed because otherwise the Operating System will limit the number of open file descriptors and restrict the concurrency to be benchmarked.

```shell
$ ulimit -n 65536
````

Next you'll install the NeMo Guardrails Poetry environment and honcho to run server-side components.
The honcho package is used to read the [`Procfile`](Procfile) and bring up the OpenAI service and Mock LLMs for benchmarking.

```shell
$ poetry install --with dev -E "server"
$ poetry run pip install honcho
```

Now all the dependencies are installed in the poetry environment, you'll use honcho to run the Procfile.
This runs the Guardrails server and Mock LLMs for the Application LLM and Content-Safety.
As the Procfile processes spin up, they log to the console with a prefix. The `system` prefix is used by Honcho, `app_llm` is the Application or Main LLM mock, `cs_llm` is the content-safety mock, and `gr` is the Guardrails service.
We'll explore the Procfile in more detail below.
Once the three 'Uvicorn running on ...' messages are printed, you can move to the next step.
Note these messages are likely not on consecutive lines.

```shell
$ cd benchmark
$ poetry run honcho start
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

### 2. Validate services are running correctly

Once Guardrails and the mock servers are up, we'll use the [validate_mocks.sh](scripts/validate_mocks.sh) script to validate everything is working.
This script checks all Mock LLMs are running by checking the models they're serving and the Guardrails service.

```shell
# In a new shell, change into the benchmark directory and run these commands.

$ cd benchmark
$ scripts/validate_mocks.sh
Starting LLM endpoint health check...

--- Checking Port: 8000 ---
Checking http://localhost:8000/health ...
Health Check PASSED: Status is 'healthy'.
Checking http://localhost:8000/v1/models for 'meta/llama-3.3-70b-instruct'...
Model Check PASSED: Found 'meta/llama-3.3-70b-instruct' in model list.
--- Port 8000: ALL CHECKS PASSED ---

--- Checking Port: 8001 ---
Checking http://localhost:8001/health ...
Health Check PASSED: Status is 'healthy'.
Checking http://localhost:8001/v1/models for 'nvidia/llama-3.1-nemoguard-8b-content-safety'...
Model Check PASSED: Found 'nvidia/llama-3.1-nemoguard-8b-content-safety' in model list.
--- Port 8001: ALL CHECKS PASSED ---

--- Checking Port: 9000 (Rails Config) ---
Checking http://localhost:9000/v1/rails/configs ...
HTTP Status PASSED: Got 200.
Body Check PASSED: Response is an array with at least one entry.
--- Port 9000: ALL CHECKS PASSED ---

--- Final Summary ---
Port 8000 (meta/llama-3.3-70b-instruct): PASSED
Port 8001 (nvidia/llama-3.1-nemoguard-8b-content-safety): PASSED
Port 9000 (Rails Config): PASSED
---------------------
Overall Status: All endpoints are healthy!
```


### 3. Run client-side benchmarking: AIPerf

Now in a second terminal you'll increase the file descriptor limit as above.

```shell
$ ulimit -n 65536
````

Next you'll create a virtual environment for AIPerf and install it.
Run these and the following commands in the repository root (one level up from `benchmark`).

```shell

$ mkdir ~/env
$ python -m venv ~/env/aiperf_env
$ source ~/env/aiperf_env/bin/activate
(aiperf_env) $ pip install aiperf

...
Successfully installed Flask-3.1.3 MarkupSafe-3.0.3 Werkzeug-3.1.8 ..... yarl-1.23.0 zipp-4.1.0 zstandard-0.25.0
(aiperf_env) $
```

To run an AIPerf benchmark with the `sweep_concurrency_benchmark.yaml` configuration, use the command below.
This makes requests against the Guardrails service created above in step 2, which in turn makes requests to the Mock LLMs for Application and Content-Safety LLMs.
The benchmark sweeps concurrency from 1 to 256 in powers-of-2 steps, with synthetic user-prompts.
Once the benchmark completes, the results can be found in the `aiperf_results` directory.

```shell
(aiperf_env) $ python -m benchmark.aiperf --config-file benchmark/aiperf/configs/sweep_concurrency_benchmark.yaml

2026-05-19 14:19:47 INFO: Running AIPerf with configuration: benchmark/aiperf/configs/sweep_concurrency_benchmark.yaml
2026-05-19 14:19:47 INFO: Results root directory: aiperf_results/sweep_concurrency_benchmark/20260519_141947
2026-05-19 14:19:47 INFO: Sweeping parameters: {'concurrency': [1, 2, 4, 8, 16, 32, 64, 128, 256]}
2026-05-19 14:19:47 INFO: Running 9 benchmarks
2026-05-19 14:19:47 INFO: Run 1/9
2026-05-19 14:19:47 INFO: Sweep parameters: {'concurrency': 1}
2026-05-19 14:21:45 INFO: Run 1 completed successfully
2026-05-19 14:21:45 INFO: Run 2/9
2026-05-19 14:21:45 INFO: Sweep parameters: {'concurrency': 2}
2026-05-19 14:23:19 INFO: Run 2 completed successfully
.....
.....
2026-05-19 14:29:58 INFO: Run 8/9
2026-05-19 14:29:58 INFO: Sweep parameters: {'concurrency': 128}
2026-05-19 14:31:17 INFO: Run 8 completed successfully
2026-05-19 14:31:17 INFO: Run 9/9
2026-05-19 14:31:17 INFO: Sweep parameters: {'concurrency': 256}
2026-05-19 14:32:37 INFO: Run 9 completed successfully
2026-05-19 14:32:37 INFO: SUMMARY
2026-05-19 14:32:37 INFO: Total runs : 9
2026-05-19 14:32:37 INFO: Completed  : 9
2026-05-19 14:32:37 INFO: Failed     : 0
```

------

## Deep-Dive: Configuration

In this section, we'll examine the configuration files used in the quickstart above. This gives more context on how the system works, and can be extended as needed.

### Procfile

The [Procfile](Procfile) contains all the processes that make up the application.
The Honcho package reads in this file, starts all the processes, and combines their logs to the console
The `gr` line runs the Guardrails server on port 9000 and sets the default Guardrails configuration as [content_safety_local](../examples/configs/content_safety_local).
The `app_llm` line runs the Application or Main Mock LLM. Guardrails calls this LLM to generate a response to the user's query. This server uses 4 uvicorn workers and runs on port 8000. The configuration file here is a Mock LLM configuration, not a Guardrails configuration.
The `cs_llm` line runs the Content-Safety Mock LLM. This uses 4 uvicorn workers and runs on port 8001.

### Guardrails Configuration

The [Guardrails Configuration](../examples/configs/content_safety_local/config.yml) is used by the Guardrails server.
Under the `models` section, the `main` model is used to generate responses to the user queries. The base URL for this model is the `app_llm` Mock LLM from the Procfile, running on port 8000. The `model` field has to match the Mock LLM model name.
The `content_safety` model is configured for use in an input and output rail. The `type` field matches the `$model` used in the input and output flows.

### Mock LLM Endpoints

The Mock LLM implements a subset of the OpenAI LLM API.
There are two Mock LLM configurations, one for the Mock [main model](mock_llm_server/configs/meta-llama-3.3-70b-instruct.env), and another for the Mock [content-safety](mock_llm_server/configs/nvidia-llama-3.1-nemoguard-8b-content-safety.env) model.
The Mock LLM has the following OpenAI-compatible endpoints:

* `/health`: Returns a JSON object with status set to healthy and timestamp in seconds-since-epoch. For example `{"status":"healthy","timestamp":1762781239}`
* `/v1/models`: Returns the `MODEL` field from the Mock configuration (see below). For example `{"object":"list","data":[{"id":"meta/llama-3.3-70b-instruct","object":"model","created":1762781290,"owned_by":"system"}]}`
* `/v1/completions`: Returns an [OpenAI completion object](https://platform.openai.com/docs/api-reference/completions/object) using the Mock configuration (see below).
* `/v1/chat/completions`: Returns an [OpenAI chat completion object](https://platform.openai.com/docs/api-reference/chat/object) using the Mock configuration (see below).

### Mock LLM Configuration

Mock LLMs are configured using the `.env` file format. These files are passed to the Mock LLM using the `--config-file` argument.
The Mock LLMs return either a `SAFE_TEXT` or `UNSAFE_TEXT` response to `/v1/completions` or `/v1/chat/completions` inference requests.
The probability of the `UNSAFE_TEXT` being returned if given by `UNSAFE_PROBABILITY`.
The latency of each response is also controllable, and works as follows:

* Latency is first sampled from a normal distribution with mean `LATENCY_MEAN_SECONDS` and standard deviation `LATENCY_STD_SECONDS`.
* If the sampled value is less than `LATENCY_MIN_SECONDS`, it is set to `LATENCY_MIN_SECONDS`.
* If the sampled value is greater than `LATENCY_MAX_SECONDS`, it is set to `LATENCY_MAX_SECONDS`.

The full list of configuration fields is shown below:

* `MODEL`: The Model name served by the Mock LLM. This will be returned on the `/v1/models` endpoint.
* `UNSAFE_PROBABILITY`: Probability of an unsafe response. This must be in the range [0, 1].
* `UNSAFE_TEXT`: String returned as an unsafe response.
* `SAFE_TEXT`: String returned as a safe response.
* `LATENCY_MIN_SECONDS`: Minimum latency in seconds.
* `LATENCY_MAX_SECONDS`: Maximum latency in seconds.
* `LATENCY_MEAN_SECONDS`: Normal distribution mean from which to sample latency.
* `LATENCY_STD_SECONDS`: Normal distribution standard deviation from which to sample latency.
