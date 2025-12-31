# Guardrails Benchmarking

NeMo Guardrails includes benchmarking tools to help users capacity-test their Guardrails applications.
Adding guardrails to an LLM-based application improves safety and security, while adding some latency. These benchmarks allow users to quantify the tradeoff between security and latency, to make data-driven decisions.
We currently have a simple testbench, which runs the Guardrails server with mocks as Guardrail and Application models. This can be used for performance-testing on a laptop without any GPUs, and run in a few minutes.

-----

## Guardrails Core Benchmarking

This benchmark measures the performance of the Guardrails application, running on CPU-only laptop or instance.
It doesn't require GPUs on which to run local models, or access to the internet to use models hosted by providers.
All models use the [Mock LLM Server](mock_llm_server), which is a simplified model of an LLM used for inference.
The aim of this benchmark is to detect performance-regressions as quickly as running unit-tests.

## Quickstart: Running Guardrails with Mock LLMs
To run Guardrails with mocks for both the content-safety and main LLM, follow the steps below.
All commands must be run in the `nemoguardrails/benchmark` directory.
These assume you already have a working environment after following the steps in [CONTRIBUTING.md](../../CONTRIBUTING.md).

First, we need to install the honcho and langchain-nvidia-ai-endpoints packages.
The `honcho` package is used to run Procfile-based applications, and is a Python port of [Foreman](https://github.com/ddollar/foreman).
The `langchain-nvidia-ai-endpoints` package is used to communicate with Mock LLMs via Langchain.

```shell
# Install dependencies
$ poetry run pip install honcho langchain-nvidia-ai-endpoints
...
Successfully installed filetype-1.2.0 honcho-2.0.0 langchain-nvidia-ai-endpoints-0.3.19
```

Now we can start up the processes that are part of the [Procfile](Procfile).
As the Procfile processes spin up, they log to the console with a prefix. The `system` prefix is used by Honcho, `app_llm` is the Application or Main LLM mock, `cs_llm` is the content-safety mock, and `gr` is the Guardrails service. We'll explore the Procfile in more detail below.
Once the three 'Uvicorn running on ...' messages are printed, you can move to the next step. Note these messages are likely not on consecutive lines.

```
# All commands must be run in the nemoguardrails/benchmark directory
$ cd nemoguardrails/benchmark
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

Once Guardrails and the mock servers are up, we can use the `validate_mocks.py` script to check they're healthy and serving the correct models.

```shell
$ cd nemoguardrails/benchmark
$ poetry run python validate_mocks.py
Starting LLM endpoint health check...

--- Checking Port: 8000 ---
Checking http://localhost:8000/health ...
HTTP Request: GET http://localhost:8000/health "HTTP/1.1 200 OK"
Health Check PASSED: Status is 'healthy'.
Checking http://localhost:8000/v1/models for 'meta/llama-3.3-70b-instruct'...
HTTP Request: GET http://localhost:8000/v1/models "HTTP/1.1 200 OK"
Model Check PASSED: Found 'meta/llama-3.3-70b-instruct' in model list.
--- Port 8000: ALL CHECKS PASSED ---

--- Checking Port: 8001 ---
Checking http://localhost:8001/health ...
HTTP Request: GET http://localhost:8001/health "HTTP/1.1 200 OK"
Health Check PASSED: Status is 'healthy'.
Checking http://localhost:8001/v1/models for 'nvidia/llama-3.1-nemoguard-8b-content-safety'...
HTTP Request: GET http://localhost:8001/v1/models "HTTP/1.1 200 OK"
Model Check PASSED: Found 'nvidia/llama-3.1-nemoguard-8b-content-safety' in model list.
--- Port 8001: ALL CHECKS PASSED ---

--- Checking Port: 9000 (Rails Config) ---
Checking http://localhost:9000/v1/rails/configs ...
HTTP Request: GET http://localhost:9000/v1/rails/configs "HTTP/1.1 200 OK"
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

Once the mocks and Guardrails are running and the script passes, we can issue curl requests against the Guardrails `/chat/completions` endpoint to generate a response and test the system end-to-end.

```shell
curl -s -X POST http://0.0.0.0:9000/v1/chat/completions \
   -H 'Accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "model": "meta/llama-3.3-70b-instruct",
      "messages": [
         {
            "role": "user",
            "content": "what can you do for me?"
         }
      ],
      "stream": false
    }' | jq
{
  "messages": [
    {
      "role": "assistant",
      "content": "I can provide information and help with a wide range of topics, from science and history to entertainment and culture. I can also help with language-related tasks, such as translation and text summarization. However, I can't assist with requests that involve harm or illegal activities."
    }
  ]
}

```

------

## Deep-Dive: Configuration

In this section, we'll examine the configuration files used in the quickstart above. This gives more context on how the system works, and can be extended as needed.

### Procfile

The [Procfile](Procfile?raw=true) contains all the processes that make up the application.
The Honcho package reads in this file, starts all the processes, and combines their logs to the console
The `gr` line runs the Guardrails server on port 9000 and sets the default Guardrails configuration as [content_safety_colang1](configs/guardrail_configs/content_safety_colang1?raw=true).
The `app_llm` line runs the Application or Main Mock LLM. Guardrails calls this LLM to generate a response to the user's query. This server uses 4 uvicorn workers and runs on port 8000. The configuration file here is a Mock LLM configuration, not a Guardrails configuration.
The `cs_llm` line runs the Content-Safety Mock LLM. This uses 4 uvicorn workers and runs on port 8001.

### Guardrails Configuration
The [Guardrails Configuration](configs/guardrail_configs/content_safety_colang1/config.yml) is used by the Guardrails server.
Under the `models` section, the `main` model is used to generate responses to the user queries. The base URL for this model is the `app_llm` Mock LLM from the Procfile, running on port 8000. The `model` field has to match the Mock LLM model name.
The `content_safety` model is configured for use in an input and output rail. The `type` field matches the `$model` used in the input and output flows.

### Mock LLM Endpoints
The Mock LLM implements a subset of the OpenAI LLM API.
There are two Mock LLM configurations, one for the Mock [main model](configs/mock_configs/meta-llama-3.3-70b-instruct.env), and another for the Mock [content-safety](configs/mock_configs/nvidia-llama-3.1-nemoguard-8b-content-safety.env) model.
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
* If the sampled value is less than `LATENCY_MAX_SECONDS`, it is set to `LATENCY_MAX_SECONDS`.

The full list of configuration fields is shown below:
* `MODEL`: The Model name served by the Mock LLM. This will be returned on the `/v1/models` endpoint.
* `UNSAFE_PROBABILITY`: Probability of an unsafe response. This is a probability, and must be in the range [0, 1].
* `UNSAFE_TEXT`: String returned as an unsafe response.
* `SAFE_TEXT`: String returned as a safe response.
* `LATENCY_MIN_SECONDS`: Minimum latency in seconds.
* `LATENCY_MAX_SECONDS`: Maximum latency in seconds.
* `LATENCY_MEAN_SECONDS`: Normal distribution mean from which to sample latency.
* `LATENCY_STD_SECONDS`: Normal distribution standard deviation from which to sample latency.
