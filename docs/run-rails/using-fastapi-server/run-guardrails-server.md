---
title:
  page: "Run the Guardrails Server"
  nav: "Run the Server"
description: "Start the Guardrails API server, configure CORS, and enable auto-reload."
keywords: ["NeMo Guardrails server", "nemoguardrails server", "Guardrails API", "CORS configuration", "auto-reload"]
topics: ["generative_ai", "developer_tools"]
tags: ["llms", "ai_inference", "ai_platforms"]
content:
  type: tutorial
  difficulty: technical_intermediate
  audience: ["data_scientist", "engineer"]
---

# Run the NeMo Guardrails Server

The NeMo Guardrails server loads configurations from a local directory at startup and exposes an HTTP API to use them.
The server uses [FastAPI](https://fastapi.tiangolo.com/) and includes a built-in Chat UI for testing.

## Start the Server

Launch the server using the `nemoguardrails` CLI:

```bash
nemoguardrails server --config examples/configs
```

For more information about the available options, see [](../../reference/cli/index.md#server).

## Link Guardrail Configurations to the Server

The server supports two modes depending on your folder structure: **multi-config mode** and **single-config mode**.

### Multi-Config Mode

When the `--config` path points to a folder containing multiple sub-folders, each sub-folder with a `config.yml` file becomes an available configuration.
The sub-folder name becomes the `config_id`.

```text
examples/configs/          # --config points here
├── content_safety/        # config_id: "content_safety"
│   ├── rails.co
│   └── config.yml
├── jailbreak_detection/   # config_id: "jailbreak_detection"
│   ├── flows.co
│   └── config.yml
└── topic_safety/          # config_id: "topic_safety"
    └── config.yml
```

1. Start the server in multi-config mode:

    ```bash
    nemoguardrails server --config examples/configs
    ```

1. List available configurations.

    ```bash
    curl http://localhost:8000/v1/rails/configs
    ```

    The endpoint returns the list of available configurations.

    ```json
    [
      {"id": "content_safety"},
      {"id": "jailbreak_detection"},
      {"id": "topic_safety"}
    ]
    ```

### Single-Config Mode

When the `--config` path points directly to a folder containing a `config.yml` file, the server runs in single-config mode.
The folder name becomes the only available `config_id`.

```text
examples/configs/content_safety/   # --config points here
├── rails.co
└── config.yml                     # config_id: "content_safety"
```

1. Start the server in single-config mode:

    ```bash
    nemoguardrails server --config examples/configs/content_safety
    ```

1. List available configurations.

    ```bash
    curl http://localhost:8000/v1/rails/configs
    ```

    The endpoint returns the single configuration named `content_safety`.

## Examples

The following examples show how to start the server with different options.

### Start with Default Settings

The following command starts the server with default settings.

```bash
nemoguardrails server
```

The server starts on port 8000 and looks for a `./config` folder in the current directory. If not found, it uses the built-in example configurations.

### Start with Custom Port

You can use the `--port` flag to start the server on a custom port.

```bash
nemoguardrails server --config examples/configs --port 8080
```

### Start with a Default Configuration

Use the following command to start the server with a default configuration within a multi-config folder. For example, when you use the [provided example configurations (`examples/configs`)](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs), you can set the default configuration to `content_safety` as follows.

```bash
nemoguardrails server --config examples/configs --default-config-id content_safety
```

Chat completions requests without a `config_id` use the `content_safety` configuration by default.

### Start in Development Mode

You can add the `--auto-reload` flag to the server to automatically reload when configuration files change.

```bash
nemoguardrails server --config ./configs --auto-reload
```

```{important}
Use `--auto-reload` only in development environments. It is not recommended for production.
```

## CORS Configuration

To enable your guardrails server to receive requests from browser-based applications, configure CORS using environment variables:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Environment Variable
  - Description

* - `NEMO_GUARDRAILS_SERVER_ENABLE_CORS`
  - Set to `true` to enable CORS. Default: `false`.

* - `NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS`
  - Comma-separated list of allowed origins. Default: `*`.
```

Example:

```bash
export NEMO_GUARDRAILS_SERVER_ENABLE_CORS=true
export NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS=http://localhost:3000,https://myapp.com
nemoguardrails server --config ./configs
```

## Chat UI

The server includes a built-in Chat UI for testing guardrails configurations.
Access it at `http://localhost:8000/` after starting the server.

```{important}
The Chat UI is for internal testing only.
For production deployments, disable it using the `--disable-chat-ui` flag.
```

## Related Topics

- [](chat-with-guardrailed-model.md)
- [](list-guardrail-configs.md)
- [](../../reference/api-server-endpoints/index.md)
