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

# Run the Guardrails Server

The Guardrails server loads a predefined set of guardrails configurations at startup and exposes an HTTP API to use them.
The server uses [FastAPI](https://fastapi.tiangolo.com/) and includes a built-in Chat UI for testing.

## Start the Server

Launch the server using the CLI:

```bash
nemoguardrails server \
  [--config PATH/TO/CONFIGS] \
  [--port PORT] \
  [--prefix PREFIX] \
  [--disable-chat-ui] \
  [--auto-reload] \
  [--default-config-id DEFAULT_CONFIG_ID]
```

### Command Options

```{list-table}
:header-rows: 1
:widths: 25 75

* - Option
  - Description

* - `--config`
  - Path to the folder containing guardrails configurations.
    If not specified, the server looks for a `config` folder in the current directory.

* - `--port`
  - Port number for the server. Default: `8000`.

* - `--prefix`
  - URL prefix for all server endpoints.
    For example, `--prefix /api` makes endpoints available at `/api/v1/chat/completions`.

* - `--disable-chat-ui`
  - Disable the built-in Chat UI. Recommended for production deployments.

* - `--auto-reload`
  - Automatically reload configurations when files change.
    Use only in development environments.

* - `--default-config-id`
  - Default configuration ID to use when none is specified in the request.
```

## Configuration Folder Structure

The server can load multiple guardrails configurations.
The configuration path must be a folder with sub-folders for each individual configuration:

```text
.
├── config
│   ├── config_1
│   │   ├── file_1.co
│   │   └── config.yml
│   ├── config_2
│   │   ├── ...
│   │   └── config.yml
│   ...
```

```{note}
If the server is pointed to a folder with a single `config.yml` file, only that configuration is available.
```

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

- [Chat with Guardrailed Model](chat-with-guardrailed-model.md)
- [List Guardrail Configurations](list-guardrail-configs.md)
- [Server Endpoints Reference](../../reference/api-server-endpoints/index.md)
