---
title:
  page: NeMo Guardrails Library CLI Reference
  nav: CLI
description: Use the nemoguardrails command-line interface to chat, serve, evaluate, and convert configurations.
topics:
- AI Safety
- LLM Guardrails
tags:
- CLI
- Python
- Server
- Evaluation
- Colang
content:
  type: reference
  difficulty: technical_intermediate
  audience:
  - engineer
  - AI Engineer
---

(nemoguardrails-cli)=

# NeMo Guardrails Library CLI Reference

You should now be able to invoke the `nemoguardrails` CLI.

```bash
> nemoguardrails --help

Usage: nemoguardrails [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --version         Show version and exit.
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it
                        or customize the installation.
  --help                Show this message and exit.

Commands:
  chat             Start an interactive chat session.
  server           Start a NeMo Guardrails server.
  convert          Convert Colang files and configs from older version to
                   the latest.
  actions-server   Start a NeMo Guardrails actions server.
  find-providers   List and select LLM providers interactively.
  eval             Evaluation a guardrail configuration.
```

You can also use the `--help` flag to learn more about each of the `nemoguardrails` commands.

---

## `chat`

```sh
nemoguardrails chat --config examples/configs/content_safety [--verbose] [--streaming]
```

### Options

| Option | Description |
|--------|-------------|
| `--config` | Path to a directory containing configuration files. Can also point to a single configuration file. Default: `config`. |
| `--verbose` | Enable verbose mode with detailed logging information. |
| `--verbose-no-llm` | Enable verbose mode but exclude LLM prompts and responses. |
| `--verbose-simplify` | Simplify the verbose output. |
| `--debug-level` | Enable debug mode with rich flow execution info. Levels: `WARNING`, `INFO`, `DEBUG`. |
| `--streaming` | Enable streaming mode if the configuration supports it. |
| `--server-url` | Connect to a server instead of loading config locally. Requires `--config-id`. |
| `--config-id` | The config ID to use when connecting to a server. |

## `actions-server`

Start a separate server for running custom actions in an isolated environment.

```bash
nemoguardrails actions-server --port 8001
```

### Options

| Option | Description |
|--------|-------------|
| `--port` | The port that the server should listen on. Default: `8001`. |

---

## `server`

Start the FastAPI server for serving guardrails configurations over HTTP.

```bash
nemoguardrails server --config examples/configs --port 8000
```

### Options

| Option | Description |
|--------|-------------|
| `--port` | The port that the server should listen on. Default: `8000`. |
| `--config` | Path to a directory containing multiple configuration sub-folders. |
| `--default-config-id` | The default configuration to use when no config is specified in requests. |
| `--verbose` | Enable verbose mode with detailed logs including prompts. |
| `--disable-chat-ui` | Disable the Chat UI served at the root path. |
| `--auto-reload` | Enable auto reload when configuration files change. |
| `--prefix` | A prefix to add to all server paths. Must start with `/`. |

---

## `eval`

Evaluate guardrail configurations with various testing and compliance workflows.

```bash
nemoguardrails eval run --config examples/configs/content_safety
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `run` | Run the interactions for an evaluation. |
| `check-compliance` | Check the policy compliance of the interactions in the output path. |
| `ui` | Launch the Evaluation UI. |
| `rail` | Run a rail evaluation task. |

Use `nemoguardrails eval <subcommand> --help` for details on each subcommand.

---

## `convert`

Convert Colang files and configurations from older versions to the latest format.

```bash
nemoguardrails convert ./my-config --from-version 1.0 --validate
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | The path to the file or directory to migrate. Required. |

### Options

| Option | Description |
|--------|-------------|
| `--from-version` | The version of the Colang files to migrate from. Options: `1.0`, `2.0-alpha`. Default: `1.0`. |
| `--verbose` | Enable verbose mode with detailed logs. |
| `--validate` | Validate the output using the Colang Parser. |
| `--use-active-decorator` | Use the active decorator in the migration. Default: enabled. |
| `--include-main-flow` | Add a main flow to the config. Default: enabled. |

---

(find-providers-command)=

## `find-providers`

List and select LLM providers interactively. This command helps you discover available providers for text completion and chat completion models.

```bash
nemoguardrails find-providers --list
```

### Options

| Option | Description |
|--------|-------------|
| `--list`, `-l` | List all available providers without interactive selection. |

### List Mode

Run the following command to list all available providers:

```bash
nemoguardrails find-providers --list
```

### Interactive Mode

Run the command without options to start an interactive provider selection:

```bash
nemoguardrails find-providers
```

1. Select a provider type:
   - Type to filter between "text completion" and "chat completion".
   - Use arrow keys to navigate through matches.
   - Press Tab to autocomplete.
   - Press Enter to select.

2. Select a specific provider:
   - Type to filter through available providers.
   - Use arrow keys to navigate through matches.
   - Press Tab to autocomplete.
   - Press Enter to select.

#### Example

```text
Available Provider Types: (type to filter, use arrows to select)
  • text completion
  • chat completion

Select provider type: text

Available text completion providers: (type to filter, use arrows to select)
  • anthropic
  • google_palm
  • openai
  • ...

Select provider: openai

Selected text completion provider: openai
```
