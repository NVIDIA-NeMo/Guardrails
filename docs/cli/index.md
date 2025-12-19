---
title:
  page: "NeMo Guardrails Library CLI Reference"
  nav: "CLI"
description: "Use the nemoguardrails command-line interface to chat, serve, evaluate, and convert configurations."
topics: ["AI Safety", "LLM Guardrails"]
tags: ["CLI", "Python", "Server", "Evaluation", "Colang"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

(nemoguardrails-cli)=

# NeMo Guardrails Library CLI Reference

**NOTE: THIS SECTION IS WORK IN PROGRESS.**

For testing purposes, the NeMo Guardrails library provides a command line chat that can be used to interact with the LLM.

```sh
nemoguardrails chat --config examples/ [--verbose] [--verbose-llm-calls]
```

## Options

- `--config`: The configuration that should be used. Can be a folder or a .co/.yml file.
- `--verbose`: In verbose mode, detailed debugging information is also shown.
- `--verbose-llm-calls`: In verbose LLM calls mode, the debugging information includes the entire prompt that is sent to the LLM and the completion.

You should now be able to invoke the `nemoguardrails` CLI.

 ```bash
 > nemoguardrails --help

 Usage: nemoguardrails [OPTIONS] COMMAND [ARGS]...

 Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

 Commands:
  actions-server  Starts a NeMo Guardrails actions server.
  chat            Starts an interactive chat session.
  convert         Convert a Colang 1.0 directory to Colang 2.0 format.
  evaluate        Run an evaluation task.
  server          Starts a NeMo Guardrails server.
 ```

 You can also use the `--help` flag to learn more about each of the `nemoguardrails` commands:

### actions-server

 ```bash
 > nemoguardrails actions-server --help

 Usage: nemoguardrails actions-server [OPTIONS]

  Starts a NeMo Guardrails actions server.

 Options:
  --port INTEGER  The port that the server should listen on.   [default: 8001]
  --help          Show this message and exit.
 ```

### chat

 ```bash
 > nemoguardrails chat --help

 Usage: nemoguardrails chat [OPTIONS]

  Starts an interactive chat session.

  --config                                       TEXT  Path to a directory containing configuration
                                                       files to use. Can also point to a single
                                                       configuration file.
                                                       [default: config]
  --verbose             --no-verbose                   If the chat should be verbose and output
                                                       detailed logging information.
                                                       [default: no-verbose]
  --verbose-no-llm      --no-verbose-no-llm            If the chat should be verbose and exclude the
                                                       prompts and responses for the LLM calls.
                                                       [default: no-verbose-no-llm]
  --verbose-simplify    --no-verbose-simplify          Simplify further the verbose output.
                                                       [default: no-verbose-simplify]
  --debug-level                                  TEXT  Enable debug mode which prints rich
                                                       information about the flows execution.
                                                       Available levels: WARNING, INFO, DEBUG
  --streaming           --no-streaming                 If the chat should use the streaming mode, if
                                                       possible.
                                                       [default: no-streaming]
  --server-url                                   TEXT  If specified, the chat CLI will interact with
                                                       a server, rather than load the config. In this
                                                       case, the --config-id must also be specified.
                                                       [default: None]
  --config-id                                    TEXT  The config_id to be used when interacting with
                                                       the server.
                                                       [default: None]
  --help                                               Show this message and exit.
 ```

### server

```bash
> nemoguardrails server --help

Usage: nemoguardrails server [OPTIONS]

Starts a NeMo Guardrails server.

Options:
--port                                        INTEGER  The port that the server should listen on. [default: 8000]
--config                                      TEXT     Path to a directory containing multiple configuration sub-folders.
--verbose           --no-verbose:                      If the server should be verbose and output detailed logs including prompts. [default: no-verbose]
--disable-chat-ui   --no-disable-chat-ui               Weather the ChatUI should be disabled [default: no-disable-chat-ui]
--auto-reload       --no-auto-reload                   Enable auto reload option. [default: no-auto-reload]
--prefix                                      TEXT     A prefix that should be added to all server paths. Should start with '/'.
--help                                                Show this message and exit.
```

### evaluate

```bash
> nemoguardrails evaluate --help

Usage: nemoguardrails evaluate [OPTIONS] COMMAND [ARGS]...

Options:
--help:          Show this message and exit.

Commands:
fact-checking:   Evaluate the performance of the fact-checking rails defined in a Guardrails application.
hallucination:   Evaluate the performance of the hallucination rails defined in a Guardrails application.
moderation:      Evaluate the performance of the moderation rails defined in a Guardrails application.
topical:         Evaluates the performance of the topical rails defined in a Guardrails application. Computes accuracy for canonical form detection, next step generation, and next bot message generation. Only a single Guardrails application can be specified in the config option.
```

### convert

```bash
> nemoguardrails convert --help

Usage: nemoguardrails convert [OPTIONS] PATH

Convert a Colang 1.0 directory to Colang 2.0.

Arguments:
  path TEXT The path to the file or directory to migrate. [default: None] [required]

Options:
--verbose                       --no-verbose                If the migration should be verbose and output detailed logs. [default: no-verbose]
--validate                      --no-validate               If the migration should validate the output using Colang Parser. [default: no-validate]
--use-active-decorator          --no-use-active-decorator   If the migration should use the active decorator. [default: use-active-decorator]
--help                                                      Show this message and exit.
```

(find-providers-command)=

### find-providers

```bash
> nemoguardrails find-providers --help

Usage: nemoguardrails find-providers [OPTIONS]

Interactive provider selection.

This command provides an interactive interface to select between text completion
and chat completion providers. It will guide you through selecting the type of
provider (text completion or chat completion) and then show you the available
providers for that type.

Options:
  --list, -l    Lists all available providers without interactive selection
  --help        Show this message and exit.
```

#### List Mode

Run the following command to list all available providers:

```bash
nemoguardrails find-providers [--list]
```

#### Interactive Mode

Run the following command start an interactive process to select a provider:

```bash
nemoguardrails find-providers
```

1. First, you'll be prompted to select a provider type:
   - Type to filter between "text completion" and "chat completion", you can press Tab to autocomplete.
   - Use arrow keys to navigate through matches
   - Press Tab to autocomplete
   - Press Enter to select

2. Then, you'll be prompted to select a specific provider:
   - Type to filter through available providers
   - Use arrow keys to navigate through matches
   - Press Tab to autocomplete
   - Press Enter to select

##### Example of Interactive Mode

```
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
