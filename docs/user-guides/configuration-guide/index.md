# Configuration Guide

A guardrails configuration includes the following components:

- **General Options**: which LLM(s) to use, general instructions (similar to system prompts), sample conversation, which rails are active, specific rails configuration options, etc.; these options are typically placed in a `config.yml` file.
- **Rails**: Colang flows implementing the rails; these are typically placed in a `rails` folder.
- **Actions**: custom actions implemented in Python; these are typically placed in an `actions.py` module in the root of the config or in an `actions` sub-package.
- **Knowledge Base Documents**: documents that can be used in a RAG (Retrieval-Augmented Generation) scenario using the built-in Knowledge Base support; these documents are typically placed in a `kb` folder.
- **Initialization Code**: custom Python code performing additional initialization, e.g. registering a new type of LLM.

These files are typically included in a `config` folder, which is referenced when initializing a `RailsConfig` instance or when starting the CLI Chat or Server.

```
.
├── config
│   ├── rails
│   │   ├── file_1.co
│   │   ├── file_2.co
│   │   └── ...
│   ├── actions.py
│   ├── config.py
│   └── config.yml
```

The custom actions can be placed either in an `actions.py` module in the root of the config or in an `actions` sub-package:

```
.
├── config
│   ├── rails
│   │   ├── file_1.co
│   │   ├── file_2.co
│   │   └── ...
│   ├── actions
│   │   ├── file_1.py
│   │   ├── file_2.py
│   │   └── ...
│   ├── config.py
│   └── config.yml
```

## Configuration Guide Sections

- [Custom Initialization](custom-initialization.md) - Setting up custom initialization code
- [General Options](general-options.md) - Configuring LLM models, embeddings, and basic settings
- [LLM Configuration](llm-configuration.md) - Detailed LLM provider configuration and options
- [Guardrails Configuration](guardrails-configuration.md) - Setting up input, output, dialog, and retrieval rails
- [Tracing Configuration](tracing-configuration.md) - Monitoring and logging interactions
- [Knowledge Base](knowledge-base.md) - Setting up document retrieval and RAG functionality
- [Exceptions and Error Handling](exceptions.md) - Managing exceptions and error responses

```{toctree}
:maxdepth: 2
:hidden:

custom-initialization.md
general-options.md
llm-configuration.md
guardrails-configuration.md
tracing-configuration.md
knowledge-base.md
exceptions.md
```
