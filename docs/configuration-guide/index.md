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
│   ├── config.yml
│   ├── rails
│   │   ├── file_1.co
│   │   ├── file_2.co
│   │   └── ...
│   ├── actions.py
│   └── config.py
```

The custom actions can be placed either in an `actions.py` module in the root of the config or in an `actions` sub-package:

```
.
├── config
│   ├── config.yml
│   ├── rails
│   │   ├── file_1.co
│   │   ├── file_2.co
│   │   └── ...
│   ├── actions
│   │   ├── file_1.py
│   │   ├── file_2.py
│   │   └── ...
│   └── config.py
```
