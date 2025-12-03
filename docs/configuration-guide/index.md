# Configuration Overview

A guardrails configuration includes the following components:

| Component                    | Required/Optional | Description                                                                                                                                                                      | Location        |
|------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| **Core Configuration**       | Required          | A `config.yml` file that contains the core configuration options such as which LLM(s) to use, general instructions (similar to system prompts), sample conversation, which rails are active, and specific rails configuration options. | `config.yml`           |
| **Colang Rails**             | Optional          | A collection of Colang files implementing the rails.                                                                                                                             | `rails` folder         |
| **Knowledge Base Documents** | Optional          | Documents that can be used in a RAG (Retrieval-Augmented Generation) scenario using the built-in Knowledge Base support.                                                         | `kb` folder            |
| **Initialization Code**      | Optional          | Custom Python code performing additional initialization, e.g. registering a new type of LLM.                                                                                     | Usually `actions.py`    |

These files are typically included in a `config` folder, which is referenced when initializing a `RailsConfig` instance or when starting the CLI Chat or Server.

## Example Configuration Folder Structures

The following are example configuration folder structures.

- Basic configuration

    ```
    .
    ├── config
    │   └── config.yml
    ```

- Configuration with Colang Rails and a custom initialization code file `actions.py`

    ```
    .
    ├── config
    │   ├── config.yml
    │   ├── rails
    │   │   ├── file_1.co
    │   │   ├── file_2.co
    │   │   └── ...
    │   └── actions.py
    ```

- A complete configuration with all components: core configuration, Colang Rails, a collection of custom initialization code files in an `actions` sub-package, and a knowledge base folder

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
    │   └── kb
    │       ├── file_1.md
    │       ├── file_2.md
    │       └── ...
    ```

## Next Steps

For each component, refer to the following sections for more details:

- [Core Configuration](yaml-schema/index.md)
- [Colang Rails](colang/index.md)
- [Initialization Code](actions/index.md)
- [Knowledge Base Documents](other-configurations/knowledge-base.md)

After preparing your configuration files, use the NeMo Guardrails SDK to instantiate the core classes (`RailsConfig` and `LLMRails`) and run guardrails on your LLM applications.

For detailed SDK usage, including loading configurations, generating responses, streaming, and debugging, refer to [Run Rails](../run-rails/index.md).
