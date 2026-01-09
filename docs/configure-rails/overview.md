---
title:
  page: "NeMo Guardrails Library Configuration Overview"
  nav: "Overview"
description: "Learn to write config.yml, Colang flows, and custom actions for guardrails."
topics: ["Configuration", "AI Safety"]
tags: ["Configuration", "YAML", "Colang", "Actions", "Setup"]
content:
  type: "Overview"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# NeMo Guardrails Library Configuration Overview

Before using the NeMo Guardrails library, you need to prepare configuration files that define your guardrails behavior. When you initialize the library's core classes or the `nemoguardrails` CLI chat or server, it will load these configuration files as shown in the next chapter [](../run-rails/index.md). This section provides complete instructions on preparing your configuration files and executable scripts.

A guardrails configuration includes the following components. You can start with a basic configuration and add more components as needed. All the components should be placed in the `config` folder, and the locations in the following table are relative to the `config` folder.

| Component                    | Required/Optional | Description                                                                                                                                                                      | Location        |
|------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| **Core Configuration**       | Required          | A `config.yml` file that contains the core configuration options such as which LLM(s) to use, general instructions (similar to system prompts), sample conversation, which rails are active, and specific rails configuration options. | `config.yml`           |
| **Colang Flows**             | Optional          | A collection of Colang files (`.co` files) implementing the rails.                                                                                                               | `rails` folder         |
| **Custom Actions**           | Optional          | Python functions decorated with `@action()` that can be called from Colang flows during request processing (for example, external API calls, validation logic).                                 | `actions.py` or `actions/` folder |
| **Custom Initialization**    | Optional          | Python code that runs once at startup to register custom LLM providers, embedding providers, or shared resources (for example, database connections).                                            | `config.py`            |
| **Knowledge Base Documents** | Optional          | Documents (`.md` files) that can be used in a RAG (Retrieval-Augmented Generation) scenario (i.e. Retrieval rail) using the built-in Knowledge Base support.                                           | `kb` folder            |

---

## Example Configuration Folder Structures

The following are example configuration folder structures.

- Basic configuration

    ```text
    config/
    └── config.yml
    ```

- Configuration with Colang rails and custom actions

    ```text
    config/
    ├── config.yml
    ├── rails/
    │   ├── input.co
    │   ├── output.co
    │   └── ...
    └── actions.py          # Custom actions called from Colang flows
    ```

- Configuration with custom LLM provider registration

    ```text
    config/
    ├── config.yml
    ├── rails/
    │   └── ...
    ├── actions.py          # Custom actions
    └── config.py           # Registers custom LLM provider at startup
    ```

- Complete configuration with all components

    ```text
    config/
    ├── config.yml          # Core configuration
    ├── config.py           # Custom initialization (LLM providers, etc.)
    ├── rails/              # Colang flow files
    │   ├── input.co
    │   ├── output.co
    │   └── ...
    ├── actions/            # Custom actions (as a package)
    │   ├── __init__.py
    │   ├── validation.py
    │   ├── external_api.py
    │   └── ...
    └── kb/                 # Knowledge base documents
        ├── policies.md
        ├── faq.md
        └── ...
    ```

---

## Next Steps

For each component, refer to the following sections for more details:

- [](yaml-schema/index.md) - A complete guide to writing your `config.yml` file.
- [](colang/index.md) - `.co` flow files in `rails` folder.
- [](actions/index.md) - `actions.py` or `actions/` folder for callable actions.
- [](custom-initialization/index.md) - `config.py` for custom initialization.
- [](other-configurations/knowledge-base.md) - `kb/` folder for RAG.

After preparing your configuration files, you can use the NeMo Guardrails SDK to instantiate the core classes (`RailsConfig` and `LLMRails`) and run guardrails on your LLM applications.
For detailed SDK usage, including loading configurations, generating responses, streaming, and debugging, refer to [](../run-rails/index.md).
