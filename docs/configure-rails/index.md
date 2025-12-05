---
title: Configure Rails
description: Prepare configuration files including config.yml, Colang flows, actions.py, config.py, and knowledge base documents.
---

# Configuration Overview

Before using the NeMo Guardrails toolkit, you need to prepare configuration files that define your guardrails behavior. This section provides complete instructions on preparing your configuration files and executable scripts.

A guardrails configuration includes the following components. You can start with a basic configuration and add more components as needed. All the components should be placed in the `config` folder, and the locations in the table are relative to the `config` folder.

| Component                    | Required/Optional | Description                                                                                                                                                                      | Location        |
|------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| **Core Configuration**       | Required          | A `config.yml` file that contains the core configuration options such as which LLM(s) to use, general instructions (similar to system prompts), sample conversation, which rails are active, and specific rails configuration options. | `config.yml`           |
| **Colang Flows**             | Optional          | A collection of Colang files (`.co` files) implementing the rails.                                                                                                               | `rails` folder         |
| **Custom Actions**           | Optional          | Python functions decorated with `@action()` that can be called from Colang flows during request processing (for example, external API calls, validation logic).                                 | `actions.py` or `actions/` folder |
| **Custom Initialization**    | Optional          | Python code that runs once at startup to register custom LLM providers, embedding providers, or shared resources (for example, database connections).                                            | `config.py`            |
| **Knowledge Base Documents** | Optional          | Documents (`.md` files) that can be used in a RAG (Retrieval-Augmented Generation) scenario using the built-in Knowledge Base support.                                           | `kb` folder            |

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

## Next Steps

For each component, refer to the following sections for more details:

- [Core Configuration](yaml-schema/index.md) - `config.yml` reference
- [Colang Rails](colang/index.md) - `.co` flow files
- [Custom Actions](actions/index.md) - `actions.py` for callable actions
- [Custom Initialization](custom-initialization/index.md) - `config.py` for provider registration
- [Knowledge Base Documents](other-configurations/knowledge-base.md) - `kb/` folder for RAG

After preparing your configuration files, use the NeMo Guardrails SDK to instantiate the core classes (`RailsConfig` and `LLMRails`) and run guardrails on your LLM applications.

For detailed SDK usage, including loading configurations, generating responses, streaming, and debugging, refer to [Run Rails](../run-rails/index.md).

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Core Configuration
:link: yaml-schema/index
:link-type: doc

Complete reference for config.yml structure including models, guardrails, prompts, and tracing settings.
:::

:::{grid-item-card} Colang Guide
:link: colang/index
:link-type: doc

Learn Colang, the event-driven language for defining guardrails flows, user messages, and bot responses.
:::

:::{grid-item-card} Custom Actions
:link: actions/index
:link-type: doc

Define custom Python actions in actions.py to extend guardrails with external integrations and validation logic.
:::

:::{grid-item-card} Custom Initialization
:link: custom-initialization/index
:link-type: doc

Use config.py to register custom LLM providers, embedding providers, and shared resources at startup.
:::

:::{grid-item-card} Other Configurations
:link: other-configurations/index
:link-type: doc

Additional configuration topics including knowledge base setup and exception handling.
:::

::::
