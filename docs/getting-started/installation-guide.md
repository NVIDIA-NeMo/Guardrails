---
title:
  page: Install the NeMo Guardrails Library
  nav: Installation
description: Install NeMo Guardrails with pip, configure your environment, and verify the installation.
topics:
- Get Started
- AI Safety
tags:
- Installation
- Python
- pip
- Docker
- Setup
content:
  type: get_started
  difficulty: technical_beginner
  audience:
  - engineer
  - AI Engineer
---

# Install the NeMo Guardrails Library

Follow these steps to install the NeMo Guardrails library.

## Prerequisites

Verify your system meets the following requirements before installation.

| Requirement | Details |
|-------------|---------|
| **Operating System** | Windows, Linux, MacOS |
| **Python** | 3.10, 3.11, 3.12, or 3.13 |
| **Hardware** | 1 CPU with 4GB RAM. The NeMo Guardrails library runs on CPU. External models may require GPUs, which may be deployed separately to the library |

## Quick Start

Use the following steps to install the NeMo Guardrails library in a virtual environment.

1. Create and activate a virtual environment:

   ::::{tab-set}

   :::{tab-item} Linux/macOS

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   :::

   :::{tab-item} Windows (Git Bash)

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```

   :::

   ::::

1. Install the NeMo Guardrails library. Set `NVIDIA_API_KEY` to your personal API key generated on [build.nvidia.com](https://build.nvidia.com/).

   ```bash
   pip install nemoguardrails
   ```

1. Set up an environment variable for your NVIDIA API key.

   ```bash
   export NVIDIA_API_KEY="your-nvidia-api-key"
   ```

   This is required to access NVIDIA-hosted models on [build.nvidia.com](https://build.nvidia.com). The tutorials and example configurations ([examples/configs](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs)) in this library include configurations that use NVIDIA-hosted models.

## Alternative Installation Methods

Install the NeMo Guardrails library from source using pip or Poetry. Choose this method if you want to contribute to the library or use the latest development version.

::::{tab-set}

:::{tab-item} pip

```bash
git clone https://github.com/NVIDIA-NeMo/Guardrails.git nemoguardrails
cd nemoguardrails
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

:::

:::{tab-item} Poetry

```bash
git clone https://github.com/NVIDIA-NeMo/Guardrails.git nemoguardrails
cd nemoguardrails
python -m venv .venv
source .venv/bin/activate
poetry install
```

When using Poetry, prefix CLI commands with `poetry run`:

```bash
poetry run nemoguardrails server --config examples/configs
```

:::

::::

## Extra Dependencies

You can install the NeMo Guardrails library with optional extra packages to add useful functionalities. The table below shows a comprehensive list.

| Extra | Description |
|-------|-------------|
| `server` | [Guardrails API server](../run-rails/using-fastapi-server/overview.md) dependencies (aiofiles for async file handling, openai for API schemas). FastAPI is a core dependency. Required to run `nemoguardrails server`. |
| `sdd` | [Sensitive data detection](../configure-rails/guardrail-catalog/pii-detection.md#presidio-based-sensitive-data-detection) using Presidio |
| `eval` | [Evaluation tools](../evaluation/evaluate-guardrails.md) for testing guardrails |
| `tracing` | OpenTelemetry tracing support |
| `gcp` | Google Cloud Platform language services |
| `jailbreak` | YARA-based jailbreak detection heuristics |
| `multilingual` | Language detection for multilingual content |
| `all` | All optional packages |

Some features such as [AlignScore](../configure-rails/guardrail-catalog/community/alignscore.md) have additional requirements. See the feature documentation for details.

## Docker

You can run the NeMo Guardrails library in a Docker container. For containerized deployment, see [NeMo Guardrails with Docker](../deployment/using-docker.md).
