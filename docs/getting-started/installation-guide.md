---
title:
  page: "Install the NeMo Guardrails Library"
  nav: "Install"
description: "Install NeMo Guardrails with pip, configure your environment, and verify the installation."
topics: ["Get Started", "AI Safety"]
tags: ["Installation", "Python", "pip", "Docker", "Setup"]
content:
  type: "Get Started"
  difficulty: "Beginner"
  audience: ["Developer", "AI Engineer"]
---

# Install the NeMo Guardrails Library

Follow these steps to install the NeMo Guardrails library.

## Requirements

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
   .venv\Scripts\activate
   ```

   :::

   ::::

1. Install the NeMo Guardrails library with support for NVIDIA-hosted models. Set `NVIDIA_API_KEY` to your personal API key generated on [build.nvidia.com](https://build.nvidia.com/).

   ```bash
   pip install "nemoguardrails[nvidia]"
   ```

1. Set up an environment variable for your NVIDIA API key.

   ```bash
   export NVIDIA_API_KEY="your-nvidia-api-key"
   ```

   This is required to access NVIDIA-hosted models on [build.nvidia.com](https://build.nvidia.com). The tutorials and example configurations ([examples/configs](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs)) in this library include configurations that use NVIDIA-hosted models.

## Install from Source

To use the latest development version:

::::{tab-set}

:::{tab-item} pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install nemoguardrails[nvidia]
export NVIDIA_API_KEY="nvapi-..."
```

:::

:::{tab-item} pip (development)

```bash
git clone https://github.com/NVIDIA-NeMo/Guardrails.git
cd NeMo-Guardrails
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

:::

:::{tab-item} Poetry (development)

```bash
git clone https://github.com/NVIDIA-NeMo/Guardrails.git
cd NeMo-Guardrails
python -m venv .venv
source .venv/bin/activate
poetry install --extras "nvidia"
```

When using Poetry, prefix CLI commands with `poetry run`:

```bash
poetry run nemoguardrails server --config examples/configs
```

:::

::::

## Extra Dependencies

The NeMo Guardrails library can be installed with optional "extras", adding functionality that isn't required for the core Guardrails library. The table below shows some popular extras. For a comprehensive list, see the `[tool.poetry.extras]` section in [pyproject.toml](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/pyproject.toml).

| Extra | Description |
|-------|-------------|
| `nvidia` | NVIDIA-hosted model integration through [build.nvidia.com](https://build.nvidia.com/) |
| `openai` | OpenAI-hosted model integration |
| `sdd` | [Sensitive data detection](../configure-rails/yaml-schema/guardrails-configuration/built-in-guardrails.md#presidio-based-sensitive-data-detection) using Presidio |
| `eval` | [Evaluation tools](../evaluation/index.rst) for testing guardrails |
| `tracing` | OpenTelemetry tracing support |
| `gcp` | Google Cloud Platform language services |
| `jailbreak` | YARA-based jailbreak detection heuristics |
| `multilingual` | Language detection for multilingual content |
| `all` | All optional packages |

Some features such as [AlignScore](../user-guides/community/alignscore.md) have additional requirements. See the feature documentation for details.

## Docker

You can run the NeMo Guardrails library in a Docker container. For containerized deployment, see [NeMo Guardrails with Docker](../deployment/using-docker.md).

## Troubleshooting Installation Issues

Use the following information to resolve common installation issues.

### C++ Runtime Errors

The library uses [annoy](https://github.com/spotify/annoy), which requires a C++ compiler. If installation fails:

::::{tab-set}

:::{tab-item} Linux/macOS

```bash
apt-get install gcc g++ python3-dev
```

:::

:::{tab-item} Windows

Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (version 14.0 or greater).

:::

::::
