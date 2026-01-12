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
   python3 -m venv venv
   source venv/bin/activate
   ```

   :::

   :::{tab-item} Windows

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   :::

   ::::

2. Install the Nemo Guardrails toolkit with support for NVIDIA-hosted models. Set NVIDIA_API_KEY to your personal API Key generated on <https://build.nvidia.com/>.

   ```bash
   pip install "nemoguardrails[nvidia]"
   export NVIDIA_API_KEY="nvapi-..."
   ```

## Install from Source

To use the latest development version:

```bash
git clone https://github.com/NVIDIA/NeMo-Guardrails.git
cd NeMo-Guardrails
pip install -e .
```

## Extra Dependencies

The NeMo Guardrails library can be installed with optional "extras", adding functionality that isn't required for the core Guardrails library. The table below shows some popular extras. For a comprehensive list, see the `[tool.poetry.extras]` section in [pyproject.toml](https://github.com/NVIDIA-NeMo/Guardrails/blob/develop/pyproject.toml).

| Extra | Description |
|-------|-------------|
| `nvidia` | NVIDIA-hosted model integration (<https://build.nvidia.com/>) |
| `openai` | OpenAI-hosted model integration |
| `eval` | [Evaluation tools](../../nemoguardrails/evaluate/README.md) |
| `sdd` | [Sensitive data detection](../configure-rails/yaml-schema/guardrails-configuration/built-in-guardrails.md#presidio-based-sensitive-data-detection) using Presidio |
| `dev` | Developer features like autoreload |
| `all` | All optional packages |

To prevent the shell from interpreting square brackets as a glob pattern, enclose the nemoguardrails and extensions in double-quotes.

```bash
pip install "nemoguardrails[openai]"     # Single extra
pip install "nemoguardrails[eval,sdd]"   # Multiple extras
pip install "nemoguardrails[all]"        # Everything
```

Some features like [AlignScore](../user-guides/community/alignscore.md) have additional requirements. Check the feature documentation for details.

## Docker

You can run the NeMo Guardrails library in a Docker container. For containerized deployment, see [NeMo Guardrails with Docker](../deployment/using-docker.md).

## Troubleshooting

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
