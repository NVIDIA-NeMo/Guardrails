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
  audience: ["Developer", "AI Engineer"]gi
---

# Install the NeMo Guardrails Library

This guide walks you through the following steps to install the NeMo Guardrails library.

1. Check the requirements.
2. Set up a fresh virtual environment.
3. Install using `pip`.
4. Install from Source Code.
5. Install optional dependencies.
6. Use Docker.

## Requirements

Review the following requirements to install the NeMo Guardrails library.

| Requirement Type     | Details                                                                                                                                      |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Hardware**         | The Guardrails process runs on CPU. Guardrails models run on GPUs and can be deployed on a separate host or environment.                            |
| **Software**         | Python 3.10, 3.11, 3.12, or 3.13                                                                                                           |

### Additional Dependencies

The NeMo Guardrails library uses [annoy](https://github.com/spotify/annoy), which is a C++ library with Python bindings. To install it, you need to have a valid C++ runtime on your computer.
Most systems already have installed a C++ runtime. If the **annoy** installation fails due to a missing C++ runtime, you can install a C++ runtime as follows:

#### Installing a C++ runtime on Linux, Mac, or Unix-based OS

  1. Install `gcc` and `g++` using `apt-get install gcc g++`.
  2. Update the following environment variables: `export CC=`*path_to_clang* and `export CXX=`*path_to_clang* (usually, *path_to_clang* is */usr/bin/clang*).
  3. In some cases, you might also need to install the `python-dev` package using `apt-get install python-dev` (or `apt-get install python3-dev`). Check out this [thread](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory) if the error persists.

#### Installing a C++ runtime on Windows

Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). This installs Microsoft Visual C++ (version 14.0 or greater is required by the latest version of **annoy**).

## Setting up a virtual environment

To experiment with the NeMo Guardrails library from scratch, use a fresh virtual environment. Otherwise, you can skip to the following section.

### Setting up a virtual environment on Linux, Mac, or Unix-based OS

1. Create a folder, such as *my_assistant*, for your project.

 ```sh
 mkdir my_assistant
 cd my_assistant
 ```

2. Create a virtual environment.

 ```sh
 python3 -m venv venv
 ```

3. Activate the virtual environment.

 ```sh
 source venv/bin/activate
 ```

### Setting up a virtual environment on Windows

1. Open a new CMD prompt (Windows Key + R, **cmd.exe**)
2. Install **virtualenv** using the command `pip install virtualenv`
3. Check that **virtualenv** is installed using the command `pip --version`.
4. Install **virtualenvwrapper-win** using the command `pip install virtualenvwrapper-win`.

Use the `mkvirtualenv` *name* command to activate a new virtual environment called *name*.

## Install the NeMo Guardrails Library

Install the NeMo Guardrails library using **pip**:

 ```sh
 pip install nemoguardrails
 ```

## Installing from source code

The NeMo Guardrails library is under active development and the main branch always contains the latest development version. To install from source:

1. Clone the repository:

   ```sh
   git clone https://github.com/NVIDIA/NeMo-Guardrails.git
   ```

2. Install the package locally:

   ```sh
   cd NeMo-Guardrails
   pip install -e .
   ```

## Extra dependencies

The `nemoguardrails` package also defines the following extra dependencies:

- `dev`: packages required by some extra Guardrails features for developers, such as the **autoreload** feature.
- `eval`: packages used for the Guardrails [evaluation tools](../../nemoguardrails/evaluate/README.md).
- `openai`: installs the latest `openai` package supported by the NeMo Guardrails library.
- `sdd`: packages used by the [sensitive data detector](../user-guides/guardrails-library.md#sensitive-data-detection) integrated in the NeMo Guardrails library.
- `all`: installs all extra packages.

To keep the footprint of `nemoguardrails` as small as possible, these are not installed by default. To install any of the extra dependencies you can use **pip** as well. For example, to install the `dev` extra dependencies, run the following command:

```sh
> pip install nemoguardrails[dev]
```

## Optional dependencies

```{warning}
If pip fails to resolve dependencies when running `pip install nemoguardrails[all]`, you should specify additional constraints directly in the `pip install` command.

Example Command:

```sh
pip install "nemoguardrails[all]" "pandas>=1.4.0,<3"
```

To use OpenAI, just use the `openai` extra dependency that ensures that all required packages are installed.
Make sure the `OPENAI_API_KEY` environment variable is set,
as shown in the following example, where *YOUR_KEY* is your OpenAI key.

 ```sh
 pip install nemoguardrails[openai]
 export OPENAI_API_KEY=YOUR_KEY
```

Some NeMo Guardrails library LLMs and features have specific installation requirements, including a more complex set of steps. For example, [AlignScore](../user-guides/advanced/align_score_deployment.md) fact-checking using [Llama-2](../../examples/configs/llm/hf_pipeline_llama2/README.md) requires two additional packages.
For each feature or LLM example, check the readme file associated with it.

## Using Docker

The NeMo Guardrails library can also be used through Docker. For details on how to build and use the Docker image see [NeMo Guardrails with Docker](../user-guides/advanced/using-docker.md).

## What's next?

- Check out the [Getting Started Guide](../getting-started/README.md) and start with the ["Hello World" example](../getting-started/1-hello-world/README.md).
- Explore more examples in the [examples](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples) folder.
- Review the [User Guides](../README.md).
