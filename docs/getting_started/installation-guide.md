# Installation Guide

This guide walks you through the following steps to install NeMo Guardrails:

1. Setting up a fresh virtual environment.
2. Installing using `pip`.
3. Installing from Source Code.
4. Optional dependencies.
5. Using Docker.

## Prerequisites

Python 3.9, 3.10 or 3.11.

## Additional dependencies

NeMo Guardrails uses [annoy](https://github.com/spotify/annoy), which is a C++ library with Python bindings. To install it, you need to have a valid C++ runtime on your computer.
Most systems already have installed a C++ runtime. If the **annoy** installation fails due to a missing C++ runtime, you can install a C++ runtime as follows:

### Installing a C++ runtime on Linux, Mac, or Unix-based OS

  1. Install `gcc` and `g++` using `apt-get install gcc g++`.
  2. Update the following environment variables: `export CC=`*path_to_clang* and `export CXX=`*path_to_clang* (usually, *path_to_clang* is */usr/bin/clang*).
  3. In some cases, you might also need to install the `python-dev` package using `apt-get install python-dev` (or `apt-get install python3-dev`). Check out this [thread](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory) if the error persists.

### Installing a C++ runtime on Windows

Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). This installs Microsoft Visual C++ (version 14.0 or greater is required by the latest version of **annoy**).

## Setting up a virtual environment

To experiment with NeMo Guardrails from scratch, use a fresh virtual environment. Otherwise, you can skip to the following section.

### Setting up a virtual environment on Linux, Mac, or Unix-based OS

1. Create a folder, such as *my_assistant*, for your project.

 ```bash
 > mkdir my_assistant
 > cd my_assistant
 ```

2. Create a virtual environment.

 ```bash
 > python3 -m venv venv
 ```

3. Activate the virtual environment.

 ```bash
 > source venv/bin/activate
 ```

### Setting up a virtual environment on Windows

1. Open a new CMD prompt (Windows Key + R, **cmd.exe**)
2. Install **virtualenv** using the command `pip install virtualenv`
3. Check that **virtualenv** is installed using the command `pip --version`.
4. Install **virtualenvwrapper-win** using the command `pip install virtualenvwrapper-win`.

Use the `mkvirtualenv` *name* command to activate a new virtual environment called *name*.

## Install NeMo Guardrails

Install NeMo Guardrails using **pip**:

 ```bash
 > pip install nemoguardrails
 ```

## Installing from source code

NeMo Guardrails is under active development and the main branch always contains the latest development version. To install from source:

1. Clone the repository:

   ```
   git clone https://github.com/NVIDIA/NeMo-Guardrails.git
   ```

2. Install the package locally:

   ```
   cd NeMo-Guardrails
   pip install -e .
   ```

## Extra dependencies

The `nemoguardrails` package also defines the following extra dependencies:

- `dev`: packages required by some extra Guardrails features for developers, such as the **autoreload** feature.
- `eval`: packages used for the Guardrails [evaluation tools](../../nemoguardrails/evaluate/README.md).
- `openai`: installs the latest `openai` package supported by NeMo Guardrails.
- `sdd`: packages used by the [sensitive data detector](../user_guides/guardrails-library.md#sensitive-data-detection) integrated in NeMo Guardrails.
- `all`: installs all extra packages.

To keep the footprint of `nemoguardrails` as small as possible, these are not installed by default. To install any of the extra dependency you can use **pip** as well. For example, to install the `dev` extra dependencies, run the following command:

```bash
> pip install nemoguardrails[dev]
```

## Optional dependencies
>
> [!IMPORTANT]
> pip fails to resolve dependencies for **v0.10.0** and **v0.10.1**. We recommend using a `constraints.txt` file to ensure compatible versions of dependencies. This only concerns you when you want to install the `[all]` extra dependencies.
>
> **Example `constraints.txt`**:
>
> ```
> pandas>=1.4.0,<3
> ```
>
> **Installation Instructions**:
> To install the package with the specified constraints, use the following command:
>
> ```sh
> pip install "nemoguardrails[all]" -c constraints.txt
> ```

To use OpenAI, just use the `openai` extra dependency that ensures that all required packages are installed.
Make sure the `OPENAI_API_KEY` environment variable is set,
as shown in the following example, where *YOUR_KEY* is your OpenAI key.

 ```bash
 > pip install nemoguardrails[openai]
 > export OPENAI_API_KEY=YOUR_KEY
```

Some NeMo Guardrails LLMs and features have specific installation requirements, including a more complex set of steps. For example, [AlignScore](../user_guides/advanced/align_score_deployment.md) fact-checking, using [Llama-2](../../examples/configs/llm/hf_pipeline_llama2/README.md) requires two additional packages.
For each feature or LLM example, check the readme file associated with it.

## Using Docker

NeMo Guardrails can also be used through Docker. For details on how to build and use the Docker image see [NeMo Guardrails with Docker](../user_guides/advanced/using-docker.md).

## What's next?

- Check out the [Getting Started Guide](../getting_started/README.md) and start with the ["Hello World" example](../getting_started/1_hello_world/README.md).
- Explore more examples in the [examples](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples) folder.
- Review the [User Guides](../README.md).
