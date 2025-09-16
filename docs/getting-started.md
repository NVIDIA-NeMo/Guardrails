<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Getting Started

## Adding Content Safety Guardrails

The following procedure adds a guardrail to check user input against a content safety model.

To simplify configuration, the sample code sends the prompt text and the model response to the
[Llama 3.1 NemoGuard 8B Content Safety model](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety) deployed on the NVIDIA API Catalog.

The prompt text is also sent to NVIDIA API Catalog as the application LLM.
The sample code uses the [Llama 3.3 70B Instruct model](https://build.nvidia.com/meta/llama-3_3-70b-instruct).

## Prerequisites

- You must be a member of the NVIDIA Developer Program and you must have an NVIDIA API key.
  For information about the program and getting a key, refer to [NVIDIA NIM FAQ](https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317/1) in the NVIDIA NIM developer forum.

- You [installed NeMo Guardrails](./getting-started/installation-guide.md).

- You installed LangChain NVIDIA AI Foundation Model Playground Integration:

  ```console
  $ pip install langchain-nvidia-ai-endpoints
  ```

## Procedure

1. Set your NVIDIA API key as an environment variable:

   ```console
   $ export NVIDIA_API_KEY=<nvapi-...>
   ```

1. Create a _configuration store_ directory, such as `config`.
2. Copy the following configuration code and save as `config.yml` in the `config` directory.

   ```{literalinclude} ../examples/configs/gs_content_safety/config/config.yml
   :language: yaml
   ```

   The `models` key in the `config.yml` file configures the LLM model.
   For more information about the key, refer to [](./user-guides/configuration-guide.md#the-llm-model).

3. Copy the following prompts code and save as `prompts.yml` in the `config` directory.

   ```{literalinclude} ../examples/configs/gs_content_safety/config/prompts.yml
   :language: yaml
   ```

4. Run the following code to load the guardrails configurations from the previous steps and try out unsafe and safe inputs.

   ```{literalinclude} ../examples/configs/gs_content_safety/demo.py
   :language: python
   :start-after: "# start-generate-response"
   :end-before: "# end-generate-response"
   ```

   The following is an example response of the unsafe input.

   ```{literalinclude} ../examples/configs/gs_content_safety/demo-out.txt
   :language: text
   :start-after: "# start-unsafe-response"
   :end-before: "# end-unsafe-response"
   ```

   The following is an example response of the safe input.

   ```{literalinclude} ../examples/configs/gs_content_safety/demo-out.txt
   :language: text
   :start-after: "# start-safe-response"
   :end-before: "# end-safe-response"
   ```

## Next Steps

- Run the `content_safety_tutorial.ipynb` notebook from the
  [example notebooks](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/notebooks)
  directory of the GitHub repository.
  The notebook compares LLM responses with and without safety checks and classifies responses
  to sample prompts as _safe_ or _unsafe_.
  The notebook shows how to measure the performance of the checks, focusing on how many unsafe
  responses are blocked and how many safe responses are incorrectly blocked.

- Refer to [](user-guides/configuration-guide.md) for information about the `config.yml` file.
