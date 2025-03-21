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

1. Create a _configuration store_ directory, such as `config` and add a `config/config.yml` file with the following contents:

   ```{literalinclude} ../examples/configs/gs_content_safety/config/config.yml
   :language: yaml
   ```

   The `models` key in the `config.yml` file configures the LLM model.
   For more information about the key, refer to [](./user-guides/configuration-guide.md#the-llm-model).

1. Create a prompts file, such as `config/prompts.yml`, ([download](../examples/configs/gs_content_safety/prompts.yml)), with contents like the following partial example:

   ```{literalinclude} ../examples/configs/gs_content_safety/config/prompts.yml
   :language: yaml
   :lines: 1-15
   ```

1. Load the guardrails configuration:

   ```{literalinclude} ../examples/configs/gs_content_safety/demo.py
   :language: python
   :start-after: "# start-load-config"
   :end-before: "# end-load-config"
   ```

1. Generate a response:

   ```{literalinclude} ../examples/configs/gs_content_safety/demo.py
   :language: python
   :start-after: "# start-generate-response"
   :end-before: "# end-generate-response"
   ```

   _Example Output_

   ```{literalinclude} ../examples/configs/gs_content_safety/demo-out.txt
   :language: text
   :start-after: "# start-generate-response"
   :end-before: "# end-generate-response"
   ```

## Timing and Token Information

The following modification of the sample code shows the timing and token information for the guardrail.

- Generate a response and print the timing and token information:

  ```{literalinclude} ../examples/configs/gs_content_safety/demo.py
  :language: python
  :start-after: "# start-get-duration"
  :end-before: "# end-get-duration"
  ```

  _Example Output_

  ```{literalinclude} ../examples/configs/gs_content_safety/demo-out.txt
  :language: text
  :start-after: "# start-get-duration"
  :end-before: "# end-get-duration"
  ```

  The timing and token information is available with the `print_llm_calls_summary()` method.

  ```{literalinclude} ../examples/configs/gs_content_safety/demo-out.txt
  :language: text
  :start-after: "# start-explain-info"
  :end-before: "# end-explain-info"
  ```
