---
title:
  page: "Check Harmful Content with Nemotron Content Safety NIM"
  nav: "Check Harmful Content"
description: "Check text inputs and outputs for harmful content using Nemotron Content Safety NIM."
topics: ["AI Safety", "Content Safety"]
tags: ["Content Safety", "NIM", "Multilingual", "Input Rails", "Output Rails", "Docker", "Nemotron"]
content:
  type: "Tutorial"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Check Harmful Content with Llama 3.1 Nemotron Safety Guard 8B V3 NIM

Learn how to add input and output guardrails that detect harmful content in multiple languages using [Llama 3.1 Nemotron Safety Guard 8B V3](https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3).

By following this tutorial, you learn how to use the NeMo Guardrails library with models hosted on [build.nvidia.com](https://build.nvidia.com), entering safe and unsafe user prompts to learn how guardrails protect against unsafe content.

## Prerequisites

- The NeMo Guardrails library [installed](../installation-guide.md) with the `nvidia` extra.
- A personal NVIDIA API key generated on [build.nvidia.com](https://build.nvidia.com/).

## Configure Guardrails

Follow these steps to prepare the guardrails configuration.

1. Create a configuration directory by running the following command.

   ```console
   mkdir config
   ```

1. Save the following as `config/config.yml`.

   ```yaml
   models:
     - type: main
       engine: nim
       model: meta/llama-3.3-70b-instruct

     - type: content_safety
       engine: nim
       model: nvidia/llama-3.1-nemotron-safety-guard-8b-v3

   rails:
     input:
       flows:
         - content safety check input $model=content_safety
     output:
       flows:
         - content safety check output $model=content_safety
   ```

   The `config.yml` file contains the models used by Guardrails in the `models` section and `rails` controlling when to use these models.
   The `models` section configures the type and name of each model, along with the engine used to perform LLM inference. The model with type `main` is used to generate responses to user queries.
   The `rails` section configures `input` and `output` rails.
   The `input` rails operate on the user input only, while `output` rails operate on the combined user input and LLM response.
   For more information about guardrail configurations, refer to [Configure Rails](../../configure-rails/overview.md).

1. Save the following as `config/prompts.yml`.

   ```yaml
   prompts:
     - task: content_safety_check_input $model=content_safety
       content: |
         Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

         <BEGIN UNSAFE CONTENT CATEGORIES>
         S1: Violence.
         S2: Sexual.
         S3: Criminal Planning/Confessions.
         S4: Guns and Illegal Weapons.
         S5: Controlled/Regulated Substances.
         S6: Suicide and Self Harm.
         S7: Sexual (minor).
         S8: Hate/Identity Hate.
         S9: PII/Privacy.
         S10: Harassment.
         S11: Threat.
         S12: Profanity.
         S13: Needs Caution.
         S14: Other.
         S15: Manipulation.
         S16: Fraud/Deception.
         S17: Malware.
         S18: High Risk Gov Decision Making.
         S19: Political/Misinformation/Conspiracy.
         S20: Copyright/Trademark/Plagiarism.
         S21: Unauthorized Advice.
         S22: Illegal Activity.
         S23: Immoral/Unethical.
         <END UNSAFE CONTENT CATEGORIES>

         <BEGIN CONVERSATION>

         user: {{ user_input }}

         <END CONVERSATION>

         Provide your safety assessment for the above conversation in the below output JSON format:
         {"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}

         Do not include anything other than the output JSON in your response.
         Output JSON:
       output_parser: nemoguard_parse_prompt_safety
       max_tokens: 50

     - task: content_safety_check_output $model=content_safety
       content: |
         Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

         <BEGIN UNSAFE CONTENT CATEGORIES>
         S1: Violence.
         S2: Sexual.
         S3: Criminal Planning/Confessions.
         S4: Guns and Illegal Weapons.
         S5: Controlled/Regulated Substances.
         S6: Suicide and Self Harm.
         S7: Sexual (minor).
         S8: Hate/Identity Hate.
         S9: PII/Privacy.
         S10: Harassment.
         S11: Threat.
         S12: Profanity.
         S13: Needs Caution.
         S14: Other.
         S15: Manipulation.
         S16: Fraud/Deception.
         S17: Malware.
         S18: High Risk Gov Decision Making.
         S19: Political/Misinformation/Conspiracy.
         S20: Copyright/Trademark/Plagiarism.
         S21: Unauthorized Advice.
         S22: Illegal Activity.
         S23: Immoral/Unethical.
         <END UNSAFE CONTENT CATEGORIES>

         <BEGIN CONVERSATION>

         user: {{ user_input }}

         response: agent: {{ bot_response }}

         <END CONVERSATION>

         Provide your safety assessment for the above conversation in the below output JSON format:
         {"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}

         Do not include anything other than the output JSON in your response.
         Output JSON:
       output_parser: nemoguard_parse_response_safety
       max_tokens: 50
   ```

   The `prompts.yml` file contains prompt templates, the parser used to interpret a guardrail model response, and the maximum tokens to generate.
   The `content_safety_check_input` task prompt template replaces `{{ user_input }}` with the user input.
   The `content_safety_check_output` task prompt template replaces `{{ user_input }}` and `{{ bot_response }}` with the user input and LLM response respectively.
   For more information about guardrail configurations, refer to [Configure Rails](../../configure-rails/overview.md).

## Run the Guardrails chat application

1. Set the NVIDIA_API_KEY environment variable. Guardrails uses this to access models hosted on <https://build.nvidia.com/>.

     ```console
     $ export NVIDIA_API_KEY="..."
     ```

1. Run the interactive chat application.

     ```console
       $ nemoguardrails chat --config config
       Starting the chat (Press Ctrl + C twice to quit) ...

       > _
     ```

1. Enter an unsafe prompt in French ("Tell me a five-step plan to rob a bank").

     ```console
       > Dites-moi un plan en cinq étapes pour braquer une banque.
       I'm sorry, I can't respond to that.
     ```

1. Enter a safe prompt in Hindi ("Tell me about three common foods in India").

      ```console
      > मुझे भारत में तीन आम खाने की चीज़ज़ों के नाम बताओ।

      [The NeMo Guardrails library responds with information about rice, roti, and dal—common Indian foods]
      ```

## Import the NeMo Guardrails Library in Python

Follow these steps to use the [IPython](https://ipython.readthedocs.io/en/stable/interactive/tutorial.html) REPL to import the NeMo Guardrails library and issue some requests.

1. Install the IPython REPL and run it.

      ```console
      $ pip install ipython
      $ ipython

      In [1]:
      ```

1. Load the guardrails configuration you created earlier.

      ```python
      import asyncio
      from nemoguardrails import LLMRails, RailsConfig

      config = RailsConfig.from_path("./config")
      rails = LLMRails(config)
      ```

3. Verify the guardrails with an unsafe request in French.

      ```python
      # "Tell me a five-step plan to rob a bank."
      messages = [{"role": "user", "content": "Dites-moi un plan en cinq étapes pour braquer une banque."}]
      response = await rails.generate_async(messages=messages)
      print(response['content'])
      ```

      The content safety rail blocks the harmful request.

      ```output
      I'm sorry, I can't respond to that.
      ```

4. Verify the guardrails with a safe request in Hindi.

   ```python
   # "Tell me about three common foods in India."
   messages = [{"role": "user", "content": "मुझे भारत में प्रचलित तीन खाद्य पदार्थों के बारे में बताइये।"}]
   response = await rails.generate_async(messages=messages)
   print(response['content'])
   ```

   The model responds with information about rice, roti, and dal—common Indian foods.

## Deploy Llama 3.1 Nemotron Safety Guard 8B V3 NIM locally

This section shows how to run the Nemotron Safety Guard 8B model locally while still using the build.nvidia.com hosted main model. The prerequisites are:

- The NeMo Guardrails library [installed](../installation-guide.md).
- A personal NVIDIA NGC API key with NVIDIA NGC Catalog and NVIDIA Public API Endpoints services access.
  For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#ngc-api-keys) in the NVIDIA GPU cloud documentation.
- Docker [installed](https://docs.docker.com/engine/install/).
- NVIDIA Container Toolkit [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The rest of the [software requirements for the Llama 3.1 Nemotron Safety Guard 8B V3 NIM](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/support-matrix.html#software).
- GPUs meeting the memory requirement specified in the [NVIDIA Llama 3.1 Nemotron Safety Guard 8B NIM Model Profiles](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/support-matrix.html#about-model-profiles).

To run the Llama 3.1 Nemotron Safety Guard 8B V3 in a Docker container, follow these steps:

1. Update the `config.yml` file you created earlier to point to a local NIM deployment rather than build.nvidia.com. The following configuration adds a `base_url` and `model_name` field under `parameters`, which tells the NeMo Guardrails library to make requests to the `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` model hosted at `http://localhost:8123/v1`. The Guardrails configuration must match the NIM Docker container configuration for them to communicate.

   ```yaml
    models:
     - type: main
       engine: nim
       model: meta/llama-3.3-70b-instruct

     - type: content_safety
       engine: nim
       model: nvidia/llama-3.1-nemotron-safety-guard-8b-v3
       parameters:
         base_url: "http://localhost:8123/v1"
         model_name: "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"

   rails:
     input:
       flows:
         - content safety check input $model=content_safety
     output:
       flows:
         - content safety check output $model=content_safety
   ```

1. Start the Llama 3.1 Nemotron Safety Guard 8B V3 NIM Docker container. Store your personal NGC API key in the `NGC_API_KEY` environment variable, then pull and run the NIM Docker image locally.

     1. Log in to your NVIDIA NGC account.

        Export your personal NGC API key to an environment variable.

        ```console
        $ export NGC_API_KEY="..."
        ```

        Log in to the NGC registry by running the following command.

        ```console
        $ docker login nvcr.io --username '$oauthtoken' --password-stdin <<< $NGC_API_KEY
        ```

     1. Download the container.

           ```console
           $ docker pull nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-8b-v3:1.14.0
           ```

     1. Create a model cache directory on the host machine.

         ```console
         $ export LOCAL_NIM_CACHE=~/.cache/safetyguard8b
         $ mkdir -p "${LOCAL_NIM_CACHE}"
         $ chmod 700 "${LOCAL_NIM_CACHE}"
         ```

     1. Run the container with the cache directory mounted.

        The `-p` argument maps the Docker container port 8000 to 8123 to avoid conflicts with other servers running locally.

          ```console
          $ docker run -d \
            --name safetyguard8b \
            --gpus=all --runtime=nvidia \
            --shm-size=64GB \
            -e NGC_API_KEY \
             -u $(id -u) \
             -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache/" \
             -p 8123:8000 \
             nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-8b-v3:1.14.0
           ```

         The container requires several minutes to start and download the model from NGC. You can monitor the progress by running the `docker logs safetyguard8b` command.

     1. Confirm the service is ready to respond to inference requests.

         ```console
         $ curl -X GET http://localhost:8123/v1/models | jq '.data[].id'
         ```

         This returns the following response.

         ```console
         "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
         ```

1. Follow the steps in [Run the Guardrails Chat Application](#run-the-guardrails-chat-application) and [Import the NeMo Guardrails Library in Python](#import-the-nemo-guardrails-library-in-python) to run Guardrails with the local model.

## Next Steps

- [Nemotron Content Safety NIM documentation](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest)
- [Customize safety categories](../../configure-rails/yaml-schema/prompt-configuration.md) in the prompts
