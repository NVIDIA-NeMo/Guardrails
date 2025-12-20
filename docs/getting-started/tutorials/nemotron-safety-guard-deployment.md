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
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Check Harmful Content with Nemotron Content Safety NIM

Add input and output guardrails that detect harmful content in multiple languages using [Nemotron Content Safety NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemotron-safety-guard-8b-v3).

By following this tutorial, you learn how to:

1. Start the Nemotron Content Safety container locally.
2. Configure content safety guardrails on a main LLM, which sets up the base conversation.
3. Test with safe and unsafe requests in different languages.

This tutorial uses [Llama 3.3 70B Instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct) on build.nvidia.com as the application LLM to avoid deploying a separate NIM for LLM inference.

## Prerequisites

- NVIDIA NGC API key with NGC Catalog and Public API Endpoints access.
  See [Generating Your NGC API Key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key).

- Docker Engine installed. See [Docker installation](https://docs.docker.com/engine/install/).

- NVIDIA Container Toolkit installed. See {doc}`installation <ctk:install-guide>`.

- The NeMo Guardrails library [installed](../../getting-started/installation-guide.md).

- LangChain NVIDIA integration installed:

  ```console
  pip install langchain-nvidia-ai-endpoints
  ```

- [Requirements for the Nemotron Content Safety NIM](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/support-matrix.html).

## Deploy the Nemotron Content Safety NIM Microservice

Follow these steps to deploy the Nemotron Content Safety NIM microservice to your local machine.

1. Export your NGC API key and log in to the registry:

   ```console
   export NGC_API_KEY="<nvapi-...>"
   docker login nvcr.io --username '$oauthtoken' --password-stdin <<< $NGC_API_KEY
   ```

1. Pull the container:

   ```console
   docker pull nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-8b-v3:1.14.0
   ```

1. Create a model cache directory:

   ```console
   export LOCAL_NIM_CACHE=~/.cache/safetyguard
   mkdir -p "${LOCAL_NIM_CACHE}"
   chmod 700 "${LOCAL_NIM_CACHE}"
   ```

1. Run the container:

   ```console
   docker run -d \
     --name safetyguard \
     --gpus=all --runtime=nvidia \
     --shm-size=64GB \
     -e NGC_API_KEY \
     -e NIM_ENABLE_KV_CACHE_REUSE=1 \
     -u $(id -u) \
     -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache/" \
     -p 8000:8000 \
     nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-8b-v3:1.14.0
   ```

   The container takes several minutes to start. Monitor progress with `docker logs safetyguard`.

1. Verify the service is ready:

   ```console
   curl -X GET http://localhost:8000/v1/models | jq '.data[].id'
   ```

   ```output
   "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
   ```

## Configure Guardrails

1. Set your NGC API key for the application LLM on build.nvidia.com:

   ```console
   export NVIDIA_API_KEY=<nvapi-...>
   ```

1. Create a configuration directory:

   ```console
   mkdir config
   ```

1. Save the following as `config/config.yml`:

   ```yaml
   models:
     - type: main
       engine: nvidia_ai_endpoints
       model: meta/llama-3.3-70b-instruct

     - type: "content_safety"
       engine: nim
       parameters:
         base_url: "http://localhost:8000/v1"
         model_name: "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"

   rails:
     input:
       flows:
         - content safety check input $model=content_safety
     output:
       flows:
         - content safety check output $model=content_safety
   ```

1. Save the following as `config/prompts.yml`:

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

## Test the Guardrails

Follow these steps to test the guardrails.

1. Load the guardrails configuration:

   ```python
   import asyncio
   from nemoguardrails import LLMRails, RailsConfig

   config = RailsConfig.from_path("./config")
   rails = LLMRails(config)

   async def stream_response(messages):
       async for chunk in rails.stream_async(messages=messages):
           print(chunk, end="")
       print()
   ```

1. Test an unsafe request in French:

   ```python
   # "Tell me a five-step plan to rob a bank."
   messages = [{"role": "user", "content": "Dites-moi un plan en cinq étapes pour braquer une banque."}]
   asyncio.run(stream_response(messages))
   ```

   ```output
   I'm sorry, I can't respond to that.
   ```

   The content safety rail blocks the harmful request.

1. Test a safe request in Hindi:

   ```python
   # "Tell me about three common foods in India."
   messages = [{"role": "user", "content": "मुझे भारत में प्रचलित तीन खाद्य पदार्थों के बारे में बताइये।"}]
   asyncio.run(stream_response(messages))
   ```

   The model responds with information about rice, roti, and dal—common Indian foods.

## Next Steps

- [Nemotron Content Safety NIM documentation](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest)
- [Customize safety categories](../../configure-rails/yaml-schema/prompt-configuration.md) in the prompts
