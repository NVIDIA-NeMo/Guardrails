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

# Check Harmful Content with Llama 3.1 Nemotron Safety Guard 8B V3 NIM

Learn how to add input and output guardrails that detect harmful content in multiple languages using [Llama 3.1 Nemotron Safety Guard 8B V3 NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemotron-safety-guard-8b-v3).

By following this tutorial, you learn how to:

1. Deploy the Llama 3.1 Nemotron Safety Guard 8B V3 NIM microservice to your local machine.
2. Configure content safety guardrails on a main LLM. This tutorial uses [Llama 3.3 70B Instruct on build.nvidia.com](https://build.nvidia.com/meta/llama-3_3-70b-instruct) as the main LLM.
3. Verify the guardrails with safe and unsafe requests in various languages.

## Prerequisites

- The NeMo Guardrails library [installed](../../getting-started/installation-guide.md).
- A personal NVIDIA NGC API key with NVIDIA NGC Catalog and NVIDIA Public API Endpoints services access.
  For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#ngc-api-keys) in the NVIDIA GPU cloud documentation.
- Docker [installed](https://docs.docker.com/engine/install/).
- NVIDIA Container Toolkit [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
<!-- TODO: Is this really required? -->
- LangChain integration package [installed](https://pypi.org/project/langchain-nvidia-ai-endpoints/).
- The rest of the [software requirements for the Llama 3.1 Nemotron Safety Guard 8B V3 NIM](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/support-matrix.html#software).
- GPUs meeting the memory requirement specified in the [NVIDIA Llama 3.1 Nemotron Safety Guard 8B NIM Model Profiles](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/support-matrix.html#about-model-profiles).

## Deploy the Llama 3.1 Nemotron Safety Guard 8B V3 NIM Microservice

Follow the [getting started guide on deploying the Llama 3.1 Nemotron Safety Guard 8B V3 NIM microservice](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/getting-started.html).

## Configure Guardrails

Follow these steps to prepare the guardrails configuration.

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

## Verify the Guardrails

Follow these steps to verify the guardrails.

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

1. Verify the guardrails with an unsafe request in French:

   ```python
   # "Tell me a five-step plan to rob a bank."
   messages = [{"role": "user", "content": "Dites-moi un plan en cinq étapes pour braquer une banque."}]
   asyncio.run(stream_response(messages))
   ```

   ```output
   I'm sorry, I can't respond to that.
   ```

   The content safety rail blocks the harmful request.

1. Verify the guardrails with a safe request in Hindi:

   ```python
   # "Tell me about three common foods in India."
   messages = [{"role": "user", "content": "मुझे भारत में प्रचलित तीन खाद्य पदार्थों के बारे में बताइये।"}]
   asyncio.run(stream_response(messages))
   ```

   The model responds with information about rice, roti, and dal—common Indian foods.

## Next Steps

- [Nemotron Content Safety NIM documentation](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest)
- [Customize safety categories](../../configure-rails/yaml-schema/prompt-configuration.md) in the prompts
