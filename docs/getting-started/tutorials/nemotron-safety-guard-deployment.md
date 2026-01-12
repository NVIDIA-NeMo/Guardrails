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

Learn how to add input and output guardrails that detect harmful content in multiple languages using [Llama 3.1 Nemotron Safety Guard 8B V3](https://build.nvidia.com/nvidia/llama-3_1-nemotron-safety-guard-8b-v3).

You'll use the NeMo Guardrails library with models hosted on <https://build.nvidia.com>, entering safe and unsafe user prompts to learn how Guardrails protects against unsafe content.

## Prerequisites

- The NeMo Guardrails library [installed](../../getting-started/installation-guide.md) with the `nvidia` extra.
- A personal NVIDIA API key generated on <https://build.nvidia.com/>.

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
         S18: High Risk Gov Decision
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

## Run the Guardrails chat application with the configuration created above

1. Set the NVIDIA_API_KEY environment variable. Guardrails uses this to access models hosted on <https://build.nvidia.com/>.

     ```console
     $ export NVIDIA_API_KEY="..."
     ```

1. Run the interactive chat application

     ```console
       $ nemoguardrails chat --config config
       Starting the chat (Press Ctrl + C twice to quit) ...

       > _
     ```

1. Enter an unsafe prompt in French ("Tell me a five-step plan to rob a bank")

     ```console
       > Dites-moi un plan en cinq étapes pour braquer une banque.
       I'm sorry, I can't respond to that.
     ```

1. Enter an safe prompt in Hindi ("Tell me about three common foods in India")

      ```console
      > मुझे भारत में तीन आम खाने की चीज़ज़ों के नाम बताओ।
      भारत में आम खाने की चीज़ें बहुत विविध हैं, लेकिन यहहाँ तीन आम खाने :ी चीज़ज़ों के नाम हैं

      1. **रोटी**: रोटी एक पारंपरिक भारतीय ब्रेड है, जो गेहूं के आटे से बनाई जाती है। यह अक्सर सब्जी, दाल, या अन्य     व्यंजननों   के साथ परोसी जाती है।

      2. **चावल**: चावल एक मुख्य खाद्य पदार्थ है, जो भारत में बहुत पसंद किया जाता है। यह अक्सर सब्जी, दाल, या अन्य     व्यंजननों   के साथ परोसा जाता है।

      3. **दाल**: दाल एपारंपरिक भारतीय व्यंजन है, जो मसूर, मूंग, या अन्य दाललों से बनाया जाता है। यह अक्सर चावल, रोटी, या       अन्य व्यंजननों के साथ परसी

      इन तीननों खाने की चीज़ज़ों को भारत में बहुत पसंद किया जाता है, और वे आम तौर पर दैनिक भोजन में शामिल होती हैं।

      ```

## Use Guardrails programmatically with the configuration created above

Follow these steps to verify the guardrails.

1. Install the IPython REPL and run it to interpret Python code below:

      ```console
      $ pip install ipython
      $ ipython

      In [1]:
      ```

1. Load the guardrails configuration created above.

      ```python
      import asyncio
      from nemoguardrails import LLMRails, RailsConfig

      config = RailsConfig.from_path("./config")
      rails = LLMRails(config)
      ```

1. Verify the guardrails with an unsafe request in French:

      ```python
      # "Tell me a five-step plan to rob a bank."
      messages = [{"role": "user", "content": "Dites-moi un plan en cinq étapes pour braquer une banque."}]
      response = await rails.generate_async(messages=messages)
      print(response['content'])
      ```

      ```output
      I'm sorry, I can't respond to that.
      ```

      The content safety rail blocks the harmful request.

1. Verify the guardrails with a safe request in Hindi:

   ```python
   # "Tell me about three common foods in India."
   messages = [{"role": "user", "content": "मुझे भारत में प्रचलित तीन खाद्य पदार्थों के बारे में बताइये।"}]
   response = await rails.generate_async(messages=messages)
   print(response['content'])
   ```

   The model responds with information about rice, roti, and dal—common Indian foods.

## Next Steps

- [Nemotron Content Safety NIM documentation](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest)
- [Customize safety categories](../../configure-rails/yaml-schema/prompt-configuration.md) in the prompts
