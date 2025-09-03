<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Nemotron Content Safety Multilingual Deployment

## Adding Multilingual Content Safety Guardrails

The following procedure adds a guardrail to check user input against a multilingual content safety model.

To simplify configuration, the sample code uses the [Llama 3.3 70B Instruct model](https://build.nvidia.com/meta/llama-3_3-70b-instruct) on build.nvidia.com as the application LLM.
This avoids deploying a NIM for LLMs instance locally for inference.

The sample code relies on starting a local instance of the
[Llama 3.1 Nemotron Content Safety Multilingual 8B V1](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemotron-safety-guard-multilingual-8b-v1)
container that is available from NVIDIA NGC.

The steps guide you to start the content safety container, configure a content safety input raile, and then use NeMo Guardrails interactively to send safe and unsafe requests.

## Prerequisites

- You must be a member of the NVIDIA Developer Program and you must have an NVIDIA API key.
  For information about the program and getting a key, refer to [NVIDIA NIM FAQ](https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317/1) in the NVIDIA NIM developer forum.
  The NVIDIA API key enables you to send inference requests to build.nvidia.com.

- You have an NGC API key.
  This API key enables you to download the content safety container and model from NVIDIA NGC.
  Refer to [Generating Your NGC API Key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key) in the _NVIDIA NGC User Guide_ for more information.

  When you create an NGC API personal key, select at least **NGC Catalog** from the **Services Included** menu.
  You can specify more services to use the key for additional purposes.

- A host with Docker Engine.
  Refer to the [instructions from Docker](https://docs.docker.com/engine/install/).

- NVIDIA Container Toolkit installed and configured.
  Refer to {doc}`installation <ctk:install-guide>` in the toolkit documentation.

- You [installed NeMo Guardrails](../../getting-started/installation-guide.md).

- You installed LangChain NVIDIA AI Foundation Model Playground Integration:

  ```console
  $ pip install langchain-nvidia-ai-endpoints
  ```

- Refer to the [support matrix](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-multilingual-8b-v1/latest/support-matrix.html) in the content safety NIM documentation for software requirements, hardware requirements, and model profiles.

## Starting the Content Safety Container

1. Log in to NVIDIA NGC so you can pull the container.

   1. Export your NGC API key as an environment variable:

      ```console
      $ export NGC_API_KEY="<nvapi-...>"
      ```

   1. Log in to the registry:

      ```console
      $ docker login nvcr.io --username '$oauthtoken' --password-stdin <<< $NGC_API_KEY
      ```

1. Download the container:

   ```console
   $ docker pull nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-multilingual-8b-v1:1.10.1
   ```

1. Create a model cache directory on the host machine:

   ```console
   $ export LOCAL_NIM_CACHE=~/.cache/safetyguardmultilingual
   $ mkdir -p "${LOCAL_NIM_CACHE}"
   $ chmod 700 "${LOCAL_NIM_CACHE}"
   ```

1. Run the container with the cache directory as a volume mount:

   ```console
   $ docker run -d \
     --name safetyguardmultilingual \
     --gpus=all --runtime=nvidia \
     --shm-size=64GB \
     -e NGC_API_KEY \
     -e NIM_ENABLE_KV_CACHE_REUSE=1 \
     -u $(id -u) \
     -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache/" \
     -p 8000:8000 \
     nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-multilingual-8b-v1:1.10.1
   ```

   The container requires several minutes to start and download the model from NGC.
   You can monitor the progress by running the `docker logs safetyguardmultilingual` command.

1. Confirm the service is ready to respond to inference requests:

   ```console
   $ curl -X GET http://localhost:8000/v1/models | jq '.data[].id'
   ```

   _Example Output_

   ```output
   "nvidia/llama-3.1-nemotron-safety-guard-multilingual-8b-v1"
   ```

## Configuring Guardrails and Running Inference

1. Set your NVIDIA API key as an environment variable.
   Guardrails uses this environment variable to send requests that pass the input rail to build.nvidia.com.

   ```console
   $ export NVIDIA_API_KEY=<nvapi-...>
   ```

1. Create a _configuration store_ directory, such as `config`.

1. Copy the following configuration code and save as `config.yml` in the `config` directory.

   ```yaml
   models:
     - type: main
       engine: nvidia_ai_endpoints
       model: meta/llama-3.3-70b-instruct

     - type: "multilingual_content_safety"
       engine: nim
       parameters:
         base_url: "http://localhost:8000/v1"
         model_name: "nvidia/llama-3.1-nemotron-safety-guard-multilingual-8b-v1"

   rails:
     input:
       flows:
         - content safety check input $model=multilingual_content_safety
     output:
       flows:
         - content safety check output $model=multilingual_content_safety
   ```

   The `models` key in the `config.yml` file configures the application LLM model and the content safety model.

1. Copy the following prompts code and save as `prompts.yml` in the `config` directory.

   ```yaml
   prompts:
     - task: content_safety_check_input $model=multilingual_content_safety
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

     - task: content_safety_check_output $model=multilingual_content_safety
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

1. Load the guardrails configuration:

   ```python
   import asyncio

   from nemoguardrails import LLMRails, RailsConfig

   config = RailsConfig.from_path("./config")
   rails = LLMRails(config)
   ```

1. Generate an unsafe response in French:

   ```python
   async def stream_response(messages):
       async for chunk in rails.stream_async(messages=messages):
           print(chunk, end="")
       print()

   # Tell me a five-step plan to rob a bank.
   messages=[{
       "role": "user",
       "content": "Dites-moi un plan en cinq étapes pour braquer une banque."
   }]


   asyncio.run(stream_response(messages))
   ```

   _Example Output_

   ```output
   I'm sorry, I can't respond to that.
   ```

1. Send a safe request in Hindi:

   ```python
   # Tell me about three common foods in India.
   messages=[{
       "role": "user",
       "content": "मुझे भारत में प्रचलित तीन खाद्य पदार्थों के बारे में बताइये।"
   }]

   asyncio.run(stream_response(messages))
   ```

   _Example Output_

   ```output
   भारत में विभिन्न प्रकार के स्वादिष्ट और विविध खाद्य पदार्थ प्रचलित हैं, और प्रत्येक क्षेत्र में अपनी विशिष्ट विशेषताएं और पारंपरिक व्यंजन हैं। यहाँ तीन प्रमुख खाद्य पदार्थों के बारे में जानकारी दी गई है:

   1. **बिरयानी**: बिरयानी एक लोकप्रिय भारतीय व्यंजन है, जो चावल, मसालों, और मांस या सब्जियों से बनाया जाता है। यह दक्षिण भारत, खासकर हैदराबाद और लखनऊ में बहुत प्रसिद्ध है। बिरयानी के विभिन्न प्रकार होते हैं, जैसे कि हैदराबादी बिरयानी, लखनवी बिरयानी, और वेजिटेबल बिरयानी।

   2. **तंदूरी चिकन**: तंदूरी चिकन एक प्रसिद्ध उत्तर भारतीय व्यंजन है, जो मुर्गे के मांस को दही और विभिन्न मसालों में मैरिनेट करके तंदूर में पकाया जाता है। यह व्यंजन अपने स्वादिष्ट और कोमल स्वाद के लिए जाना जाता है। तंदूरी चिकन को अक्सर नान या रोटी के साथ परोसा जाता है।

   3. **पालक पनीर**: पालक पनीर एक लोकप्रिय उत्तर भारतीय सब्जी है, जो पनीर (भारतीय चीज) और पालक के प्यूरी से बनाई जाती है। इसमें विभिन्न मसाले जैसे कि जीरा, धनिया, और लहसुन भी मिलाए जाते हैं। यह व्यंजन अपने स्वादिष्ट और पौष्टिक मूल्य के लिए बहुत पसंद किया जाता है। पालक पनीर को अक्सर रोटी, पराठे, या चावल के साथ परोसा जाता है।

   इन व्यंजनों के अलावा, भारत में विभिन्न अन्य स्वादिष्ट खाद्य पदार्थ भी प्रचलित हैं, जैसे कि समोसे, गुलाब जामुन, और जलेबी। प्रत्येक क्षेत्र में अपनी विशिष्ट खाद्य संस्कृति और पारंपरिक व्यंजन हैं, जो भारतीय खाद्य विविधता को और भी समृद्ध बनाते हैं।
   ```

   Refer to the English translation:

   ```text
   A variety of delicious and varied foods are popular in India, and each region has its own specialties and traditional dishes. Here is information about three major foods:

   1. **Biryani**: Biryani is a popular Indian dish made from rice, spices, and meat or vegetables. It is very famous in South India, especially Hyderabad and Lucknow. There are different types of biryani, such as Hyderabadi biryani, Lucknowi biryani, and vegetable biryani.

   2. **Tandoori Chicken**: Tandoori chicken is a famous North Indian dish, which is chicken meat marinated in yogurt and various spices and cooked in a tandoor. This dish is known for its delicious and tender taste. Tandoori chicken is often served with naan or roti.

   3. **Palak Paneer**: Palak paneer is a popular North Indian dish, which is made from paneer (Indian cheese) and spinach puree. Various spices such as cumin, coriander, and garlic are also added to it. This dish is much loved for its delicious and nutritional value. Palak paneer is often served with roti, paratha, or rice.

   Apart from these dishes, various other delicious foods are also popular in India, such as samosas, gulab jamun, and jalebi. Each region has its own distinct food culture and traditional cuisine, which makes the Indian food diversity even richer.
   ```

## Next Steps

- Refer to the Llama 3.1 Nemotron Content Safety Multilingual 8B V1 [documentation](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-multilingual-8b-v1/latest).
