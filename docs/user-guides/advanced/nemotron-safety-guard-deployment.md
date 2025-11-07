<!--
  SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Nemotron Safety Guard Deployment

## Adding Multilingual Content Safety Guardrails

The following procedure adds a guardrail to check user input against a GPU-accelerated content safety model that can detect harmful content in several languages.

To simplify configuration, the sample code uses the [Llama 3.3 70B Instruct model](https://build.nvidia.com/meta/llama-3_3-70b-instruct) on build.nvidia.com as the application LLM.
This avoids deploying a NIM for LLMs instance locally for inference.

The sample code relies on starting a local instance of the
[Llama 3.1 Nemotron Safety Guard 8B V3](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemotron-safety-guard-8b-v3)
container that is available from NVIDIA NGC.

The steps guide you to start the content safety container, configure input and output content safety rails, and then use NeMo Guardrails interactively to send safe and unsafe requests.

## Prerequisites

- You have an NGC API key.
  This API key enables you to download the content safety container and model from NVIDIA NGC and to access models on build.nvidia.com.
  Refer to [Generating Your NGC API Key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key) in the _NVIDIA NGC User Guide_ for more information.

  When you create a personal API key, select at least **NGC Catalog** and **Public API Endpoints** from the **Services Included** menu.
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

- Refer to the [support matrix](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest/support-matrix.html) in the content safety NIM documentation for software requirements, hardware requirements, and model profiles.

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
   $ docker pull nvcr.io/nim/nvidia/llama-3.1-nemotron-safety-guard-8b-v3:1.14.0
   ```

1. Create a model cache directory on the host machine:

   ```console
   $ export LOCAL_NIM_CACHE=~/.cache/safetyguard
   $ mkdir -p "${LOCAL_NIM_CACHE}"
   $ chmod 700 "${LOCAL_NIM_CACHE}"
   ```

1. Run the container with the cache directory as a volume mount:

   ```console
   $ docker run -d \
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

   The container requires several minutes to start and download the model from NGC.
   You can monitor the progress by running the `docker logs safetyguard` command.

1. Confirm the service is ready to respond to inference requests:

   ```console
   $ curl -X GET http://localhost:8000/v1/models | jq '.data[].id'
   ```

   _Example Output_

   ```output
   "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
   ```

## Configuring Guardrails and Running Inference

1. Set your NGC API key as an environment variable.
   Guardrails uses this environment variable to send requests that pass the input rail to build.nvidia.com.

   ```console
   $ export NVIDIA_API_KEY=<nvapi-...>
   ```

1. Create a _configuration store_ directory, such as `config`.

1. Copy the following configuration code and save as `config/config.yml`:

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

   The `models` key in the `config.yml` file configures the application LLM model and the content safety model.

1. Copy the following prompts code and save as `config/prompts.yml`:

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

   ```text
   भारत में विभिन्न प्रकार के खाद्य पदार्थ प्रचलित हैं, जिनमें से तीन प्रमुख खाद्य पदार्थ हैं:

   1. **चावल**: चावल भारत में एक मुख्य खाद्य पदार्थ है, जो लगभग हर घर में खाया जाता है। यह एक प्रमुख अनाज है जो कार्बोहाइ रेट से भरपूर होता है और इसे विभिन्न प्रकार के  यंज ों में उपयोग किय

   जाता है, जैसे कि बिरयानी, पुलाव, और सादा चावल।

   2. **रोटी**: रोटी भारतीय आहार का एक अन्य महत्वपूर्ण हिस्सा है। यह गेहूं के आटे से बनाई जाती है और इसे विभिन्न प्रकार की सब्जि ों और दा ों के साथ परोसा जाता है। रोटी को तंदूर में या तवे प
   पकाया जा सकता है, और यह पूरे भारत में विभिन्न रू ों में पाई जाती है, जैसे कि नान, पराठा, और पूरी。

   3. **दाल**: दाल भारतीय  यंज ों में एक महत्वपूर्ण स्थान रखती है। यह मुख्य रूप से दा ों जैसे कि मूंग, चना, और तूर दाल से बनाई जाती है। दाल प्रोटीन से भरपूर होती है और इसे विभिन्न प्रकार के मस
    ों और सब्जि ों के साथ पकाया जाता है। दाल को चावल या रोटी के साथ परोसा जाता है और यह एक स्वस्थ और पौष्टिक विकल्प है।

   इन ती ों खाद्य पदा ों का महत्व भारतीय सं कृति और खान-पान में बहुत अधिक है, और वे लगभग हर भारतीय घर में नियमित रूप से खाए जाते हैं।

   ```

   Refer to the English translation:

   ```text
   India has a wide variety of foods, three of which are primarily:

   1. Rice: Rice is a staple food in India, eaten in almost every household. It is a staple grain that is rich in carbohydrates and is used in a variety of dishes, such as biryani, pulao, and plain rice.

   2. Roti: Roti is another important part of the Indian diet. It is made from wheat flour and served with a variety of vegetables and lentils. Roti can be cooked in a tandoor or on a griddle, and is found in various forms throughout India, such as naan, paratha, and puri.

   3. Dal: Dal plays an important role in Indian cuisine. It is made primarily from pulses such as moong, chana, and toor dal. Lentils are rich in protein and are cooked with a variety of spices and vegetables. Served with rice or roti, lentils are a healthy and nutritious option.

   These three foods hold immense significance in Indian culture and cuisine, and are eaten regularly in almost every Indian household.
   ```

## Next Steps

- Refer to the Llama 3.1 Nemotron Safety Guard 8B [documentation](https://docs.nvidia.com/nim/llama-3-1-nemotron-safety-guard-8b/latest).
