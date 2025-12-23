---
title:
  page: "Restrict Topics with Nemotron Topic Control NIM"
  nav: "Restrict Topics"
description: "Restrict conversations to allowed topics using Nemotron Topic Control NIM."
topics: ["AI Safety", "Content Moderation"]
tags: ["Topic Control", "NIM", "Input Rails", "LoRA", "Docker", "Nemotron"]
content:
  type: "Tutorial"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

# Restrict Topics with Llama 3.1 NemoGuard 8B TopicControl NIM

Learn how to restrict conversations to allowed topics using [Llama 3.1 NemoGuard 8B TopicControl NIM](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/index.html).

By following this tutorial, you learn how to:

1. Deploy the Llama 3.1 NemoGuard 8B TopicControl NIM microservice to your local machine.
2. Configure topic control rails on a main LLM.
3. Restrict conversations to specific allowed topics.

## Prerequisites

- The NeMo Guardrails library [installed](../../getting-started/installation-guide.md).
- A personal NVIDIA NGC API key with NVIDIA NGC Catalog and NVIDIA Public API Endpoints services access.
  For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#ngc-api-keys) in the NVIDIA GPU cloud documentation.
- Docker [installed](https://docs.docker.com/engine/install/).
- NVIDIA Container Toolkit [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- GPUs meeting the memory requirement specified in the [NVIDIA Llama 3.1 NemoGuard 8B TopicControl NIM Model Profiles](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/support-matrix.html#nvidia-llama-3-1-nemoguard-8b-topicguard-model-profiles).

## Deploy the Llama 3.1 NemoGuard 8B TopicControl NIM Microservice

Follow the [getting started guide on deploying the Llama 3.1 NemoGuard 8B TopicControl NIM microservice](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/getting-started.html).

## Configure Guardrails

1. Create a `config/config.yaml` file and add the following content. This sets up the following:

   - OpenAI's `gpt-3.5-turbo-instruct` as the main LLM model
   - `llama-3.1-nemoguard-8b-topic-control` as the topic control model

   ```{code-block} yaml
   :emphasize-lines: 9-10

   models:
     - type: main
       engine: openai
       model: gpt-3.5-turbo-instruct

     - type: "topic_control"
       engine: nim
       parameters:
         base_url: "http://localhost:8123/v1"
         model_name: "llama-3.1-nemoguard-8b-topic-control"

   rails:
     input:
       flows:
         - topic safety check input $model=topic_control
   ```

   The following table explains the configuration parameters for the topic control model highlighted in the code above.

   | Parameter | Requirement |
   |-----------|-------------|
   | `base_url` | Must match the NIM host and port (8123 in this example) |
   | `model_name` | Must match `$MODEL_NAME` from the docker run command |

1. Create a `config/prompts.yml` file with the topic control prompt template:

    ```yaml
    prompts:
      - task: topic_safety_check_input $model=topic_control
        content: |
          You are to act as a customer service agent, providing users with factual information in accordance to the knowledge base. Your role is to ensure that you respond only to relevant queries and adhere to the following guidelines

          Guidelines for the user messages:
          - Do not answer questions related to personal opinions or advice on user's order, future recommendations
          - Do not provide any information on non-company products or services.
          - Do not answer enquiries unrelated to the company policies.
          - Do not answer questions asking for personal details about the agent or its creators.
          - Do not answer questions about sensitive topics related to politics, religion, or other sensitive subjects.
          - If a user asks topics irrelevant to the company's customer service relations, politely redirect the conversation or end the interaction.
          - Your responses should be professional, accurate, and compliant with customer relations guidelines, focusing solely on providing transparent, up-to-date information about the company that is already publicly available.
          - allow user comments that are related to small talk and chit-chat.
    ```

    Customize the guidelines to match your specific use case and allowed topics.

## Verify the Guardrails

1. Set your OpenAI API key for the main LLM:

   ```console
   export OPENAI_API_KEY=<your-openai-api-key>
   ```

1. Load the guardrails configuration:

   ```python
   import asyncio
   from nemoguardrails import LLMRails, RailsConfig

   config = RailsConfig.from_path("./config")
   rails = LLMRails(config)

   async def generate_response(messages):
       response = await rails.generate_async(messages=messages)
       return response
   ```

1. Verify the guardrails with an off-topic request:

   ```python
   messages = [{"role": "user", "content": "What is the best political party to vote for?"}]
   response = asyncio.run(generate_response(messages))
   print(response["content"])
   ```

   ```output
   I'm sorry, I can't respond to that.
   ```

   The topic control rail blocks the off-topic request about politics.

1. Verify the guardrails with an allowed request:

   ```python
   messages = [{"role": "user", "content": "What is your return policy?"}]
   response = asyncio.run(generate_response(messages))
   print(response["content"])
   ```

   The model responds normally with information about the return policy.

## (Optional) Cache TensorRT-LLM Engines

Cache the optimized TensorRT-LLM engines to avoid rebuilding them on each container start.

1. Create a cache directory.

   ```bash
   export LOCAL_NIM_CACHE=<path-to-cache-directory>
   mkdir -p $LOCAL_NIM_CACHE
   sudo chmod 666 $LOCAL_NIM_CACHE
   ```

1. Run the container with the cache mounted.

   ```bash
   docker run -it --name=$MODEL_NAME \
       --gpus=all --runtime=nvidia \
       -e NGC_API_KEY="$NGC_API_KEY" \
       -e NIM_SERVED_MODEL_NAME=$MODEL_NAME \
       -e NIM_CUSTOM_MODEL_NAME=$MODEL_NAME \
       -v $LOCAL_NIM_CACHE:/opt/nim/.cache/ \
       -u $(id -u) \
       -p 8123:8000 \
       $NIM_IMAGE
   ```

## Next Steps

- [Nemotron Safety models overview](../../configure-rails/yaml-schema/guardrails-configuration/built-in-guardrails.md#nvidia-models)
- [Topic safety example configuration](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/topic_safety)
- [Topic Control research paper (EMNLP 2024)](https://arxiv.org/abs/2404.03820)
