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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Restrict Topics with Llama 3.1 NemoGuard 8B TopicControl NIM

Learn how to restrict conversations to allowed topics using [Llama 3.1 NemoGuard 8B TopicControl NIM](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/index.html).

By following this tutorial, you learn how to configure a set of allowed topics and interact with both on-topic and off-topic requests.

## Prerequisites

- The NeMo Guardrails library [installed](../installation-guide.md) with the `nvidia` extra.
- A personal NVIDIA API key generated on <https://build.nvidia.com/>.

## Configure Guardrails

1. Create a configuration directory:

   ```console
   mkdir config
   ```

1. Create a `config/config.yml` file and add the following content.

   ```yaml
   models:
     - type: main
       engine: nim
       model: meta/llama-3.3-70b-instruct

     - type: topic_control
       engine: nim
       model: nvidia/llama-3.1-nemoguard-8b-topic-control

   rails:
     input:
       flows:
         - topic safety check input $model=topic_control
   ```

   The `config.yml` file contains the models used by Guardrails in the `models` section and `rails` controlling when to use these models.
   The `models` section configures the type and name of each model, along with the engine used to perform LLM inference. The model with type `main` is used to generate responses to user queries.
   The `rails` section configures `input` and `output` rails. Topic control only operates on user input, so there is no output rail flow.
   For more information about guardrail configurations, refer to [Configure Rails](../../configure-rails/overview.md).

1. Create a `config/prompts.yml` file with the topic control prompt template.

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

    You can customize the guidelines to match your specific use case and allowed topics. These guidelines are passed to the topic control model in the system prompt.
    The user request is placed in the user prompt.
    The topic control model responds with either `on-topic` or `off-topic` depending on whether the user input matches one of the topics in the prompt.

## Run the Guardrails chat application

1. Set the NVIDIA_API_KEY environment variable. Guardrails uses this to access models hosted on <https://build.nvidia.com/>.

     ```console
     $ export NVIDIA_API_KEY="..."
     ```

1. Run the interactive chat application.

     ```console
       $ nemoguardrails chat --config config
     ```

     ```terminaloutput
       Starting the chat (Press Ctrl + C twice to quit) ...

       > _
     ```

1. Enter an off-topic request.

    The prompt specifically instructs the model not to respond to questions about politics.
    The topic control input rail detects a policy violation and responds with the `I'm sorry, I can't respond to that.` refusal text.
    Because this input rail blocked the user input, an LLM response is not generated.

     ```console
       > Which party should I vote for in the next election?
       I'm sorry, I can't respond to that.
     ```

1. Enter an on-topic request.

     This request is in line with the topics in the prompt, so the topic control rail does not block the user input.
     The user input is passed to the Application LLM for generation.

      ```console
      > I'd like to cancel my subscription. Can I do this by phone or on the website?
      I'd be happy to help you with canceling your subscription. You have a couple of options to do so, and I'll walk you
      through them.

      [The NeMo Guardrails toolkit responds with instructions and information on subscription cancellations]
      ```

## Import the NeMo Guardrails Library in Python

Follow these steps to use the [IPython](https://ipython.readthedocs.io/en/stable/interactive/tutorial.html) REPL to import the NeMo Guardrails library and issue some requests:

1. Install the IPython REPL and run it to interpret the Python code below.

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

1. Verify the guardrails with an off-topic political question.

      ```python
      messages = [{"role": "user", "content": "Which party should I vote for in the next election?"}]
      response = await rails.generate_async(messages=messages)
      print(response['content'])
      ```

      The model blocks the Application LLM from generating a response.

      ```output
      "I'm sorry, I can't respond to that."
      ```

1. Verify the guardrails with an on-topic question.

      ```python
      messages = [{"role": "user", "content": "I'd like to cancel my subscription. Can I do this by phone or on the website?"}]
      response = await rails.generate_async(messages=messages)
      print(response['content'])
      ```

      The model responds with advice on how to cancel a subscription by phone or website.

## Deploy Llama 3.1 NemoGuard 8B TopicControl NIM Locally

This section shows how to run the NemoGuard 8B TopicControl model locally while still using the main model hosted on [build.nvidia.com](https://build.nvidia.com). The prerequisites are:

- The NeMo Guardrails library [installed](../installation-guide.md).
- A personal NVIDIA NGC API key with NVIDIA NGC Catalog and NVIDIA Public API Endpoints services access.
  For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#ngc-api-keys) in the NVIDIA GPU cloud documentation.
- Docker [installed](https://docs.docker.com/engine/install/).
- NVIDIA Container Toolkit [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- GPUs meeting the memory requirement specified in the [NVIDIA Llama 3.1 NemoGuard 8B TopicControl NIM Model Profiles](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/support-matrix.html#nvidia-llama-3-1-nemoguard-8b-topicguard-model-profiles).

To run the Llama 3.1 NemoGuard 8B TopicControl in a Docker container, follow these steps:

1. Update the `config.yml` file you created earlier to point to a local NIM deployment rather than build.nvidia.com. The following configuration adds a `base_url` and `model_name` field under `parameters`, which tells the NeMo Guardrails toolkit to make requests to the `nvidia/llama-3.1-nemoguard-8b-topic-control` model hosted at `http://localhost:8123/v1`. The Guardrails configuration must match the NIM Docker container configuration for them to communicate.

   ```yaml
    models:
     - type: main
       engine: nim
       model: meta/llama-3.3-70b-instruct

     - type: topic_control
       engine: nim
       model: nvidia/llama-3.1-nemoguard-8b-topic-control
       parameters:
         base_url: "http://localhost:8123/v1"
         model_name: "nvidia/llama-3.1-nemoguard-8b-topic-control"

   rails:
     input:
       flows:
         - topic safety check input $model=topic_control
   ```

1. Start the Llama 3.1 Topic Control NIM Docker container. Store your personal NGC API key in the `NGC_API_KEY` environment variable, then pull and run the NIM Docker image locally.

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
           $ docker pull nvcr.io/nim/nvidia/llama-3.1-nemoguard-8b-topic-control:1.10.1
           ```

     1. Create a model cache directory on the host machine.

         ```console
         $ export LOCAL_NIM_CACHE=~/.cache/llama-nemotron-topic-guard
         $ mkdir -p "${LOCAL_NIM_CACHE}"
         $ chmod 700 "${LOCAL_NIM_CACHE}"
         ```

     1. Run the container with the cache directory mounted.

        The `-p` argument maps the Docker container port 8000 to 8123 to avoid conflicts with other servers running locally.

          ```console
          $ docker run -d \
            --name llama-nemotron-topic-guard \
            --gpus=all --runtime=nvidia \
            --shm-size=64GB \
            -e NGC_API_KEY \
             -u $(id -u) \
             -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache/" \
             -p 8123:8000 \
             nvcr.io/nim/nvidia/llama-3.1-nemoguard-8b-topic-control:1.10.1
           ```

         The container requires several minutes to start and download the model from NGC. You can monitor the progress by running the `docker logs llama-nemotron-topic-guard` command.

     1. Confirm the service is ready to respond to inference requests.

         ```console
         $ curl -X GET http://localhost:8123/v1/health/ready
         ```

         This returns the following response.

         ```console
         {"object":"health-response","message":"ready"}
         ```

1. Follow the steps in [Run the Guardrails Chat Application](#run-the-guardrails-chat-application) and [Import the NeMo Guardrails Library in Python](#import-the-nemo-guardrails-library-in-python) to run Guardrails with the local model.

## Next Steps

- [Nemotron Safety models overview](../../configure-rails/yaml-schema/guardrails-configuration/built-in-guardrails.md#nvidia-models)
- [Topic safety example configuration](https://github.com/NVIDIA-NeMo/Guardrails/tree/develop/examples/configs/topic_safety)
- [Topic Control research paper (EMNLP 2024)](https://arxiv.org/abs/2404.03820)
- [NeMo Guardrails Toolkit Configuration Guide](../../configure-rails/overview.md)
