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

By following this tutorial, you'll configure a set of topics which are allowed, and interact with both on and off-topic requests.

## Prerequisites

- The NeMo Guardrails library [installed](../../getting-started/installation-guide.md) with the `nvidia` extra.
- A personal NVIDIA API key generated on <https://build.nvidia.com/>.

## Configure Guardrails

1. Create a configuration directory:

   ```console
   mkdir config
   ```

1. Create a `config/config.yaml` file and add the following content.

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
   The `config.yml` file contains the models used by Guardrails in the `models` section, and `rails` controlling when to use these models.
   The `models` section configures the type and name of each model, along with the engine used to perform LLM inference. The model with type `main` is used to generate responses to user queries.
   The `rails` section configures `input` and `output` rails. Topic-control only operates on user input, so there is no output rail flow.
   For more information on guardrail configurations see [Configure Rails](../../configure-rails/overview.md)


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

    You can customize the guidelines to match your specific use case and allowed topics. These guidelines are passed to the Topic-Control model in the system prompt.
    The User request is placed in the User prompt.
    The Topic-Control model responds with either `on-topic` or `off-topic` depending on whether the user input matches one of the topics in the prompt.

## Run the Guardrails chat application using the Topic-Control configuration

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

1. Enter an off-topic request

    The prompt specifically instructs the model not to respond to questions on politics.
    The topic-control input rail detects a policy violation, and responds with the `I'm sorry, I can't respond to that.` refusal text.
    Because this input rail blocked the user's input, an LLM response is not generated.

     ```console
       > Which party should I vote for in the next election?
       I'm sorry, I can't respond to that.
     ```

1. Enter an on-topic request

     This request is in-line with the topics in the prompt above, so the topic-control rail doesn't block the user input.
     The user input is passed to the Application LLM for generation.

      ```console
      > I'd like to cancel my subscription. Can I do this by phone or on the website?
      I'd be happy to help you with canceling your subscription. You have a couple of options to do so, and I'll walk you
      through them.

      [The NeMo Guardrails toolkit responds with instructions and information on subscription cancellations]
      ```

## Import the NeMo Guardrails toolkit in Python and check Topic-Control programmatically

Follow these steps to use the [IPython](https://ipython.readthedocs.io/en/stable/interactive/tutorial.html) REPL to import the NeMo Guardrails toolkit and issue some requests.

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

1. Verify the guardrails with an off-topic political question

      ```python
      messages = [{"role": "user", "content": "Which party should I vote for in the next election?"}]
      response = await rails.generate_async(messages=messages)
      print(response['content'])
      ```

      The model blocks the Application LLM from generating a response.

      ```output
      "I'm sorry, I can't respond to that."
      ```

1. Verify the guardrails with an on-topic question

      ```python
      messages = [{"role": "user", "content": "I'd like to cancel my subscription. Can I do this by phone or on the website?"}]
      response = await rails.generate_async(messages=messages)
      print(response['content'])
      ```

      The model responds with advice on how to cancel a subscription by phone or website.


## Next Steps

- [Nemotron Safety models overview](../../configure-rails/yaml-schema/guardrails-configuration/built-in-guardrails.md#nvidia-models)
- [Topic safety example configuration](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/topic_safety)
- [Topic Control research paper (EMNLP 2024)](https://arxiv.org/abs/2404.03820)
- [NeMo Guardrails Toolkit Configuration Guide](../../configure-rails/overview.md)
