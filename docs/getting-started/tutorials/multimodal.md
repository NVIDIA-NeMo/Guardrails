---
title:
  page: "Add Multimodal Content Safety Using a Vision Model"
  nav: "Add Multimodal Content Safety"
description: "Add safety checks to images and text using a vision model as LLM-as-a-Judge."
topics: ["AI Safety", "Content Safety"]
tags: ["Multimodal", "Vision", "Images", "LLM-as-a-Judge", "OpenAI"]
content:
  type: "Tutorial"
  difficulty: "Intermediate"
  audience: ["Developer", "AI Engineer"]
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Add Multimodal Content Safety Using a Vision Model as LLM-as-a-Judge

Learn how to add safety checks to images and text using a vision model as LLM-as-a-Judge with [OpenAI GPT-4 Vision](https://platform.openai.com/docs/guides/vision), Llama Vision, or Llama Guard.

By following this tutorial, you learn how to:

1. Configure multimodal content safety rails for images and text.
2. Use a vision model as LLM-as-a-Judge to evaluate content safety.
3. Test with safe and unsafe image requests.

The NeMo Guardrails library supports multimodal content safety for input and output rails. You can provide images as base64-encoded data or URLs, depending on the model.

:::{important}
Ensure image size and prompt length do not exceed the model's maximum context length.
:::

## Prerequisites

- The NeMo Guardrails library [installed](../installation-guide.md) with the `openai` extra.
- A personal NVIDIA API key generated on <https://build.nvidia.com/>.

## Configure Guardrails

1. Create a configuration directory and add `config.yml`.

   ```{literalinclude} ../../../examples/configs/content_safety_vision/config.yml
   :language: yaml
   ```

1. Add `prompts.yml`.

   ```{literalinclude} ../../../examples/configs/content_safety_vision/prompts.yml
   :language: yaml
   ```

## Test with OpenAI

This example sends image requests to OpenAI endpoints and tests safety checks on a handgun image.

1. Set your OpenAI API key.

   ```console
   export OPENAI_API_KEY=<your-openai-api-key>
   ```

1. Install the IPython REPL and run it to interpret the Python code below.

      ```console
      $ pip install ipython
      $ ipython

      In [1]:
      ```

1. Import libraries.

   ```{literalinclude} ../../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-prerequisites"
   :end-before: "# end-prerequisites"
   ```

1. Load the configuration.

   ```{literalinclude} ../../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-config"
   :end-before: "# end-config"
   ```

1. Send a safe image reasoning request.

   ```{literalinclude} ../../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-image-reasoning"
   :end-before: "# end-image-reasoning"
   ```

1. Send an unsafe request.

   ```{literalinclude} ../../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-potentially-unsafe"
   :end-before: "# end-potentially-unsafe"
   ```

## Use Base64-Encoded Images

Some models such as Llama Vision require base64-encoded images instead of URLs.

```{code-block} python
import base64
import json

from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./content_safety_vision")
rails = LLMRails(config)

with open("<path-to-image>", "rb") as image_file:
  base64_image = base64.b64encode(image_file.read()).decode()

messages = [{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "what is the surface color that the object is placed on?",
    },
    {
      "type": "image_url",
      "image_url": {
          "url": f"data:image/jpeg;base64,{base64_image}"
      },
    },
  ],
}]

response = rails.generate(messages=messages)
print(json.dumps(response, indent=2))
```
