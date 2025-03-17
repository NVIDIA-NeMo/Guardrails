<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Multimodal Data with NeMo Guardrails

## About Working with Multimodal Data

NeMo Guardrails toolkit supports adding safety checks to multimodal content---images and text.
The support is for input and output guardrails only.
Depending on the image reasoning model, you can specify the image to check as a base64 encoded data or as a URL.

The safety check uses the image reasoning model as LLM as-a-judge to determine if the content is safe.
The OpenAI, Llama Vision, and Llama Guard models can accept multimodal input and act as a judge model.

You must ensure the image size and prompt length do not exceed the maximum context length of the model.

## Sample Configuration

1. Create a directory, such as `configs/content_safety_vision`, and add a `config.yml` file with the following content:

   ```{literalinclude} ../../examples/configs/content_safety_vision/config.yml
   :language: yaml
   ```

1. Add a `configs/content_safety_vision/prompts.yml` file with the following content:

   ```{literalinclude} ../../examples/configs/content_safety_vision/prompts.yml
   :language: yaml
   ```

## Example

The following sample code uses the preceding configuration and sends requests to OpenAI endpoints.
The sample image is a handgun.

1. Set the OpenAI environment variable with your token:

   ```console
   export OPENAI_API_KEY=<api-key>
   ```

1. Import required libraries:

   ```{literalinclude} ../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-prerequisites"
   :end-before: "# end-prerequisites"
   ```

1. Load the vision content safety configuration:

   ```{literalinclude} ../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-config"
   :end-before: "# end-config"
   ```

1. Send an image reasoning request:

   ```{literalinclude} ../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-image-reasoning"
   :end-before: "# end-image-reasoning"
   ```

1. Send a potentially unsafe request:

   ```{literalinclude} ../../examples/configs/content_safety_vision/demo.py
   :language: python
   :start-after: "# start-potentially-unsafe"
   :end-before: "# end-potentially-unsafe"
   ```

## Tips for Base 64 Encoded Images

Some models, such as the Llama Vision models, do not read an image from a URL.
For these models, encode the image in base 64 and provide the encoded image to the model.

The following code sample shows the common Python statements.

```{code-block} python
:emphasize-lines: 11, 23

import base64
import json

from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails

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
