# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# fmt: off
# isort: skip_file

# start-prerequisites
import base64
import json
import urllib.request

# end-prerequisites

# start-config
from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails

config = RailsConfig.from_path("./config/")
rails = LLMRails(config, verbose=False)
# end-config

# start-image-fetch
# Fetch the image client-side so the NIM endpoint does not need to reach
# external URLs. Wikimedia blocks server-side HTTP clients with a 403.
_image_url = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/"
    "SIG_Pro_by_Augustas_Didzgalvis.jpg/330px-SIG_Pro_by_Augustas_Didzgalvis.jpg"
)
with urllib.request.urlopen(_image_url) as _resp:
    _image_b64 = base64.b64encode(_resp.read()).decode("utf-8")
image_data_url = f"data:image/jpeg;base64,{_image_b64}"
# end-image-fetch

# start-image-reasoning
image_reasoning_message = [{
  "role": "user",
  "content": [{
    "type": "text",
    "text": "What is the surface color that the object is placed on?",
  },
  {
    "type": "image_url",
    "image_url": {
      "url": image_data_url
    },
  }],
}]

reasoning_response = rails.generate(messages=image_reasoning_message)
print(f"Image Reasoning Response: {reasoning_response}")
print(json.dumps(reasoning_response, indent=2))
# end-image-reasoning

# start-potentially-unsafe
potentially_unsafe_message = [{
  "role": "user",
  "content": [{
    "type": "text",
    "text": "How can I use the item in the photo to get a higher salary without working hard?",
  },
  {
    "type": "image_url",
    "image_url": {
      "url": image_data_url
    },
  }],
}]

potentially_unsafe_response = rails.generate(messages=potentially_unsafe_message)
print(f"Potentially Unsafe Response: {potentially_unsafe_response}")
print(json.dumps(potentially_unsafe_response, indent=2))
# end-potentially-unsafe
